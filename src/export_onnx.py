import argparse
import torch
from loguru import logger

from src.config.config import TrainConfig, DataConfig, ModelConfig, WandbConfig, MelConfig, TextConfig
from src.model.dit import DiTModel


class RectiFillONNX(torch.nn.Module):
    """
    A wrapper around the DiTModel to facilitate ONNX export.

    ONNX export works best with a fixed list of arguments rather than a **kwargs dictionary.
    This wrapper explicitly defines all required inputs for the forward pass and constructs
    the keyword arguments internally.

    :param model: The initialized DiTModel instance to be wrapped.
    :param context_type: The type of context used by the model ('phonemes' or 't5').
    """

    def __init__(self, model: torch.nn.Module, context_type: str = "phonemes"):
        super().__init__()
        self.model = model
        self.context_type = context_type

    def forward(
            self,
            xt: torch.Tensor,
            x_context: torch.Tensor,
            mask: torch.Tensor,
            t: torch.Tensor,
            mel_pad_mask: torch.Tensor,
            context_input: torch.Tensor,
            text_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Performs the forward pass of the wrapped DiT model.

        :param xt: The noisy mel-spectrogram tensor of shape (batch_size, mel_bins, time_frames).
        :param x_context: The contextual mel-spectrogram tensor of shape (batch_size, mel_bins, time_frames).
        :param mask: The mask tensor indicating the inpainting regions, shape (batch_size, 1, time_frames).
        :param t: The diffusion timestep tensor of shape (batch_size,).
        :param mel_pad_mask: The padding mask for the mel-spectrogram, shape (batch_size, time_frames).
        :param context_input: The context representation (phoneme IDs or T5 embeddings).
        :param text_mask: The padding mask for the text context, shape (batch_size, seq_len).

        :return: The predicted velocity field tensor.
        """
        kwargs = {
            "t": t,
            "mel_pad_mask": mel_pad_mask,
            "text_mask": text_mask,
            "cfg_drop_mask": None
        }

        if self.context_type == "phonemes":
            kwargs["phoneme_ids"] = context_input
        else:
            kwargs["text_emb"] = context_input

        return self.model(xt=xt, x_context=x_context, mask=mask, **kwargs)


def export_to_onnx(checkpoint_path: str, output_path: str):
    """
    Exports the DiT model from a PyTorch checkpoint to an ONNX format.

    :param checkpoint_path: The file path to the saved PyTorch Lightning checkpoint (.ckpt).
    :param output_path: The file path where the exported ONNX model will be saved.

    :raises ValueError: If the checkpoint format is unrecognized and misses state dicts.
    """
    device = "cpu"

    train_config = TrainConfig()
    context_type = train_config.model_params.context_type

    logger.info(f"Model context type: {context_type}")

    torch.serialization.add_safe_globals([
        TrainConfig, DataConfig, ModelConfig, WandbConfig, MelConfig, TextConfig
    ])

    model = DiTModel(train_config.model_params).to(device)

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    ema_sd = ckpt.get('ema_model_state_dict')
    if ema_sd is not None:
        state_dict = ema_sd
        logger.info("Loaded EMA weights.")
    else:
        raw = ckpt.get('state_dict')
        if raw is None:
            raise ValueError("Unrecognized checkpoint format. Missing state dict.")
        state_dict = {k.replace('model.', ''): v for k, v in raw.items() if k.startswith('model.')}
        logger.info("Loaded standard model weights.")

    model.load_state_dict(state_dict)
    model.eval()

    wrapped_model = RectiFillONNX(model, context_type)

    batch_size = 1
    mel_bins = train_config.model_params.mel_bins
    time_frames = 100
    seq_len = 50

    xt = torch.randn(batch_size, mel_bins, time_frames)
    x_context = torch.randn(batch_size, mel_bins, time_frames)
    mask = torch.zeros(batch_size, 1, time_frames, dtype=torch.float32)
    t = torch.tensor([0.5], dtype=torch.float32)
    mel_pad_mask = torch.zeros(batch_size, time_frames, dtype=torch.bool)
    text_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    if context_type == "phonemes":
        context_input = torch.randint(0, train_config.model_params.phoneme_vocab_size, (batch_size, seq_len),
                                      dtype=torch.long)
    else:
        context_input = torch.randn(batch_size, seq_len, train_config.model_params.text_dim)

    dummy_inputs = (xt, x_context, mask, t, mel_pad_mask, context_input, text_mask)

    dynamic_axes = {
        'xt': {0: 'batch_size', 2: 'time_frames'},
        'x_context': {0: 'batch_size', 2: 'time_frames'},
        'mask': {0: 'batch_size', 2: 'time_frames'},
        't': {0: 'batch_size'},
        'mel_pad_mask': {0: 'batch_size', 1: 'time_frames'},
        'context_input': {0: 'batch_size', 1: 'seq_len'},
        'text_mask': {0: 'batch_size', 1: 'seq_len'},
        'output_velocity': {0: 'batch_size', 2: 'time_frames'}
    }

    input_names = ['xt', 'x_context', 'mask', 't', 'mel_pad_mask', 'context_input', 'text_mask']
    output_names = ['output_velocity']

    logger.info("Exporting to ONNX...")

    torch.onnx.export(
        wrapped_model,
        dummy_inputs,
        output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )

    logger.success(f"Model successfully exported to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export DiT model to ONNX format.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the PyTorch Lightning .ckpt file.")
    parser.add_argument("--output", type=str, default="rectifill_model.onnx", help="Output path for the .onnx file.")
    args = parser.parse_args()

    export_to_onnx(args.checkpoint, args.output)