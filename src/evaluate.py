import argparse
import time
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from tqdm import tqdm

from src.config.config import DataConfig, MelConfig, ModelConfig, TextConfig, TrainConfig, WandbConfig
from src.data.dataset import get_dataloader
from src.data.utils import denormalize_mel, mel_to_waveform, normalize_mel, save_wav, UniversalMasker
from src.evaluation.asr import HuBERTEvaluator
from src.evaluation.metrics import calculate_lsd, calculate_speech_metrics
from src.model.dit import DiTModel
from src.utils.helpers import set_seed
from src.utils.rfm import sample_euler


class ONNXModelWrapper:
    def __init__(self, onnx_path: str):
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        providers = ['CUDAExecutionProvider',
                     'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available_providers else [
            'CPUExecutionProvider']

        self.session = ort.InferenceSession(onnx_path, providers=providers)
        active_provider = self.session.get_providers()[0]
        logger.info(f"Loaded ONNX Model on: {active_provider}")

    def __call__(self, xt: torch.Tensor, x_context: torch.Tensor, mask: torch.Tensor, t: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        context_input = kwargs['phoneme_ids'] if 'phoneme_ids' in kwargs else kwargs['text_emb']
        cfg_drop_mask = kwargs.get('cfg_drop_mask')

        if cfg_drop_mask is None:
            cfg_drop_mask = torch.zeros(xt.shape[0], 1, 1, dtype=torch.bool, device=xt.device)

        inputs = {
            "xt": xt.cpu().numpy(),
            "x_context": x_context.cpu().numpy(),
            "mask": mask.cpu().numpy(),
            "t": t.cpu().numpy(),
            "mel_pad_mask": kwargs['mel_pad_mask'].cpu().numpy(),
            "context_input": context_input.cpu().numpy(),
            "text_mask": kwargs['text_mask'].cpu().numpy(),
            "cfg_drop_mask": cfg_drop_mask.cpu().numpy()
        }

        outputs = self.session.run(["output_velocity"], inputs)
        return torch.from_numpy(outputs[0]).to(xt.device)


@torch.no_grad()
def evaluate(ckpt_path: str, onnx_path: str, data_path: str, output_dir: str, device: str, min_mask: int,
             max_mask: int):
    output_dir_path = Path(output_dir)
    samples_dir = output_dir_path / ("best_samples_onnx" if onnx_path else "best_samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    train_config = TrainConfig()
    data_config = DataConfig(batch_size=1, shuffle=False, data_path=data_path, drop_last=False)
    test_loader = get_dataloader(data_config)

    test_loader.dataset.mask_generator = UniversalMasker(
        p_tts=0.0, p_continuation=0.0, p_prefix=0.0, p_inpainting=1.0,
        min_tokens_inpaint=min_mask, max_tokens_inpaint=max_mask
    )

    set_seed(train_config.seed)

    if onnx_path:
        logger.info(f"Evaluating using ONNX model: {onnx_path}")
        model = ONNXModelWrapper(onnx_path)
    else:
        logger.info(f"Evaluating using PyTorch checkpoint: {ckpt_path}")
        model = DiTModel(train_config.model_params).to(device)

        torch.serialization.add_safe_globals([TrainConfig, DataConfig, ModelConfig, WandbConfig, MelConfig, TextConfig])
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)

        ema_sd = ckpt.get('ema_model_state_dict')
        state_dict = ema_sd if ema_sd is not None else {k.replace('model.', ''): v for k, v in
                                                        ckpt.get('state_dict').items() if k.startswith('model.')}
        model.load_state_dict(state_dict)
        model.eval()

    asr_evaluator = HuBERTEvaluator(device=device)
    sr = data_config.mel_params.sample_rate

    stats = {k: [] for k in ["lsd", "pesq", "stoi", "wer", "gen_time", "audio_len", "rtf"]}
    best_scores = {"lsd": float('inf'), "pesq": float('-inf'), "stoi": float('-inf'), "wer": float('inf')}

    for batch in tqdm(test_loader, desc="Evaluation"):
        mel_raw = batch['mel'].squeeze(1).to(device)
        mel_norm = normalize_mel(mel_raw)
        mask_bool = batch['inpainting_mask'].to(device)
        mel_pad_mask = batch['mel_padding_mask'].to(device)

        condition_kwargs = {
            "text_mask": batch['text_padding_mask'].to(device),
            "mel_pad_mask": mel_pad_mask
        }
        if 'embedding' in batch:
            condition_kwargs['text_emb'] = batch['embedding'].to(device)
        elif 'phoneme_ids' in batch:
            condition_kwargs['phoneme_ids'] = batch['phoneme_ids'].to(device)

        t0 = time.perf_counter()

        gen_mel_norm = sample_euler(
            model=model, x1_context=mel_norm, mask_bool=mask_bool, num_steps=50,
            cfg_scale=train_config.cfg_scale, verbose=False, **condition_kwargs
        )
        gen_mel_db = denormalize_mel(gen_mel_norm)
        gen_time = time.perf_counter() - t0

        valid_len = (~mel_pad_mask[0]).sum().item()

        mel_raw_b = mel_raw[0, :, :valid_len]
        gen_mel_db_b = gen_mel_db[0, :, :valid_len]
        mask_bool_b = mask_bool[0, :, :valid_len]

        target_wav = mel_to_waveform(mel_raw_b)
        pred_wav = mel_to_waveform(gen_mel_db_b)

        lsd = calculate_lsd(gen_mel_db_b.unsqueeze(0), mel_raw_b.unsqueeze(0), mask_bool_b.unsqueeze(0))

        try:
            speech = calculate_speech_metrics(pred_wav, target_wav, sr)
            pesq_val, stoi_val = speech["pesq"], speech["stoi"]
        except Exception as e:
            logger.warning(f"Metric error: {e}. Skipping sample scores.")
            pesq_val, stoi_val = -0.5, 0.0

        try:
            pred_text = asr_evaluator.transcribe(pred_wav)
            wer = asr_evaluator.calculate_wer(batch['text'][0], pred_text)
        except Exception as e:
            logger.error(f"ASR error: {e}")
            wer = 1.0

        duration = pred_wav.shape[-1] / sr

        stats["lsd"].append(lsd)
        stats["pesq"].append(pesq_val)
        stats["stoi"].append(stoi_val)
        stats["wer"].append(wer)
        stats["gen_time"].append(gen_time)
        stats["audio_len"].append(duration)
        stats["rtf"].append(gen_time / duration)

        curr = {"lsd": lsd, "pesq": pesq_val, "stoi": stoi_val, "wer": wer}
        for m, val in curr.items():
            if (m in ["lsd", "wer"] and val < best_scores[m]) or (m in ["pesq", "stoi"] and val > best_scores[m]):
                best_scores[m] = val
                save_wav(samples_dir / f"best_{m}_pred.wav", pred_wav.unsqueeze(0), sr)
                save_wav(samples_dir / f"best_{m}_target.wav", target_wav.unsqueeze(0), sr)

    logger.success("\nEvaluation Summary:")
    for k, v in stats.items():
        unit = "s" if "time" in k or "len" in k else ""
        logger.info(f"{k.upper():<10}: {float(np.mean(v)):.4f}{unit} +/- {float(np.std(v)):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--ckpt_model", type=str, default=None)
    parser.add_argument("-x", "--onnx_model", type=str, default=None)
    parser.add_argument("-d", "--data_path", type=str, default="data/processed/test-clean")
    parser.add_argument("-o", "--output_dir", type=str, default="evaluation_results")
    parser.add_argument("--min_mask", type=int, default=5)
    parser.add_argument("--max_mask", type=int, default=10)
    args = parser.parse_args()

    if not args.ckpt_model and not args.onnx_model:
        raise ValueError("You must provide either --ckpt_model (-c) or --onnx_model (-x)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate(args.ckpt_model, args.onnx_model, args.data_path, args.output_dir, device, args.min_mask, args.max_mask)
