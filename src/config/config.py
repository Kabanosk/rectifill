import dataclasses
import pathlib


@dataclasses.dataclass
class MelConfig:
    """Configuration for Log-Mel-Spectrogram transformation."""
    sample_rate: int = 16000
    n_fft: int = 1024
    hop_length: int = 512
    n_mels: int = 128


@dataclasses.dataclass
class TextConfig:
    """Configuration for the text encoder."""
    model_name: str = "t5-base"


@dataclasses.dataclass
class DataConfig:
    """Main data configuration for the dataset and dataloaders."""
    data_path: str | pathlib.Path = "data"
    batch_size: int = 8
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = False
    pin_memory: bool = True

    mel_params: MelConfig = dataclasses.field(default_factory=MelConfig)
    text_params: TextConfig = dataclasses.field(default_factory=TextConfig)

    # E.g., 30 seconds of audio = (30 * 16000) / 512 = ~937 frames.
    max_mel_length: int = 1000


@dataclasses.dataclass
class WandbConfig:
    """ Configuration for the wandb project. """
    use_wandb: bool = True
    project_name: str = "rectifill"


@dataclasses.dataclass
class ModelConfig:
    # Architecture dimensions
    hidden_size: int = 384
    depth: int = 6
    num_heads: int = 12
    dropout: float = 0.2

    # Audio & Text
    mel_bins: int = 128
    text_dim: int = 768
    max_seq_len: int = 4000


@dataclasses.dataclass
class TrainConfig:
    """Configuration for the training process. """
    model_name: str = "rfm_dit"
    device: str = "cuda"
    checkpoint_path: str = "checkpoints/run_14_ema"
    log_interval: int = 100
    epochs: int = 50
    seed: int = 42

    learning_rate: float = 3e-4
    # for lr scheduler
    eta_min: float = 1e-6
    warmup_steps: int = 500

    weight_decay: float = 1e-2
    gradient_clip_val: float = 1.0
    accumulation_steps: int = 8  # for gradient accumulation

    validation_metrics_steps: int = 5

    # CFG Parameters
    cfg_prob: float = 0.1   # Probability of dropping text condition during training
    cfg_scale: float = 3.0  # Guidance scale for validation and inference

    # EMA Parameters
    use_ema: bool = True
    ema_decay: float = 0.9999
    ema_update_every: int = 1

    model_params: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    wandb_params: WandbConfig = dataclasses.field(default_factory=WandbConfig)
