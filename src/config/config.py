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
    data_path: str | pathlib.Path
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = False

    mel_params: MelConfig = dataclasses.field(default_factory=MelConfig)
    text_params: TextConfig = dataclasses.field(default_factory=TextConfig)

    # E.g., 30 seconds of audio = (30 * 16000) / 512 = ~937 frames.
    max_mel_length: int = 1000


@dataclasses.dataclass
class TrainConfig:
    """Configuration for the training process."""
    epochs: int = 10
    learning_rate: float = 3e-4
    device: str = "cuda"
    checkpoint_path: str = "checkpoints"
    log_interval: int = 10
