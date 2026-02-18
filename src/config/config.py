import dataclasses
import pathlib


@dataclasses.dataclass
class DataConfig:
    data_path: str | pathlib.Path
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    drop_last: bool = False
    max_audio_length: int = 16000 * 300  # Max length of audio in samples (e.g., 300 seconds at 16kHz)


@dataclasses.dataclass
class TrainConfig:
    epochs: int = 10
    learning_rate: float = 3e-4
    device: str = "cuda"
    checkpoint_path: str = "checkpoints"
    log_interval: int = 10
