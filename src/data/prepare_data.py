import random
from pathlib import Path
from typing import Optional


import torchaudio
from tqdm import tqdm

from .utils import load_wav, save_wav


def load_transcripts(source_dir: Path) -> dict[str, str]:
    """
    Load all transcripts from LibriSpeech .trans.txt files.
    Returns a dict mapping file_id (e.g. '211-122425-0000') to transcript text.
    """
    transcripts = {}
    trans_files = list(source_dir.rglob("*.trans.txt"))
    for trans_file in trans_files:
        with open(trans_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) == 2:
                    file_id, text = parts
                    transcripts[file_id] = text
    return transcripts


def download_librispeech(target_dir: Path, split: str = "train-clean-100") -> None:
    """
    Download LibriSpeech dataset using torchaudio.
    :param target_dir: Directory to download and extract data
    :param split: Which split to download (e.g. 'train-clean-100', 'dev-clean', 'test-clean')
    """
    print(f"Downloading LibriSpeech split '{split}' to {target_dir} ...")
    torchaudio.datasets.LIBRISPEECH(str(target_dir), url=split, download=True)
    print(f"LibriSpeech '{split}' downloaded to {target_dir}")


def resample_wav(input_dir: Path, output_dir: Path, sample_rate: int = 16000, transcripts: dict[str, str] | None = None) -> None:
    """Resample all WAV files in input_dir to output_dir at sample_rate and save transcripts."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_files = list(input_dir.rglob("*.flac"))
    for wav_path in tqdm(wav_files, desc="Resampling audio"):
        wav = load_wav(wav_path, sample_rate)
        out_path = output_dir / wav_path.name
        save_wav(out_path, wav, sample_rate)
        # Save transcript as .txt file with the same name
        if transcripts is not None:
            file_id = wav_path.stem  # e.g., '211-122425-0000'
            if file_id in transcripts:
                txt_path = output_dir / f"{file_id}.txt"
                txt_path.write_text(transcripts[file_id], encoding="utf-8")
    print(f"Resampled {len(wav_files)} files to {output_dir}")

def split_train_val(dataset_dir: Path, val_ratio: float = 0.1, seed: int = 42) -> None:
    """Split WAV files and their transcripts in dataset_dir into train/val subfolders."""
    dataset_dir = Path(dataset_dir)
    # Only get flac files directly in dataset_dir, not in subdirectories
    wav_files = sorted([f for f in dataset_dir.glob("*.flac")])
    print(len(wav_files))
    random.seed(seed)
    random.shuffle(wav_files)
    n_val = int(len(wav_files) * val_ratio)
    val_files = wav_files[:n_val]
    train_files = wav_files[n_val:]
    (dataset_dir / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "val").mkdir(parents=True, exist_ok=True)
    for f in tqdm(train_files, desc="Copying train"):
        # Copy audio file
        (dataset_dir / "train" / f.name).write_bytes(f.read_bytes())
        # Copy corresponding transcript file
        txt_file = f.with_suffix(".txt")
        if txt_file.exists():
            (dataset_dir / "train" / txt_file.name).write_text(txt_file.read_text(encoding="utf-8"), encoding="utf-8")
    for f in tqdm(val_files, desc="Copying val"):
        # Copy audio file
        (dataset_dir / "val" / f.name).write_bytes(f.read_bytes())
        # Copy corresponding transcript file
        txt_file = f.with_suffix(".txt")
        if txt_file.exists():
            (dataset_dir / "val" / txt_file.name).write_text(txt_file.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"Split {len(wav_files)} files: {len(train_files)} train, {len(val_files)} val")


def split_train_val_test(dataset_dir: Path, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> None:
    """Split WAV files and their transcripts in dataset_dir into train/val/test subfolders."""
    dataset_dir = Path(dataset_dir)
    # Only get flac files directly in dataset_dir, not in subdirectories
    wav_files = sorted([f for f in dataset_dir.glob("*.flac")])
    print(len(wav_files))
    random.seed(seed)
    random.shuffle(wav_files)
    
    n_test = int(len(wav_files) * test_ratio)
    n_val = int(len(wav_files) * val_ratio)
    
    test_files = wav_files[:n_test]
    val_files = wav_files[n_test:n_test + n_val]
    train_files = wav_files[n_test + n_val:]
    
    (dataset_dir / "train").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "val").mkdir(parents=True, exist_ok=True)
    (dataset_dir / "test").mkdir(parents=True, exist_ok=True)
    
    for f in tqdm(train_files, desc="Copying train"):
        # Copy audio file
        (dataset_dir / "train" / f.name).write_bytes(f.read_bytes())
        # Copy corresponding transcript file
        txt_file = f.with_suffix(".txt")
        if txt_file.exists():
            (dataset_dir / "train" / txt_file.name).write_text(txt_file.read_text(encoding="utf-8"), encoding="utf-8")
    
    for f in tqdm(val_files, desc="Copying val"):
        # Copy audio file
        (dataset_dir / "val" / f.name).write_bytes(f.read_bytes())
        # Copy corresponding transcript file
        txt_file = f.with_suffix(".txt")
        if txt_file.exists():
            (dataset_dir / "val" / txt_file.name).write_text(txt_file.read_text(encoding="utf-8"), encoding="utf-8")
    
    for f in tqdm(test_files, desc="Copying test"):
        # Copy audio file
        (dataset_dir / "test" / f.name).write_bytes(f.read_bytes())
        # Copy corresponding transcript file
        txt_file = f.with_suffix(".txt")
        if txt_file.exists():
            (dataset_dir / "test" / txt_file.name).write_text(txt_file.read_text(encoding="utf-8"), encoding="utf-8")
    
    print(f"Split {len(wav_files)} files: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")


def copy_to_test(dataset_dir: Path) -> None:
    """Copy all WAV files and transcripts in dataset_dir to test subfolder (for dedicated test sets)."""
    dataset_dir = Path(dataset_dir)
    wav_files = sorted([f for f in dataset_dir.glob("*.flac")])
    print(f"Found {len(wav_files)} files for test set")
    
    (dataset_dir / "test").mkdir(parents=True, exist_ok=True)
    
    for f in tqdm(wav_files, desc="Copying test"):
        # Copy audio file
        (dataset_dir / "test" / f.name).write_bytes(f.read_bytes())
        # Copy corresponding transcript file
        txt_file = f.with_suffix(".txt")
        if txt_file.exists():
            (dataset_dir / "test" / txt_file.name).write_text(txt_file.read_text(encoding="utf-8"), encoding="utf-8")
    
    print(f"Copied {len(wav_files)} files to test/")


def prepare_dataset(source_dir: Path, target_dir: Path, sample_rate: int = 16000, val_ratio: float = 0.1, test_ratio: float = 0.0, librispeech: bool = False, librispeech_split: str = "train-clean-100") -> None:
    """
    Run full pipeline: download (optional), resample, split train/val/test.
    :param source_dir: Directory with raw or intermediate dataset
    :param target_dir: Directory to write the final prepared dataset
    :param sample_rate: Target sampling rate for audio files
    :param val_ratio: Validation split ratio
    :param test_ratio: Test split ratio (if 0, no test split)
    :param librispeech: If True, download LibriSpeech to source_dir first
    :param librispeech_split: Which LibriSpeech split to download
    """
    source_dir.mkdir(parents=True, exist_ok=True)
    if librispeech:
        download_librispeech(source_dir, librispeech_split)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    # Load transcripts from source directory
    transcripts = load_transcripts(source_dir)
    print(f"Loaded {len(transcripts)} transcripts")
    resample_wav(source_dir, target_dir, sample_rate, transcripts)
    if test_ratio > 0:
        split_train_val_test(target_dir, val_ratio, test_ratio)
    else:
        split_train_val(target_dir, val_ratio)
    print(f"Prepared dataset in {target_dir}")


def prepare_test_dataset(source_dir: Path, target_dir: Path, sample_rate: int = 16000, librispeech: bool = False, librispeech_split: str = "test-clean") -> None:
    """
    Prepare a dedicated test dataset (no train/val split, all files go to test/).
    :param source_dir: Directory with raw or intermediate dataset
    :param target_dir: Directory to write the final prepared dataset
    :param sample_rate: Target sampling rate for audio files
    :param librispeech: If True, download LibriSpeech to source_dir first
    :param librispeech_split: Which LibriSpeech split to download
    """
    source_dir.mkdir(parents=True, exist_ok=True)
    if librispeech:
        download_librispeech(source_dir, librispeech_split)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    # Load transcripts from source directory
    transcripts = load_transcripts(source_dir)
    print(f"Loaded {len(transcripts)} transcripts")
    resample_wav(source_dir, target_dir, sample_rate, transcripts)
    copy_to_test(target_dir)
    print(f"Prepared test dataset in {target_dir}")


if __name__ == "__main__":
    prepare_dataset(Path("data/raw"), Path("data/processed"), sample_rate=16000, val_ratio=0.1, librispeech=True, librispeech_split="train-clean-100")
    prepare_dataset(Path("data/raw_val"), Path("data/processed_val"), sample_rate=16000, val_ratio=0.1, librispeech=True, librispeech_split="dev-clean")
    prepare_test_dataset(Path("data/raw_test"), Path("data/processed_test"), sample_rate=16000, librispeech=True, librispeech_split="test-clean")
    print("Data preparation complete.")