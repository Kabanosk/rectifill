import torch
import torchaudio
import jiwer
from loguru import logger


class HuBERTEvaluator:
    def __init__(self, device: str | torch.device):
        self.device = device
        logger.info("Loading HuBERT-Large ASR model for WER evaluation...")

        self.bundle = torchaudio.pipelines.HUBERT_ASR_LARGE
        self.model = self.bundle.get_model().to(device)
        self.model.eval()

        self.labels = self.bundle.get_labels()
        self.blank_id = {c: i for i, c in enumerate(self.labels)}.get("-", 0)

        self.transformation = jiwer.Compose([
            jiwer.ToUpperCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ])

    @torch.no_grad()
    def transcribe(self, waveform: torch.Tensor) -> str:
        """Transcribes a 16kHz audio waveform using Greedy CTC decoding."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        waveform = waveform.to(self.device)
        emissions, _ = self.model(waveform)

        indices = torch.argmax(emissions[0], dim=-1)

        tokens = []
        prev_idx = -1
        for idx in indices:
            idx = idx.item()
            if idx != prev_idx and idx != self.blank_id:
                tokens.append(idx)
            prev_idx = idx

        transcript = "".join([self.labels[t] for t in tokens]).replace("|", " ").strip()
        return transcript

    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """Calculates WER with strict text normalization."""
        ref_clean = self.transformation(reference)
        hyp_clean = self.transformation(hypothesis)

        if not ref_clean:
            return 1.0 if hyp_clean else 0.0

        return jiwer.wer(ref_clean, hyp_clean)
