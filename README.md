
# RectiFill

## Prepare the dataset:

```bash
# Download the Librispeech dataset and prepare the training data for RectiFill.
uv run python -m src.data.prepare_dataset
```

## Train the model:

```bash
# Train the RectiFill model using the prepared dataset.
uv run python -m src.train
```
