# Donut Fine-tuning Template

This project fine-tunes [NAVER CLOVA Donut (`naver-clova-ix/donut-base`)](https://huggingface.co/naver-clova-ix/donut-base) on a custom dataset of clothing labels. The pipeline converts the raw Excel annotations to JSONL, splits the data, caches images on demand, and trains with Hugging Face's `Seq2SeqTrainer`.

## Environment Setup

### Prerequisites
- NVIDIA GPU with CUDA support (recommended for training)
- CUDA 11.8 or 12.1 compatible drivers
- Miniconda or Anaconda

### Installation Steps

1. **Create conda environment with Python 3.10 or 3.11:**
   ```bash
   conda create -n donut python=3.10 -y
   conda activate donut
   ```

2. **Install GPU-enabled PyTorch (CUDA 11.8 version):**
   ```bash
   pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
   ```

3. **Install remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify GPU availability:**
   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
   ```


> **Note:** Training the Donut model requires a GPU for practical runtimes. Set `train.fp16` to `false` in `config.json` if you must train on CPU. CPU training can be 100-1000x slower than GPU training.

## Data Preparation

1. **Transform** the master Excel sheet into Donut-style JSONL:
   ```bash
   python data_scripts/transform_data.py
   ```
   The script reads `data/raw_data/exported_raw_data.xlsx` and writes `data/raw_data/donut_finetune_dataset.jsonl` with a structure of `{ "input": {...}, "output": {...} }` per line.

2. **Split** the processed JSONL into train/validation/test:
   ```bash
   python data_scripts/divide_data.py --config data_scripts/divide_config.json
   ```
   The command produces `data/train_dataset.jsonl`, `data/validation_dataset.jsonl`, `data/test_dataset.jsonl`, and a `data/data_split_summary.json` log.

3. **Images**: Each JSONL row references either `input.image_path` (local file) or `input.image_url` (remote). When only URLs are provided, the trainer downloads and caches images under `data/image_cache/`. Make sure you have network access during the first epoch.

## Configuration (`config.json`)

Key fields:

- `model_name`: Hugging Face ID or local path for the pretrained Donut weights.
- `output_root`: Directory where training runs are logged (e.g., `results/run_YYYYMMDD_HHMMSS/`).
- `seed`: Global seed for data shuffling and Trainer.
- `tokenizer.task_prompt`: Optional prefix inserted before the JSON target. Leave empty unless you add prompt engineering.
- `data.*`: Paths to the split JSONL files, optional `image_root`, required `image_cache_dir` for remote images, and `max_target_length` to cap decoder sequence length.
- `train.*`: Hugging Face `Seq2SeqTrainingArguments`. Adjust batch sizes, learning rate, evaluation/save cadence, etc. `metric_for_best_model` defaults to the exact-match score computed on the validation set.
- `generation.*`: Defaults used both during validation (`predict_with_generate=True`) and for interactive inference.

## Training

```bash
python src/train.py --config config.json
```

### Main Training Pipeline

1. `src/train.py` loads `config.json`, prepares timestamped run folders via `utils.config_manager.make_run_dirs`, and seeds all libraries.
2. The Donut processor/model are fetched from Hugging Face (`model_name`), optional special tokens are registered, and padding/eos ids are aligned.
3. `src/datasets.DonutJsonlDataset` wraps the train/validation JSONL files, resolves each record's image (local file or cached download), applies optional rotation, and tokenises the JSON target string (prompt + serialized fields).
4. `DonutDataCollator` stacks images, pads label sequences with `-100`, and feeds batches to Hugging Face's `Seq2SeqTrainer` configured by `train.*` and `generation.*`.
5. The trainer runs fine-tuning with `predict_with_generate=True`, reporting exact-match and per-field accuracies on the validation set when present.
6. After training, the best model/processor checkpoint is written to `results/run_*/model/`, intermediate checkpoints live under `results/run_*/checkpoints/`, and `run_metadata.json` captures summary metrics.

## Inference

After training, run greedy decoding on a single image:

```bash
python src/inference.py --model_dir results/run_YYYYMMDD_HHMMSS/model --image path/to/image.png [--config config.json]
```

`src/inference.py` mirrors the training prompt (`tokenizer.task_prompt`) and `generation` settings to format predictions. The script prints the raw decoded JSON string.

## Project Layout

```
.
├── config.json                 # Training + data configuration
├── data/                       # Split datasets + cached images
│   ├── raw_data/
│   │   ├── exported_raw_data.xlsx
│   │   └── donut_finetune_dataset.jsonl
│   ├── train_dataset.jsonl
│   ├── validation_dataset.jsonl
│   ├── test_dataset.jsonl
│   ├── data_split_summary.json
│   └── image_cache/
├── data_scripts/
│   ├── transform_data.py       # Excel -> JSONL converter
│   ├── divide_data.py          # Train/val/test splitter
│   └── divide_config.json      # Split ratios/config for the splitter
├── src/
│   ├── datasets.py             # JSONL dataset + data collator for Donut
│   ├── train.py                # Seq2SeqTrainer fine-tuning entry point
│   ├── inference.py            # Single-image inference helper
│   └── __init__.py
├── utils/
│   ├── config_manager.py       # load_config + timestamped run directories
│   └── logging_utils.py        # stdout + file logging setup
└── requirements.txt
```

## Troubleshooting

- **Tokenizer lacks pad token:** The trainer automatically maps the pad token to EOS if missing, but double-check `processor.tokenizer.pad_token_id` before launching long runs.
- **Remote image download errors:** Populate `input.image_path` with local files or ensure your environment can reach the `image_url` endpoints. Cached copies live in `data/image_cache/`.
- **Decoding malformed JSON:** Inspect predictions in `results/run_*/` logs. Prompt engineering (setting `tokenizer.task_prompt`) or adding task-specific special tokens can stabilise decoding.
