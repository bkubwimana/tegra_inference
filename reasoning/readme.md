# MMLU-Redux Evaluation with DeepSeek

This repository provides scripts to evaluate a distilled DeepSeek Qwen-14B model on subsets of the MMLU-Redux dataset. It runs inference locally, logs detailed outputs, and computes accuracy and timing metrics.

## Features

- Load any Hugging Face causal LM (default: `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`).
- Evaluate on specific MMLU-Redux subsets (e.g., `electrical_engineering`).
- Extract predicted choice letters (A–D) from various output formats.
- Record per-question logs and aggregate results in CSV.
- Print summary metrics (accuracy, average inference time).

## Setup

1. Clone this repo and navigate to the scripts folder:
   ```bash
   cd /mnt/packages/models/deepseek/distill_qwen14b/mmlu-redux/scripts
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env_example` to `.env` and set `HF_READ_TOKEN`.

## Usage

```bash
python reasoning_custom_template.py \
  --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --subset_name electrical_engineering \
  --num_questions 100 \
  --config reasoning
```

- Outputs and logs are saved under `./outputs/reasoning/`.
- `results_*.csv` contains per-question predictions.
- Log files include prompts, raw outputs, and debug info.

## Extending

- Modify `chat_deepseek.jinja` for custom prompt templates.
- Adjust `extract_predicted_choice` regexes to match new output styles.
- Change stopping token or generation parameters in `predict_local`.
