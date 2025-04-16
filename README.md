# YouronMath

This repository contains scripts used for the paper "Can A Gamer Train A Mathematical Reasoning Model?" that trains and evaluates a mathematical reasoning model on a single RTX 3080 Ti. 

## Files

- `collect_responses.py`: Collects teacher model responses using DeepSeekMath (or any other model of your choice) for data augmentation.
- `eval_gsm8k.py`: Evaluates models on the GSM8K dataset.
- `eval_mmlu_stem.py`: Evaluates models on MMLU-STEM (math-focused subjects: college_mathematics, high_school_mathematics).
- `gsm8k.py`: Dataset utilities for GSM8K.
- `train.py`: Fine-tunes models on GSM8K.
- `utils.py`: Helper functions for training.

`train.py`,`gsm8k.py`,`utils.py` are forked from [here](https://github.com/Mohammadjafari80/GSM8K-RLVR).

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/shinandrew/YouronMath
   cd YouronMath
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

Fine-tune a model on GSM8K. By defalut, it trains Qwen2.5-Math-1.5B as base model.

```bash
python train.py
```

### Evaluation

- **GSM8K**:

  ```bash
  python eval_gsm8k.py
  ```

- **MMLU-STEM** (requires MMLU data at `data/mmlu/data/`):

  ```bash
  python eval_mmlu_stem.py
  ```

  You can download MMLU data as following:

  ```bash
  wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -P data/mmlu
  tar -xf data/mmlu/data.tar -C data/mmlu
  ```

### Collecting Teacher Responses

In our trial, this actually lowered the performance compared to using GSM8K only. By default, it collects responses from DeepSeekMath-Base-7B.

```bash
python collect_responses.py
```


## Citation

## License

MIT License
