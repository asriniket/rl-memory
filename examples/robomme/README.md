# RoboMME Counting Task Suite - LoRA Finetuning

## Dataset

Download H5 files from HuggingFace:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Yinpei/robomme_data_h5', repo_type='dataset', local_dir='./robomme_data_h5')"
```

Convert to LeRobot format and compute normalization statistics:

```bash
uv run examples/robomme/convert_robomme_to_lerobot.py --h5_data_dir ./robomme_data_h5

uv run scripts/compute_norm_stats.py --config-name pi05_robomme_counting_lora
```

## Training

```bash
uv run scripts/train.py pi05_robomme_counting_lora --exp-name my_experiment --overwrite
```

To resume:

```bash
uv run scripts/train.py pi05_robomme_counting_lora --exp-name my_experiment --resume
```

Disable W&B:

```bash
WANDB_MODE=disabled uv run scripts/train.py pi05_robomme_counting_lora --exp-name my_experiment
```
