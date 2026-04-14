# RoboMME Counting Task Suite LoRA Finetuning

## Dataset

Convert H5 dataset LeRobot format and compute normalization statistics:

```bash
uv run examples/robomme/convert_robomme_to_lerobot.py --h5_data_dir ./robomme_datasets/h5

uv run scripts/compute_norm_stats.py --config-name pi05_robomme_counting_lora
```

## Training

```bash
HF_LEROBOT_HOME=./robomme_datasets/lerobot XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_robomme_counting_lora --exp-name temporal-mem-test --overwrite --batch-size 1
```

## Eval

Start the policy server:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_robomme_counting_lora --policy.dir=checkpoints/pi05_robomme_counting_lora/<exp_name>/<step>
```

Run the RoboMME counting eval script:

```bash
uv run examples/robomme/eval_robomme_counting.py --host 127.0.0.1 --port 8000
```
