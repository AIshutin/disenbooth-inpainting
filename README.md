# disenbooth-inpainting

**Note:** the quality of inpainted images is mediocre.

To run telegram bot:
```bash
TOKEN=$(cat token.txt) python3 main.py
```

To train new checkpoints use `train_disenbooth.sh` or `train_disenbooth.py`. Select the best checkpoint according to the validation prompts and edit `config.json`.