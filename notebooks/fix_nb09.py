import json
import os

path = r'c:\Users\mpstme.student\Documents\NamanShah\deepfake_detection\notebooks\09_train_multi_dataset.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if 'source' in cell and isinstance(cell['source'], str):
        if "epochs = range(1, len(history['train_loss']) + 1)" in cell['source']:
            cell['source'] = cell['source'].replace(
                "epochs = range(1, len(history['train_loss']) + 1)",
                "history = mlp_multi.history\nepochs = range(1, len(history['train_loss']) + 1)"
            )
        if "ckpt_exists = os.path.exists(ckpt_path)" in cell['source']:
            cell['source'] = cell['source'].replace(
                "ckpt_exists = os.path.exists(ckpt_path)",
                "ckpt_path = str(CKPT_DIR / 'mlp_multi_best.pt')\nckpt_exists = os.path.exists(ckpt_path)"
            )

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook fixed.")
