# ══════════════════════════════════════════════════════════════════════
# fix_nb03_retrain_mlp.py  (updated)
# 
# Situation: Z_train.npy is 974-dim AND mlp_supervised_best.pt was
# trained on 974-dim — they are consistent. No retraining needed.
#
# This script just VERIFIES everything is consistent and runs a quick
# test evaluation to confirm NB06 will work.
# ══════════════════════════════════════════════════════════════════════
import sys, numpy as np, torch
sys.path.insert(0, '/data/mpstme-naman/deepfake_detection')
from pathlib import Path

BASE     = Path('/data/mpstme-naman/deepfake_detection')
FEAT_DIR = BASE / 'data' / 'features'
CKPT_DIR = BASE / 'checkpoints'

# ── 1. Check feature dims ─────────────────────────────────────────────
Z_train = np.load(FEAT_DIR / 'Z_train.npy')
Z_val   = np.load(FEAT_DIR / 'Z_val.npy')
Z_test  = np.load(FEAT_DIR / 'Z_test.npy')
y_train = np.load(FEAT_DIR / 'y_train.npy')
y_val   = np.load(FEAT_DIR / 'y_val.npy')
y_test  = np.load(FEAT_DIR / 'y_test.npy')

disk_dim = Z_train.shape[1]
print(f'  Feature dim on disk  : {disk_dim}')

# ── 2. Check checkpoint dim ───────────────────────────────────────────
ckpt     = torch.load(str(CKPT_DIR / 'mlp_supervised_best.pt'), map_location='cpu')
ckpt_dim = int(ckpt['model_state']['net.0.weight'].shape[1])
print(f'  Checkpoint input_dim : {ckpt_dim}')

if disk_dim == ckpt_dim:
    print(f'\n  ✓  Features and checkpoint are CONSISTENT ({disk_dim}-dim).')
    print(f'     No retraining needed. The patch already handles loading.')
else:
    print(f'\n  ✗  Still mismatched — please report this.')
    sys.exit(1)

# ── 3. Load MLP with patched loader and evaluate ──────────────────────
from config.config_loader import load_config
from src.models.mlp_classifier import MLPTrainer

cfg     = load_config()
trainer = MLPTrainer(cfg=cfg, input_dim=disk_dim)
trainer.load_checkpoint(str(CKPT_DIR / 'mlp_supervised_best.pt'))
print(f'\n  ✓  MLP loaded successfully (input_dim={disk_dim})')

# ── 4. CelebDF test set evaluation ────────────────────────────────────
m = trainer.evaluate(Z_test, y_test)
print('\n' + '='*50)
print('  CelebDF Test Set')
print('='*50)
for k in ['accuracy', 'auc', 'precision', 'recall', 'f1']:
    print(f'  {k:<12}: {m[k]:.4f}')
print(f'\n  TP={m["tp"]}  FP={m["fp"]}  FN={m["fn"]}  TN={m["tn"]}')

# ── 5. Cross-dataset ──────────────────────────────────────────────────
print('\nCross-dataset test:')
print('='*55)
for name, zf, yf in [('FaceForensics++', 'Z_ff.npy', 'y_ff.npy'),
                      ('Stable Diffusion', 'Z_sd.npy', 'y_sd.npy')]:
    Z = np.load(FEAT_DIR / zf)
    y = np.load(FEAT_DIR / yf)
    if len(np.unique(y)) < 2:
        prob = trainer.predict_proba(Z)
        print(f'  {name:<20} mean_score={prob.mean():.4f}  (fake-only)')
    else:
        mx = trainer.evaluate(Z, y)
        print(f'  {name:<20} ACC={mx["accuracy"]:.2f}%  AUC={mx["auc"]:.4f}  F1={mx["f1"]:.4f}')

print('\n  ✅  All good. Now open NB06 and run all cells — it will work.')
