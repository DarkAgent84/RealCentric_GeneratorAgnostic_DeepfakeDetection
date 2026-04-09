# ══════════════════════════════════════════════════════════════════════
# fix_pipeline_state.py
# Run ONCE on the cluster:
#   python3 /data/mpstme-naman/deepfake_detection/fix_pipeline_state.py
#
# What it does:
#   The saved pipeline_state.pkl has 974-dim norm stats (old pipeline).
#   The current pipeline code produces 981-dim features.
#   This script refits the normalisation stats on the CURRENT pipeline
#   output using the real images from CelebDF, saves a new
#   pipeline_state.pkl, and re-extracts all feature matrices so
#   Z_train/val/test/ff/sd are all 981-dim and consistent.
#
# Time: ~60-90 minutes (re-extracts 222K images)
# ══════════════════════════════════════════════════════════════════════
import sys, cv2, numpy as np, pickle, time
sys.path.insert(0, '/data/mpstme-naman/deepfake_detection')
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

BASE     = Path('/data/mpstme-naman/deepfake_detection')
PROC     = BASE / 'data' / 'processed'
FEAT_DIR = BASE / 'data' / 'features'
CKPT_DIR = BASE / 'checkpoints'

from config.config_loader import load_config
from src.features.extractor import FeatureFusionPipeline

cfg      = load_config()
pipeline = FeatureFusionPipeline(cfg=cfg, backbone='clip_vit_b32')

print('='*60)
print('  Step 1: Fit envelope on 10,000 real CelebDF images')
print('='*60)
real_paths = sorted((PROC / 'celebdf' / 'real').glob('*.png'))
fit_imgs   = [cv2.imread(str(p)) for p in real_paths[:10000]]
fit_imgs   = [x for x in fit_imgs if x is not None]
t0 = time.time()
pipeline.fit_on_real(fit_imgs)
print(f'  Done in {time.time()-t0:.1f}s  output_dim={pipeline.output_dim}')
assert pipeline.output_dim == 981, f"Expected 981, got {pipeline.output_dim}"

# Save new pipeline state
state_path = CKPT_DIR / 'pipeline_state.pkl'
bak_path   = CKPT_DIR / 'pipeline_state_725.pkl.bak'
if state_path.exists():
    import shutil; shutil.copy(state_path, bak_path)
    print(f'  Backup → {bak_path}')
with open(state_path, 'wb') as f:
    pickle.dump(pipeline.get_state(), f)
print(f'  ✓  New pipeline_state.pkl saved (981-dim)')

def load_folder(folder, label, max_n=None):
    folder = Path(folder)
    if not folder.exists(): return [], []
    paths = sorted(folder.glob('*.png'))
    if max_n: paths = paths[:max_n]
    imgs, lbls = [], []
    for p in tqdm(paths, desc=f'  {folder.parent.name}/{folder.name}', leave=False):
        img = cv2.imread(str(p))
        if img is not None: imgs.append(img); lbls.append(label)
    return imgs, lbls

print('\n' + '='*60)
print('  Step 2: Re-extract CelebDF features (981-dim)')
print('='*60)
cr, clr = load_folder(PROC/'celebdf'/'real', 0)
cf, clf = load_folder(PROC/'celebdf'/'fake', 1)
imgs = cr + cf; lbls = np.array(clr + clf)
t0 = time.time()
Z = pipeline.extract_batch(imgs, normalise=True, cnn_batch_size=64)
print(f'  {time.time()-t0:.0f}s  shape={Z.shape}')
assert Z.shape[1] == 981
np.save(FEAT_DIR/'Z_celebdf.npy', Z); np.save(FEAT_DIR/'y_celebdf.npy', lbls)

# Re-split
Z_tr,Z_tmp,y_tr,y_tmp = train_test_split(Z, lbls, test_size=0.30, random_state=42, stratify=lbls)
Z_val,Z_test,y_val,y_test = train_test_split(Z_tmp,y_tmp,test_size=0.50,random_state=42,stratify=y_tmp)
for name,Zs,ys in [('train',Z_tr,y_tr),('val',Z_val,y_val),('test',Z_test,y_test)]:
    np.save(FEAT_DIR/f'Z_{name}.npy', Zs); np.save(FEAT_DIR/f'y_{name}.npy', ys)
print(f'  Train={len(y_tr):,}  Val={len(y_val):,}  Test={len(y_test):,}')
del cr, cf, imgs

print('\n' + '='*60)
print('  Step 3: Re-extract FaceForensics++ features')
print('='*60)
ffr, ffrl = load_folder(PROC/'ff'/'real', 0)
fff, fffl = load_folder(PROC/'ff'/'fake', 1)
imgs = ffr + fff; lbls = np.array(ffrl + fffl)
t0 = time.time()
Z = pipeline.extract_batch(imgs, normalise=True, cnn_batch_size=64)
print(f'  {time.time()-t0:.0f}s  shape={Z.shape}')
np.save(FEAT_DIR/'Z_ff.npy', Z); np.save(FEAT_DIR/'y_ff.npy', lbls)
del ffr, fff, imgs

print('\n' + '='*60)
print('  Step 4: Re-extract Stable Diffusion features')
print('='*60)
sd, sdl = load_folder(PROC/'sd'/'fake', 1)
t0 = time.time()
Z = pipeline.extract_batch(sd, normalise=True, cnn_batch_size=64)
print(f'  {time.time()-t0:.0f}s  shape={Z.shape}')
np.save(FEAT_DIR/'Z_sd.npy', Z); np.save(FEAT_DIR/'y_sd.npy', np.array(sdl))
del sd

print('\n' + '='*60)
print('  Step 5: Retrain MLP on new 981-dim features (~20 min)')
print('='*60)
from src.models.mlp_classifier import MLPTrainer
Z_train = np.load(FEAT_DIR/'Z_train.npy'); y_train = np.load(FEAT_DIR/'y_train.npy')
Z_val   = np.load(FEAT_DIR/'Z_val.npy');   y_val   = np.load(FEAT_DIR/'y_val.npy')
trainer = MLPTrainer(cfg=cfg, input_dim=981)
best_auc = trainer.train(Z_train, y_train, Z_val, y_val,
                         checkpoint_dir=str(CKPT_DIR), run_name='mlp_supervised')
print(f'  ✓  Best val AUC: {best_auc:.4f}')

print('\n' + '='*60)
print('  Step 6: Retrain DualOneClassEnsemble on new 981-dim features')
print('='*60)
from sklearn.preprocessing import StandardScaler
from src.models.one_class_ensemble import DualOneClassEnsemble
scaler   = StandardScaler()
Z_tr_sc  = scaler.fit_transform(Z_train)
Z_val_sc = scaler.transform(Z_val)
Z_fk_val = Z_val_sc[y_val == 1]
Z_re_val = Z_val_sc[y_val == 0]
Z_re_tr  = Z_tr_sc[y_train == 0]
# Also need FF++ fake validations for the dual ensemble
y_ff_full = np.load(FEAT_DIR/'y_ff.npy')
Z_ff_full = np.load(FEAT_DIR/'Z_ff.npy')
Z_ff_sc = scaler.transform(Z_ff_full)
Z_ff_fk = Z_ff_sc[y_ff_full == 1][:5000] # Subsample for calibration speed

ens = DualOneClassEnsemble(cfg=cfg)
ens.fit_envelope(Z_re_tr)
tau_s, tau_a = ens.calibrate_threshold(Z_re_val, val_fake_smooth=Z_fk_val, val_fake_artifact=Z_ff_fk, fpr_target=0.025)
ens.save(str(CKPT_DIR / 'ensemble.pkl'))
print(f'  ✓  Ensemble saved  tau_smooth={tau_s:.4f}, tau_artifact={tau_a:.4f}')

print('\n  ✅  ALL DONE — everything is now 981-dim and consistent.')
print('     Open NB06 and run all cells.')
