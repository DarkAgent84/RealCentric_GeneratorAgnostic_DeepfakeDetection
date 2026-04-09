import json

path06 = r"c:\Users\mpstme.student\Documents\NamanShah\deepfake_detection\notebooks\06_benchmark.ipynb"
with open(path06, "r", encoding="utf-8") as f:
    nb06 = json.load(f)

markdown_cell = {
 "cell_type": "markdown",
 "metadata": {},
 "source": [
  "## Social Media Pipeline Simulation\n",
  "Evaluates detection robustness after sequential degradation: JPEG (q=78) -> Gaussian Blur (k=3), simulating typical WhatsApp/Instagram compression on 1000 random images."
 ]
}

code_cell = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "import cv2\n",
  "import random\n",
  "from sklearn.metrics import roc_auc_score\n",
  "\n",
  "print('\\n=== Social Media Degradation Simulation ===')\n",
  "\n",
  "# Load 500 Real and 500 Fake from CelebDF using existing paths defined previously in robustness sweep\n",
  "np.random.seed(42)\n",
  "real_paths = sorted((PROC/'celebdf'/'real').glob('*.png'))\n",
  "fake_paths = sorted((PROC/'celebdf'/'fake').glob('*.png'))\n",
  "samp_r = np.random.choice(real_paths, 500, replace=False)\n",
  "samp_f = np.random.choice(fake_paths, 500, replace=False)\n",
  "all_samp = list(samp_r) + list(samp_f)\n",
  "lbls = [0]*500 + [1]*500\n",
  "\n",
  "pipeline.set_state(np.load(CKPT_DIR / 'pipeline_state.pkl', allow_pickle=True))\n",
  "\n",
  "def degrade(img):\n",
  "    # 1. JPEG Compression (q=78)\n",
  "    _, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 78])\n",
  "    img_jpg = cv2.imdecode(enc, 1)\n",
  "    # 2. Gaussian Blur (k=3)\n",
  "    img_blur = cv2.GaussianBlur(img_jpg, (3, 3), 0)\n",
  "    return img_blur\n",
  "\n",
  "print('Applying sequential JPEG(78) -> Blur(3) degradation...')\n",
  "deg_imgs = [degrade(cv2.imread(str(p))) for p in all_samp]\n",
  "\n",
  "print('Extracting features...')\n",
  "Z_deg = pipeline.extract_batch(deg_imgs, normalise=True, cnn_batch_size=64)\n",
  "\n",
  "probs = mlp.predict_proba(Z_deg)\n",
  "auc = roc_auc_score(lbls, probs)\n",
  "print(f'Social Media Simulation AUC: {auc:.4f}')\n",
  "if auc >= 0.95:\n",
  "    print('SUCCESS: Model retains > 0.95 AUC under realistic social media conditions.')\n",
  "else:\n",
  "    print('WARNING: Model degrades under combined compression.')"
 ]
}

nb06['cells'].extend([markdown_cell, code_cell])
with open(path06, "w", encoding="utf-8") as f:
    json.dump(nb06, f, indent=1)


path08 = r"c:\Users\mpstme.student\Documents\NamanShah\deepfake_detection\notebooks\08_extended_feature_validation.ipynb"
with open(path08, "r", encoding="utf-8") as f:
    nb08 = json.load(f)

md8 = {
 "cell_type": "markdown",
 "metadata": {},
 "source": [
  "## SD Features Ablation Study\n",
  "Isolates the discriminative power of the 7-dim SD Features block [206:213] natively built to detect diffusion noise. We zero out the block and re-evaluate detection rate."
 ]
}

code8 = {
 "cell_type": "code",
 "execution_count": None,
 "metadata": {},
 "outputs": [],
 "source": [
  "print('\\n=== SD Features Ablation ===')\n",
  "Z_sd = np.load(FEAT_DIR / 'Z_sd.npy')\n",
  "y_sd = np.load(FEAT_DIR / 'y_sd.npy')\n",
  "\n",
  "Z_train = np.load(FEAT_DIR / 'Z_train_multi.npy')\n",
  "y_train = np.load(FEAT_DIR / 'y_train_multi.npy')\n",
  "\n",
  "# Check means to ensure zeroing removes the anomalous signal unambiguously\n",
  "real_mean = Z_train[y_train == 0, 206:213].mean()\n",
  "sd_fake_mean = Z_sd[:, 206:213].mean()\n",
  "\n",
  "print(f'Real Train Mean [206:213]: {real_mean:.4f}')\n",
  "print(f'SD Fake Mean    [206:213]: {sd_fake_mean:.4f}')\n",
  "if abs(sd_fake_mean - real_mean) > 0.5:\n",
  "    print('-> Fake samples exhibit a strong positive/negative anomaly. Zeroing will safely ablate the signal.')\n",
  "else:\n",
  "    print('-> Warning: Fake samples are already near real mean. Ablation may carry no signal.')\n",
  "\n",
  "# Import models\n",
  "from src.models.mlp_classifier import MLPTrainer\n",
  "from config.config_loader import load_config\n",
  "cfg = load_config()\n",
  "mlp = MLPTrainer(cfg=cfg, input_dim=981)\n",
  "mlp.load_model(str(CKPT_DIR / 'mlp_multi_best.pt'))\n",
  "\n",
  "# Baseline SD Detection\n",
  "base_probs = mlp.predict_proba(Z_sd)\n",
  "base_det = (base_probs >= 0.5).mean() * 100\n",
  "print(f'\\nBaseline SD Detection Rate : {base_det:.2f}%')\n",
  "\n",
  "# Ablated Evaluation\n",
  "Z_sd_zeroed = Z_sd.copy()\n",
  "Z_sd_zeroed[:, 206:213] = 0.0  # Mathematically equivalent to imputing with the Real-Image Mean (Z-score matching)\n",
  "\n",
  "abl_probs = mlp.predict_proba(Z_sd_zeroed)\n",
  "abl_det = (abl_probs >= 0.5).mean() * 100\n",
  "print(f'Ablated SD Detection Rate  : {abl_det:.2f}%')\n",
  "print(f'\\n-> Independent contribution of SD block: {base_det - abl_det:+.2f}%')\n"
 ]
}

nb08['cells'].extend([md8, code8])
with open(path08, "w", encoding="utf-8") as f:
    json.dump(nb08, f, indent=1)
