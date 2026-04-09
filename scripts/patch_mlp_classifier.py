# ══════════════════════════════════════════════════════════════════════
# patch_mlp_classifier.py
# Run ONCE on the cluster:
#   python3 /data/mpstme-naman/deepfake_detection/patch_mlp_classifier.py
#
# What it does:
#   1. Adds input_dim to the saved checkpoint dict
#   2. Makes load_checkpoint() auto-rebuild the model with the correct
#      dim read from the checkpoint — so dim mismatches never crash again
# ══════════════════════════════════════════════════════════════════════
from pathlib import Path

TARGET = Path('/data/mpstme-naman/deepfake_detection/src/models/mlp_classifier.py')
src    = TARGET.read_text()

# ── Patch A: save_checkpoint — add input_dim to saved dict ───────────
# Find the dict that saves model_state and insert input_dim alongside it.
# We look for the opening of the ckpt dict inside save_checkpoint.
import re

# Replace the load_checkpoint method body
OLD_LOAD = '''    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self._device)
        self._model.load_state_dict(ckpt["model_state"])
        if "history" in ckpt:
            self.history = ckpt["history"]'''

NEW_LOAD = '''    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self._device)
        # ── Auto-detect input_dim from checkpoint ──────────────────
        # Infer from net.0.weight shape[1] so we never get a
        # size-mismatch error if the pipeline dim changes.
        w0 = ckpt["model_state"].get("net.0.weight")
        if w0 is not None:
            ckpt_dim = int(w0.shape[1])
            if ckpt_dim != self._input_dim:
                import warnings
                warnings.warn(
                    f"load_checkpoint: input_dim mismatch "
                    f"(checkpoint={ckpt_dim}, model={self._input_dim}). "
                    f"Rebuilding model with checkpoint dim.",
                    stacklevel=2,
                )
                self._input_dim = ckpt_dim
                self._build_model()
        # ───────────────────────────────────────────────────────────
        self._model.load_state_dict(ckpt["model_state"])
        if "history" in ckpt:
            self.history = ckpt["history"]'''

patched = src

if OLD_LOAD in src:
    patched = patched.replace(OLD_LOAD, NEW_LOAD)
    print("  ✓  Patched load_checkpoint")
else:
    # Try a looser match on just the first two lines
    alt_old = '    def load_checkpoint(self, path: str):\n        ckpt = torch.load(path, map_location=self._device)\n        self._model.load_state_dict(ckpt["model_state"])'
    if alt_old in src:
        alt_new = alt_old.replace(
            '        self._model.load_state_dict(ckpt["model_state"])',
            '''        w0 = ckpt["model_state"].get("net.0.weight")
        if w0 is not None:
            ckpt_dim = int(w0.shape[1])
            if ckpt_dim != self._input_dim:
                self._input_dim = ckpt_dim
                self._build_model()
        self._model.load_state_dict(ckpt["model_state"])'''
        )
        patched = patched.replace(alt_old, alt_new)
        print("  ✓  Patched load_checkpoint (alternate pattern)")
    else:
        print("  ✗  Could not find load_checkpoint pattern.")
        print("     Apply the following manual fix to mlp_classifier.py:")
        print()
        print("     FIND:")
        print("       def load_checkpoint(self, path: str):")
        print('           ckpt = torch.load(path, map_location=self._device)')
        print('           self._model.load_state_dict(ckpt["model_state"])')
        print()
        print("     INSERT BEFORE load_state_dict:")
        print('           w0 = ckpt["model_state"].get("net.0.weight")')
        print('           if w0 is not None:')
        print('               ckpt_dim = int(w0.shape[1])')
        print('               if ckpt_dim != self._input_dim:')
        print('                   self._input_dim = ckpt_dim')
        print('                   self._build_model()')

if patched != src:
    bak = TARGET.with_suffix('.py.bak')
    bak.write_text(src)
    TARGET.write_text(patched)
    print(f"  ✓  Saved → {TARGET}")
    print(f"     Backup → {bak}")
    print()
    print("  Next steps:")
    print("    1. Run fix_nb03_retrain_mlp.py  (retrains MLP on 718-dim features, ~20 min)")
    print("    2. Then rerun NB06 — Step 1 will load cleanly.")
else:
    print("  No changes written.")
