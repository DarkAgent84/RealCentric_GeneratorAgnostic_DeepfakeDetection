"""
cnn_backbone.py — CNN Backbone Feature Extractor
==================================================
Implements Component 3 from the Mathematical Formulation paper.

Feature vector: f_CNN = phi_Theta(X) ∈ R^{d_c}

where phi_Theta is a pretrained CNN with the classification head
removed, used as a feature extractor.

Two backbones supported (as specified in the paper):
  ResNet-18      → d_c = 512
  EfficientNet-B0 → d_c = 1280

Both are initialised with ImageNet pretrained weights.
During supervised training the backbone is fine-tuned end-to-end.
During unsupervised envelope fitting the backbone is frozen.

Reference:
  Mathematical Formulation.pdf — Learned CNN Features
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    import torchvision.transforms as T
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────
#  ImageNet normalisation transform (required for pretrained weights)
# ─────────────────────────────────────────────────────────────────────

def _get_transform(image_size: int = 256):
    """
    Standard ImageNet preprocessing transform.
    Matches the normalisation used during pretrained weight training.
    """
    return T.Compose([
        T.ToPILImage(),
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet mean (RGB)
            std =[0.229, 0.224, 0.225],   # ImageNet std  (RGB)
        )
    ])


# ─────────────────────────────────────────────────────────────────────
#  Backbone builder
# ─────────────────────────────────────────────────────────────────────

def _build_backbone(name: str, pretrained: bool,
                    freeze: bool) -> tuple:
    """
    Build CNN backbone with classification head removed.

    Returns:
        model     : nn.Module feature extractor
        output_dim: int dimensionality of output feature vector
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch not installed.\n"
            "On cluster: pip install torch torchvision"
        )

    name = name.lower()

    if name == "resnet18":
        # ResNet-18 — 512-dim output from avgpool layer
        backbone   = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        output_dim = backbone.fc.in_features   # 512
        # Remove classification head
        backbone.fc = nn.Identity()

    elif name == "efficientnet_b0":
        # EfficientNet-B0 — 1280-dim output from avgpool layer
        backbone   = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        output_dim = backbone.classifier[1].in_features   # 1280
        # Remove classification head
        backbone.classifier = nn.Identity()

    else:
        raise ValueError(
            f"Unknown backbone: {name}. "
            "Supported: 'resnet18', 'efficientnet_b0'"
        )

    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False

    return backbone, output_dim


# ─────────────────────────────────────────────────────────────────────
#  Main extractor
# ─────────────────────────────────────────────────────────────────────

class CNNBackboneExtractor:
    """
    CNN feature extractor using pretrained ResNet-18 or EfficientNet-B0.

    Extracts the penultimate-layer representation (before classification
    head) as the feature vector f_CNN.

    Usage (inference / feature extraction):
        extractor = CNNBackboneExtractor(backbone='resnet18', cfg=cfg)
        f = extractor.extract(img)        # shape (512,)
        F = extractor.extract_batch(imgs) # shape (N, 512)

    Usage (training — returns nn.Module for end-to-end training):
        model = extractor.get_model()     # nn.Module
    """

    def __init__(self, backbone: str = "resnet18",
                 cfg: dict = None,
                 pretrained: bool = True,
                 freeze: bool = False,
                 device: str = None):
        """
        Args:
            backbone  : 'resnet18' or 'efficientnet_b0'
            cfg       : config dict (overrides backbone/pretrained/freeze)
            pretrained: use ImageNet pretrained weights
            freeze    : freeze backbone weights (for unsupervised mode)
            device    : 'cuda', 'cpu', or None (auto-detect)
        """
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch not available. Install with:\n"
                "  pip install torch torchvision"
            )

        # Config overrides
        if cfg is not None:
            cnn_cfg   = cfg["features"]["cnn"]
            backbone  = backbone or cfg.get("supervised", {}).get("backbone", "resnet18")
            pretrained = cnn_cfg.get("pretrained", pretrained)
            freeze     = cnn_cfg.get("freeze_backbone", freeze)

        self._backbone_name = backbone.lower()
        self._pretrained    = pretrained
        self._freeze        = freeze

        # Device setup
        if device is None:
            self._device = self._auto_device()
        else:
            self._device = torch.device(device)

        # Build backbone
        self._model, self._output_dim = _build_backbone(
            self._backbone_name, pretrained, freeze
        )
        self._model = self._model.to(self._device)
        self._model.eval()

        # Preprocessing transform
        image_size = cfg["data"]["image_size"] if cfg else 256
        self._transform = _get_transform(image_size)

    # ── Public interface ──────────────────────────────────────────────

    def extract(self, img: np.ndarray) -> np.ndarray:
        """
        Extract CNN feature vector from one image.

        Args:
            img: uint8 numpy array, shape (H, W, C) BGR or RGB

        Returns:
            f_cnn: shape (d_c,)  512 for ResNet-18, 1280 for EfficientNet
        """
        tensor = self._preprocess(img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            features = self._model(tensor)

        return features.squeeze(0).cpu().numpy().astype(np.float32)

    def extract_batch(self, images: list,
                      batch_size: int = 32) -> np.ndarray:
        """
        Extract features from a list of images efficiently.
        Processes in batches to maximise GPU utilisation.

        Args:
            images    : list of uint8 numpy arrays
            batch_size: number of images per GPU batch

        Returns:
            features: shape (N, d_c)
        """
        all_features = []

        for start in range(0, len(images), batch_size):
            batch_imgs = images[start:start + batch_size]
            tensors    = torch.stack([
                self._preprocess(img) for img in batch_imgs
            ]).to(self._device)

            with torch.no_grad():
                feats = self._model(tensors)

            all_features.append(feats.cpu().numpy())

        return np.vstack(all_features).astype(np.float32)

    def get_model(self) -> "nn.Module":
        """
        Return the underlying nn.Module for end-to-end training.
        Used by the supervised MLP training pipeline.
        """
        return self._model

    def set_train_mode(self):
        """Switch to training mode (enables dropout, BN updates)."""
        self._model.train()
        if self._freeze:
            # Keep frozen layers in eval mode even during training
            for module in self._model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.eval()

    def set_eval_mode(self):
        """Switch to inference mode."""
        self._model.eval()

    def unfreeze(self):
        """Unfreeze all backbone parameters for fine-tuning."""
        for param in self._model.parameters():
            param.requires_grad = True
        self._freeze = False

    def get_trainable_params(self) -> list:
        """Return list of trainable parameters (for optimizer)."""
        return [p for p in self._model.parameters()
                if p.requires_grad]

    # ── Preprocessing ─────────────────────────────────────────────────

    def _preprocess(self, img: np.ndarray) -> "torch.Tensor":
        """
        Convert numpy uint8 BGR/RGB image to normalised tensor.
        Handles both BGR (OpenCV) and RGB inputs.
        """
        img = np.asarray(img, dtype=np.uint8)

        # Ensure 3-channel RGB for torchvision transform
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        # OpenCV loads as BGR — convert to RGB for ImageNet normalisation
        # (ImageNet stats were computed on RGB images)
        img_rgb = img[:, :, ::-1].copy()   # BGR → RGB

        return self._transform(img_rgb)

    # ── Device ────────────────────────────────────────────────────────

    @staticmethod
    def _auto_device():
        """Auto-detect best available device."""
        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu    = torch.cuda.get_device_name(0)
            mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  CNN Backbone: GPU — {gpu}  ({mem_gb:.1f} GB)")
        else:
            device = torch.device("cpu")
            print("  CNN Backbone: CPU")
        return device

    # ── Save / load ───────────────────────────────────────────────────

    def save_weights(self, path: str):
        """Save backbone weights to file."""
        import torch
        torch.save(self._model.state_dict(), path)

    def load_weights(self, path: str):
        """Load backbone weights from file."""
        import torch
        state = torch.load(path, map_location=self._device)
        self._model.load_state_dict(state)
        self._model.eval()

    # ── Properties ───────────────────────────────────────────────────

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def backbone_name(self) -> str:
        return self._backbone_name

    @property
    def device(self):
        return self._device

    @property
    def feature_names(self) -> list:
        return [f"cnn_{self._backbone_name}_{i:04d}"
                for i in range(self._output_dim)]

    def describe(self) -> str:
        frozen_str = "frozen" if self._freeze else "trainable"
        return "\n".join([
            f"CNN Backbone Extractor",
            "-" * 40,
            f"  Backbone   : {self._backbone_name}",
            f"  Pretrained : {self._pretrained}  (ImageNet)",
            f"  Weights    : {frozen_str}",
            f"  Output dim : {self._output_dim}",
            f"  Device     : {self._device}",
        ])


# ─────────────────────────────────────────────────────────────────────
#  Factory — create both backbones for comparison
# ─────────────────────────────────────────────────────────────────────

class CNNBackboneFactory:
    """
    Creates and manages both backbone variants for comparison experiments.

    Usage:
        factory = CNNBackboneFactory(cfg)
        resnet_extractor      = factory.get("resnet18")
        efficientnet_extractor = factory.get("efficientnet_b0")
    """

    def __init__(self, cfg: dict = None, freeze: bool = False,
                 device: str = None):
        self._cfg    = cfg
        self._freeze = freeze
        self._device = device
        self._cache  = {}

    def get(self, backbone: str) -> CNNBackboneExtractor:
        """Get or create extractor for given backbone name."""
        name = backbone.lower()
        if name not in self._cache:
            self._cache[name] = CNNBackboneExtractor(
                backbone=name,
                cfg=self._cfg,
                freeze=self._freeze,
                device=self._device
            )
        return self._cache[name]

    def get_output_dims(self) -> dict:
        """Return output dimensions for all supported backbones."""
        if not TORCH_AVAILABLE:
            return {"resnet18": 512, "efficientnet_b0": 1280}
        dims = {}
        for name in ["resnet18", "efficientnet_b0"]:
            dims[name] = self.get(name).output_dim
        return dims


# ─────────────────────────────────────────────────────────────────────
#  Fallback extractor when PyTorch is not available
# ─────────────────────────────────────────────────────────────────────

class CNNBackboneExtractorFallback:
    """
    Fallback when PyTorch is unavailable.
    Returns zero vectors of the correct shape so the pipeline
    can still run with handcrafted features only.
    Logs a clear warning so the user knows CNN features are absent.
    """

    DIMS = {"resnet18": 512, "efficientnet_b0": 1280}

    def __init__(self, backbone: str = "resnet18", **kwargs):
        self._backbone_name = backbone.lower()
        self._output_dim    = self.DIMS.get(self._backbone_name, 512)
        print(
            f"\n  WARNING: PyTorch not available.\n"
            f"  CNN features will be ZEROS ({self._output_dim}-dim).\n"
            f"  Install PyTorch on the cluster:\n"
            f"    pip install torch torchvision\n"
        )

    def extract(self, img: np.ndarray) -> np.ndarray:
        return np.zeros(self._output_dim, dtype=np.float32)

    def extract_batch(self, images: list, batch_size: int = 32) -> np.ndarray:
        return np.zeros((len(images), self._output_dim), dtype=np.float32)

    @property
    def output_dim(self) -> int:
        return self._output_dim

    @property
    def backbone_name(self) -> str:
        return self._backbone_name

    @property
    def feature_names(self) -> list:
        return [f"cnn_{self._backbone_name}_{i:04d}"
                for i in range(self._output_dim)]


def create_cnn_extractor(backbone: str = "resnet18",
                         cfg: dict = None,
                         **kwargs) -> "CNNBackboneExtractor":
    """
    Factory function — returns real extractor if PyTorch available,
    fallback extractor otherwise.
    """
    if TORCH_AVAILABLE:
        return CNNBackboneExtractor(backbone=backbone, cfg=cfg, **kwargs)
    else:
        return CNNBackboneExtractorFallback(backbone=backbone, cfg=cfg)
