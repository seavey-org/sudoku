"""Model registry for lazy loading and caching models."""
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

from ..utils.logging_config import get_logger

logger = get_logger(__name__)

# Singleton registry instance
_registry: Optional['ModelRegistry'] = None


def get_model_registry() -> 'ModelRegistry':
    """Get or create the global model registry."""
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


class ModelRegistry:
    """Centralized registry for lazy-loading ML models."""

    def __init__(self, models_dir: Optional[Path] = None):
        """Initialize the model registry.

        Args:
            models_dir: Path to models directory. Defaults to ./models
        """
        if models_dir is None:
            models_dir = Path(__file__).parent.parent / 'models'
        self.models_dir = models_dir

        # Cached model instances
        self._easyocr_reader = None
        self._digit_cnn = None
        self._boundary_classifier = None
        self._boundary_scaler = None
        self._cage_sum_cnn = None
        self._cage_sum_device = None
        self._cage_sum_label_mapping = None

    def get_easyocr_reader(self):
        """Get or create EasyOCR reader."""
        if self._easyocr_reader is None:
            try:
                import easyocr
                logger.info("Loading EasyOCR model...")
                self._easyocr_reader = easyocr.Reader(['en'], gpu=True)
                logger.info("EasyOCR model loaded.")
            except Exception as e:
                logger.error(f"Failed to load EasyOCR: {e}")
                return None
        return self._easyocr_reader

    def get_digit_cnn(self):
        """Get or create digit CNN classifier.

        Returns:
            CNNDigitClassifier instance or None
        """
        if self._digit_cnn is None:
            try:
                from digit_classifier import CNNDigitClassifier, get_model_path
                model_path = get_model_path()
                if os.path.exists(model_path):
                    logger.info("Loading CNN digit classifier...")
                    self._digit_cnn = CNNDigitClassifier(model_path)
                    logger.info("CNN classifier loaded.")
                else:
                    logger.warning(f"CNN model not found at {model_path}")
                    return None
            except Exception as e:
                logger.error(f"Failed to load CNN classifier: {e}")
                return None
        return self._digit_cnn

    def get_boundary_classifier(self) -> Tuple[Any, Any]:
        """Get or create boundary classifier.

        Returns:
            Tuple of (classifier, scaler) or (None, None)
        """
        if self._boundary_classifier is None:
            try:
                import joblib
                classifier_path = self.models_dir / 'boundary_classifier_rf.pkl'
                scaler_path = self.models_dir / 'boundary_scaler.pkl'

                if classifier_path.exists() and scaler_path.exists():
                    logger.info("Loading ML boundary classifier...")
                    self._boundary_classifier = joblib.load(classifier_path)
                    self._boundary_scaler = joblib.load(scaler_path)
                    logger.info("ML boundary classifier loaded.")
                else:
                    logger.warning(f"ML boundary model not found at {self.models_dir}")
                    return None, None
            except Exception as e:
                logger.error(f"Failed to load boundary classifier: {e}")
                return None, None
        return self._boundary_classifier, self._boundary_scaler

    def get_cage_sum_cnn(self) -> Tuple[Any, Any, Optional[Dict[int, int]]]:
        """Get or create cage sum CNN classifier.

        Returns:
            Tuple of (model, device, label_mapping) or (None, None, None)
        """
        if self._cage_sum_cnn is None:
            try:
                import torch
                from .architectures import CageSumCNN, ImprovedCageSumCNN

                model_path = self.models_dir / 'cage_sum_cnn.pth'

                if model_path.exists():
                    logger.info("Loading cage sum CNN classifier...")
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

                    if 'idx_to_label' in checkpoint:
                        self._cage_sum_label_mapping = checkpoint['idx_to_label']
                        num_classes = checkpoint.get('num_classes', len(self._cage_sum_label_mapping))
                    else:
                        self._cage_sum_label_mapping = {i: i + 1 for i in range(45)}
                        num_classes = 45

                    model_type = checkpoint.get('model_type', 'CageSumCNN')
                    if model_type == 'ImprovedCageSumCNN':
                        model = ImprovedCageSumCNN(num_classes=num_classes)
                        logger.info("Using ImprovedCageSumCNN (ResNet-style)")
                    else:
                        model = CageSumCNN(num_classes=num_classes)
                        logger.info("Using CageSumCNN (original)")

                    model.load_state_dict(checkpoint['model_state_dict'])
                    model.to(device)
                    model.eval()

                    self._cage_sum_cnn = model
                    self._cage_sum_device = device
                    logger.info(f"Cage sum CNN loaded ({len(self._cage_sum_label_mapping)} classes)")
                else:
                    logger.warning(f"Cage sum CNN not found at {model_path}")
                    return None, None, None
            except Exception as e:
                logger.error(f"Failed to load cage sum CNN: {e}")
                import traceback
                traceback.print_exc()
                return None, None, None

        return self._cage_sum_cnn, self._cage_sum_device, self._cage_sum_label_mapping

    def clear_cache(self):
        """Clear all cached models to free memory."""
        self._easyocr_reader = None
        self._digit_cnn = None
        self._boundary_classifier = None
        self._boundary_scaler = None
        self._cage_sum_cnn = None
        self._cage_sum_device = None
        self._cage_sum_label_mapping = None
        logger.info("Model cache cleared")
