"""nids: network intrusion detection backend (UNSW-NB15 flow features)."""
from .config import Config, FEATURES
from .features import SchemaError
from .inference import Detector
from .pipeline import Bundle, load_bundle, save_bundle, train

__all__ = ["Config", "FEATURES", "SchemaError", "Detector", "Bundle", "load_bundle", "save_bundle", "train"]
__version__ = "0.2.0"
