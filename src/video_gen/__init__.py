"""Reference-guided video generation with Wan2.1-VACE-1.3B on Apple MPS."""

import os

# Must be set before torch is imported: lets ops without an MPS kernel (e.g. some Conv3D paths) run on CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.1.0"
