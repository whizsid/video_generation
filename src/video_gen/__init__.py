"""Reference-guided video generation with Wan2.1-VACE on Apple MPS and CUDA (Colab T4)."""

import os

# Must be set before torch is imported: lets ops without an MPS kernel (e.g. some Conv3D paths) run on CPU.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# The 14B model leaves little VRAM headroom on a T4; without this, fragmentation alone wastes over 1 GB.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

__version__ = "0.1.0"
