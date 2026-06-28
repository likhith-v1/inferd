"""inferd — local LLM inference stack (package root).

Import ``inferd.env`` at the top of any GPU entry point before torch or
transformers so CUDA 13 libs are preloaded correctly on WSL2.
"""

from inferd.env import bootstrap, bootstrap_finetune

__all__ = ["bootstrap", "bootstrap_finetune", "__version__"]

__version__ = "0.1.0"
