"""Shared runtime bootstrap: CUDA lib preload and import-order guards.

Import this module before torch, transformers, or bitsandbytes in any entry point:

    import inferd.env  # noqa: F401

Why this exists: PyTorch cu130 bundles CUDA 13 libs inside the venv at
nvidia/cu13/lib/ but does NOT add that path to LD_LIBRARY_PATH. Two of those
libs must be loaded into the process before torch._C.so is dlopened or the
dynamic linker raises ImportError:

  libcusparseLt.so.0  — needed directly by torch._C
  libnvJitLink.so.13  — needed by bitsandbytes CUDA JIT

Loading them with RTLD_GLOBAL makes their symbols visible to any subsequent
dlopen call, exactly as if the directory were on LD_LIBRARY_PATH.
"""

from __future__ import annotations

import ctypes
import logging
import sys
import sysconfig
from pathlib import Path

logger = logging.getLogger(__name__)

# All four libs must be present before torch._C.so is dlopened.
# libnccl and libnvshmem_host live in their own nvidia/* subdirs (not cu13/).
_CUDA_LIBS_TO_PRELOAD = [
    Path("nvidia") / "nccl" / "lib" / "libnccl.so.2",
    Path("nvidia") / "nvshmem" / "lib" / "libnvshmem_host.so.3",
    Path("nvidia") / "cu13" / "lib" / "libcusparseLt.so.0",
    Path("nvidia") / "cu13" / "lib" / "libnvJitLink.so.13",
]
_preloaded = False


def preload_cuda_libs() -> list[str]:
    """
    Pre-load bundled CUDA 13 runtime libs so torch and bitsandbytes can import.

    Returns the list of library paths that were successfully loaded.
    Raises RuntimeError if a library exists on disk but ctypes.CDLL fails.
    """
    global _preloaded
    if _preloaded:
        return []

    site_packages = Path(sysconfig.get_paths()["purelib"])
    loaded: list[str] = []

    for rel in _CUDA_LIBS_TO_PRELOAD:
        lib_path = site_packages / rel
        if not lib_path.is_file():
            logger.debug("CUDA lib not found at %s — skipping", lib_path)
            continue
        try:
            ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
            loaded.append(str(lib_path))
        except OSError as exc:
            raise RuntimeError(
                f"Failed to load CUDA dependency at {lib_path}: {exc}\n"
                "Try: uv sync --reinstall-package nvidia-cusparselt-cu13 "
                "nvidia-nvjitlink-cu13"
            ) from exc

    _preloaded = True
    return loaded


def bootstrap() -> None:
    """Call at inferd entry points before GPU/CUDA library imports."""
    preload_cuda_libs()


def bootstrap_finetune() -> None:
    """Finetune entry points: preload CUDA libs and import unsloth before transformers."""
    bootstrap()
    if "transformers" in sys.modules:
        raise ImportError(
            "transformers was imported before unsloth. "
            "Start finetune scripts with:\n"
            "    from inferd.env import bootstrap_finetune\n"
            "    bootstrap_finetune()\n"
            "before any transformers import so Unsloth patches apply."
        )
    import unsloth  # noqa: F401


bootstrap()
