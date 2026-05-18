"""JIT-compiled native swapper extension.

The C++ swapper sources live alongside this module at ``./native/``.
We compile them on first import using
:func:`torch.utils.cpp_extension.load`; subsequent imports reuse the
cached binary under ``~/.cache/torch_extensions/``.

Build dependencies (must be discoverable to the C++ compiler):

* libzmq runtime + ``zmq.h``
* cppzmq header (``zmq.hpp``)
* msgpack-cxx header (``msgpack.hpp``)

On Debian / Ubuntu::

    sudo apt install libzmq3-dev libmsgpack-cxx-dev
    # cppzmq is a single header; fetch a pinned release if it is not
    # already on the include path:
    sudo curl -fsSL -o /usr/local/include/zmq.hpp \\
        https://raw.githubusercontent.com/zeromq/cppzmq/v4.10.0/zmq.hpp

Pybind11 and the CUDA runtime are pulled in through PyTorch.
"""
from __future__ import annotations

from pathlib import Path

from torch.utils.cpp_extension import load

from vllm.logger import init_logger

logger = init_logger(__name__)

_NATIVE_DIR = Path(__file__).resolve().parent / "native"

_module = None


def _load() -> object:
    global _module
    if _module is not None:
        return _module

    sources = [str(_NATIVE_DIR / "swapper.cpp"),
               str(_NATIVE_DIR / "binding.cpp")]
    logger.info("Compiling swapper_mt extension (first run only) ...")
    _module = load(
        name="swapper_mt",
        sources=sources,
        extra_include_paths=[str(_NATIVE_DIR)],
        extra_cflags=["-O3", "-std=c++20", "-DMSGPACK_NO_BOOST"],
        extra_ldflags=["-lzmq"],
        verbose=False,
    )
    return _module


def __getattr__(name: str):
    return getattr(_load(), name)
