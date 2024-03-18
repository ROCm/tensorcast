"""TensorCast: Conversion and compression of arbitrary datatypes."""
# tcast/extension.py: loads torch extension

from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import torch
from torch.utils.cpp_extension import load

from .utils import is_float8_available


class Extension:
    """Wrapper for PyTorch extension."""

    def __init__(
        self,
        extname: str = None,
        srcpath: Path = None,
        exec_only: bool = False,
        cpu_only: bool = False,
        verbose: bool = True,
    ):
        if extname is None:
            extname = "tcast_extension"
        cpu_only = cpu_only or not torch.cuda.is_available()
        if isinstance(srcpath, str):
            srcpath = Path(str)
        elif not isinstance(srcpath, Path):
            srcpath = Path(__file__).parent.with_name("csrc")
        if not srcpath.is_dir():
            raise RuntimeError(f"Extension: cannot find source path {str(srcpath)}")
        srcfiles = self.get_source_files(srcpath, cpu_only)
        if not srcfiles:
            raise RuntimeError(f"Extension: no source files (.cpp/.cxx/.c/.cu) found in {str(srcpath)}")
        cpu_flags = self.get_cpu_flags()
        if is_float8_available():
            cpu_flags.append("-DFLOAT8_AVAILABLE_CPU")
        is_rocm = (
            hasattr(torch.version, "hip") and torch.version.hip is not None and torch.utils.cpp_extension.ROCM_HOME is not None
        )
        gpu_flags = self.get_gpu_flags(is_rocm, verbose) if not cpu_only else []
        extension = self.load_extension(extname, srcfiles, cpu_flags, gpu_flags, exec_only, cpu_only, verbose)
        if isinstance(extension, ModuleType):
            self.extension, self.exec_path = extension, None
            print(f"Extension: loaded module {extension.__name__}")
        elif isinstance(extension, Path):
            self.extension, self.exec_path = None, extension
            print(f"Extension: lpath to executable is {str(extension)}")
        else:
            raise RuntimeError(f"Extension: failed to load, recieved {str(extension)}")

    def get_source_files(self, srcpath: Path, cpu_only: bool) -> list[Path]:
        """Get the source files.  If cpu_only, skip .cu files."""
        srcfiles = list(srcpath.glob("*.cpp")) + list(srcpath.glob("*.cxx")) + list(srcpath.glob("*.c"))
        if not cpu_only:
            srcfiles += list(srcpath.glob("*.cu"))
        return srcfiles

    def get_gpu_flags(self, is_rocm: bool, verbose: bool = False) -> list[str]:
        """Get any GPU flags we might need."""
        if not is_rocm:
            flags = ["-O4", "--gpu-architecture=native"]
            if verbose:
                flags += ["--ptxas-options=-v", "-v"]
        else:
            flags = ["-O3"]
        return flags

    def get_cpu_flags(self) -> list[str]:
        """See what AVX is available, if any."""
        flags, result = ["-O3", "-march=native"], None
        try:
            from cpuinfo import cpu_info

            info = cpu_info()
            if info["vendor_id_raw"] == "AuthenticAMD":
                result = info["flags"]
        except ImportError:
            from subprocess import check_output

            result = check_output("lscpu", shell=True).decode("utf-8").strip().lower().split()
        finally:
            if result:
                flags += [f"-m{i}" for i in result if i.startswith("avx")]
        return flags

    def load_extension(
        self,
        name: str,
        srcfiles: list[Path],
        cflags: list[str],
        gflags: list[str],
        exec_only: bool,
        cpu_only: bool,
        verbose: bool,
    ) -> ModuleType | Path:
        return load(
            name=name,
            sources=srcfiles,
            extra_cflags=cflags,
            extra_cuda_cflags=gflags,
            verbose=verbose,
            is_standalone=exec_only,
            with_cuda=not cpu_only,
            is_python_module=not exec_only,
        )

    def list_operations(self) -> list[str]:
        """Print the operations that are supported."""
        ops = []
        for k, v in self.__dict__.items():
            if isinstance(v, Callable) and (k.endswith("_cpu") or k.endswith("_gpu")):
                print(k)
                ops.append(k)
        return ops

    def has_operation(self, name: str, platform: str) -> bool:
        """If the extension exists and has the op for gpu/cpu are requested, do it."""
        if not self.extension:
            return False
        assert platform in ("cpu", "gpu")
        return hasattr(self, f"{name}_{platform}")

    def exec_operation(self, tensor: torch.Tensor, name: str, platform: str, **kwargs) -> torch.Tensor:
        """Run an operation that has (at least) a tensor."""
        assert self.has_operation(name, platform)
        tplatform = "cpu" if tensor.is_cpu else "gpu"
        if tplatform != platform:
            raise NotImplementedError(f"Extension: tensor is on {tplatform}, but op '{name}' is for {platform}")
        return getattr(self, f"{name}_{platform}")(tensor, **kwargs)
