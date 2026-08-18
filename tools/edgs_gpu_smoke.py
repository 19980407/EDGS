"""Fast A10/CUDA-extension smoke test for the EDGS container image."""

import json
import os
import platform
import subprocess
import sys

import torch


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(device)
    info = {
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "memory_gib": round(props.total_memory / 1024**3, 2),
        "edgs_commit": os.environ.get("EDGS_COMMIT", "unknown"),
    }
    print(json.dumps(info, ensure_ascii=False, indent=2))
    print(subprocess.check_output(["nvidia-smi", "-L"], text=True).strip())

    assert props.major == 8 and props.minor == 6, (
        f"Expected A10 compute capability 8.6, got {props.major}.{props.minor}"
    )

    from simple_knn._C import distCUDA2

    points = torch.rand((4096, 3), device=device, dtype=torch.float32)
    distances = distCUDA2(points)
    assert distances.shape == (4096,)
    assert torch.isfinite(distances).all()

    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )

    identity = torch.eye(4, device=device, dtype=torch.float32)
    settings = GaussianRasterizationSettings(
        image_height=32,
        image_width=32,
        tanfovx=1.0,
        tanfovy=1.0,
        bg=torch.zeros(3, device=device),
        scale_modifier=1.0,
        viewmatrix=identity,
        projmatrix=identity,
        sh_degree=0,
        campos=torch.zeros(3, device=device),
        prefiltered=False,
        debug=False,
        antialiasing=False,
    )
    visible = GaussianRasterizer(settings).markVisible(points[:64] * 0.1)
    assert visible.shape == (64,)

    import romatch  # noqa: F401
    from source.trainer import EDGSTrainer  # noqa: F401

    torch.cuda.synchronize()
    print(
        "EDGS_GPU_SMOKE_PASS "
        f"knn_mean={distances.mean().item():.8f} visible={int(visible.sum())}/64"
    )


if __name__ == "__main__":
    main()
