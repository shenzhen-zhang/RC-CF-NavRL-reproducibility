"""Launch Isaac Sim with an optional headless experience override."""

from __future__ import annotations

import os
import sys
import traceback
from multiprocessing.reduction import ForkingPickler
from pathlib import Path
from typing import Iterable

import functorch
import torch
import torch.multiprocessing.reductions as torch_mp_reductions


def _validate_torch_family() -> None:
    torch_root = Path(torch.__file__).resolve().parent.parent
    functorch_root = Path(functorch.__file__).resolve().parent.parent
    if torch_root != functorch_root:
        raise RuntimeError(
            "torch and functorch must come from the same installation: "
            f"torch={torch_root}, functorch={functorch_root}"
        )


# Isaac's omni.pip.torch extension prepends its bundled Torch path during Kit
# startup. Import both packages first so a CUDA-compatible override stays atomic.
_validate_torch_family()

torch_override_dir = os.environ.get("NAVRL_TORCH_OVERRIDE_DIR")
if torch_override_dir:
    # TensorDict 0.5 retains NavRL's functional-module API, while Torch 2.7
    # moved this standard multiprocessing export out of its compatibility shim.
    if not hasattr(torch_mp_reductions, "ForkingPickler"):
        torch_mp_reductions.ForkingPickler = ForkingPickler

    import tensordict
    import torchrl

    expected_root = Path(torch_override_dir).resolve()
    for package in (torch, functorch, tensordict, torchrl):
        package_path = Path(package.__file__).resolve()
        if not package_path.is_relative_to(expected_root):
            raise RuntimeError(
                f"{package.__name__} was not loaded from the Torch override: "
                f"{package_path}"
            )

from omni.isaac.kit import SimulationApp


def validate_navrl_dynamic_obstacles(num_obstacles: int) -> None:
    """Validate counts supported by the exact-count compatibility patch."""
    if num_obstacles < 0 or (num_obstacles != 0 and num_obstacles < 8):
        raise ValueError(
            "env_dyn.num_obstacles must be 0 or at least 8 so every official "
            "NavRL obstacle category remains represented"
        )


def create_simulation_app(
    *,
    headless: bool,
    anti_aliasing: int = 1,
    required_extensions: Iterable[str] = (
        "omni.isaac.debug_draw",
        "omni.replicator.core",
    ),
) -> SimulationApp:
    config = {
        "headless": bool(headless),
        "anti_aliasing": int(anti_aliasing),
        # Keep Python in control of shutdown so exceptions and smoke-test
        # failures are not hidden by Kit's immediate process termination.
        "fast_shutdown": False,
    }
    experience = os.environ.get("NAVRL_ISAAC_EXPERIENCE") if headless else None
    if experience:
        experience_path = Path(experience)
        if not experience_path.is_file():
            raise FileNotFoundError(
                f"NAVRL_ISAAC_EXPERIENCE does not exist: {experience_path}"
            )
        simulation_app = SimulationApp(config, experience=str(experience_path))
    else:
        simulation_app = SimulationApp(config)

    try:
        from omni.isaac.core.utils.extensions import enable_extension

        asset_root_url = os.environ.get("NAVRL_ASSET_ROOT_URL")
        if asset_root_url:
            simulation_app.set_setting(
                "/persistent/isaac/asset_root/default", asset_root_url
            )
            simulation_app.set_setting(
                "/persistent/isaac/asset_root/nvidia", asset_root_url
            )

        for extension_name in required_extensions:
            if not enable_extension(extension_name):
                raise RuntimeError(
                    f"Isaac extension could not be enabled: {extension_name}"
                )
        simulation_app.update()
        return simulation_app
    except BaseException:
        simulation_app.close()
        raise


def close_simulation_app(simulation_app: SimulationApp, *, success: bool) -> None:
    """Exit after flushing diagnostics because Kit 2023.1 crashes on close."""
    if success:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(1)
