"""Small real EDGS reconstruction smoke test using the bundled fruit video."""

import os
import shutil
import time
from pathlib import Path

import hydra
import torch
from hydra import compose, initialize

from romatch import roma_indoor
from source.trainer import EDGSTrainer
from source.utils_aux import set_seed
from source.utils_preprocess import (
    preprocess_frames,
    read_video_frames,
    run_colmap_on_scene,
    save_frames_to_scene_dir,
    select_optimal_frames,
)


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is not available"
    root = Path(os.environ.get("EDGS_SMOKE_ROOT", "/tmp/edgs-e2e"))
    scene = root / "scene"
    output = root / "output"
    shutil.rmtree(root, ignore_errors=True)
    scene.mkdir(parents=True)

    started = time.perf_counter()
    frames = read_video_frames(
        "assets/examples/video_fruits.mp4", k=3, max_size=512
    )
    scores = preprocess_frames(frames)
    selected = select_optimal_frames(scores, k=min(12, len(frames)))
    save_frames_to_scene_dir([frames[i] for i in selected], str(scene))
    run_colmap_on_scene(str(scene))
    colmap_seconds = time.perf_counter() - started

    roma = roma_indoor(device="cuda:0", coarse_res=560, upsample_res=560)
    roma.upsample_preds = False
    roma.symmetric = False

    with initialize(config_path="../configs", version_base="1.2"):
        cfg = compose(config_name="train")
    cfg.wandb.mode = "disabled"
    cfg.gs.dataset.source_path = str(scene)
    cfg.gs.dataset.model_path = str(output)
    cfg.gs.dataset.images = "images"
    cfg.gs.dataset.data_device = "cuda"
    cfg.gs.opt.batch_size = 1
    cfg.gs.opt.save_iterations = []
    cfg.train.gs_epochs = 1
    cfg.train.no_densify = True
    cfg.train.reduce_opacity = False
    cfg.init_wC.use = True
    cfg.init_wC.matches_per_ref = 256
    cfg.init_wC.num_refs = min(4, len(selected))
    cfg.init_wC.nns_per_ref = 1
    cfg.init_wC.add_SfM_init = False
    set_seed(cfg.seed)

    generator = hydra.utils.instantiate(cfg.gs, do_train_test_split=False)
    trainer = EDGSTrainer(
        GS=generator,
        training_config=cfg.gs.opt,
        device=cfg.device,
        log_wandb=False,
    )
    trainer.saving_iterations = []
    trainer.evaluate_iterations = []

    edgs_started = time.perf_counter()
    trainer.timer.start()
    trainer.init_with_corr(cfg.init_wC, roma_model=roma)
    init_seconds = time.perf_counter() - edgs_started
    trainer.train(cfg.train)
    torch.cuda.synchronize()
    total_seconds = time.perf_counter() - started

    assert trainer.gaussians.get_xyz.shape[0] > 0
    print(
        "EDGS_E2E_PASS "
        f"frames={len(selected)} gaussians={trainer.gaussians.get_xyz.shape[0]} "
        f"colmap_s={colmap_seconds:.2f} init_s={init_seconds:.2f} total_s={total_seconds:.2f}"
    )


if __name__ == "__main__":
    main()
