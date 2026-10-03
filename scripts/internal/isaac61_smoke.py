"""Run an asset-backed task reset and validate native camera frames without a policy server."""

import argparse
import faulthandler
import json
from pathlib import Path

from isaaclab.app import AppLauncher

faulthandler.dump_traceback_later(90, repeat=True)
parser = argparse.ArgumentParser()
parser.add_argument("--task", default="push_T")
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--without-planner", action="store_true")
parser.add_argument("--resets", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.require_kit = True
args.enable_cameras = True
args.headless = True
if not args.visualizer:
    args.visualizer = ["kit"]
from env.camera_manager.capture.render_sync import add_zero_delay_kit_args

add_zero_delay_kit_args(args)
app = AppLauncher(args).app

import isaacsim
import numpy as np
import omni.kit.app

extension_manager = omni.kit.app.get_app().get_extension_manager()
for package_path in isaacsim.__path__:
    extension_manager.add_path(str(Path(package_path) / "extsDeprecated"))

for extension in (
    "isaacsim.core.api",
    "isaacsim.core.prims",
    "isaacsim.sensors.camera",
    "isaacsim.replicator.behavior",
):
    extension_manager.set_extension_enabled_immediate(extension, True)

from omegaconf import OmegaConf

from env.global_configs import ENV_CONFIG_PATH, ROOT_DIR
from env.seed_manager.seed_manager import SeedManager
from task.RoboDojo.task_registry import load_task_class, task_config_path
from utils.load_file import load_yaml
from utils.pipeline_utils import process_config, process_randomization

root = Path(ROOT_DIR)
base = Path(ENV_CONFIG_PATH)
eval_cfg = load_yaml(base / "arx_x5.yml")
cfg = OmegaConf.create(
    {
        **{key: load_yaml(base / key / (name + ".yml")) for key, name in eval_cfg["config"].items()},
        "task_env": load_yaml(task_config_path(root / "task/RoboDojo/config", args.task)),
        "eval_cfg": eval_cfg,
    }
)
cfg = process_randomization(cfg)
cfg, _ = process_config(cfg, args.task)
cfg.sim.scene.num_envs = 1
cfg.sim.seed = [0]
cfg.camera.default_frequency = eval_cfg["observation"].get("collect_freq", 0)
if args.without_planner:
    for robot_cfg in cfg.robot.robots:
        robot_cfg.need_planner = False
_, task_cls = load_task_class(args.task)
env = task_cls(cfg, app)
cfg.eval_cfg.num_envs = 1
cfg.eval_cfg.task_name = args.task
seeds = SeedManager(cfg.eval_cfg)
seeds.init_eval()
env.scene_manager.layout_manager.set_saved_layout(0, seeds.get_seed_scene_info(0))
from env.camera_manager.capture.render_sync import wait_for_latest_cameras

for reset_index in range(args.resets):
    env.reset(seed=[0])
    env.scene_manager.apply_saved_poses(env_idx_list=[0])
    env.robot_manager.set_origin_endpose()
    for _ in range(30):
        env.sim_step()
    wait_for_latest_cameras(env, env.capture_manager, min_passes=20, max_passes=40)
    for robot in env.robot_manager.robot_list:
        assert np.isfinite(env.robot_manager.get_real_endpose(robot)[0]).all()
    print("RESET_SUCCESS", reset_index, flush=True)
frames = env.capture_manager.step()
args.output.mkdir(parents=True, exist_ok=True)
results = []
for index, camera in enumerate(frames):
    for name, batch in camera.items():
        image = np.asarray(batch[0]["data"])
        assert image.ndim == 3 and image.shape[-1] in (3, 4), (name, image.shape)
        assert np.isfinite(image).all(), (name, image.shape)
        print(
            "CAMERA_STATS",
            index,
            name,
            image.shape,
            float(image.min()),
            float(image.max()),
            float(image[..., :3].std()),
            flush=True,
        )
        np.save(args.output / f"camera-{index}-{name}.npy", image)
        results.append(
            {"camera": index, "annotator": name, "shape": list(image.shape), "std": float(image[..., :3].std())}
        )
assert results, "No native camera frames"
assert all(item["std"] > 1 for item in results), results
env.run_reward()
report = {
    "task": args.task,
    "cameras": results,
    "planner_enabled": not args.without_planner,
    "resets": args.resets,
    "robot_poses": [env.robot_manager.get_real_endpose(robot)[0].tolist() for robot in env.robot_manager.robot_list],
}
(args.output / "result.json").write_text(json.dumps(report, indent=2))
print("ROBODOJO_ISAAC61_SMOKE_SUCCESS", json.dumps(report), flush=True)
env.close()
app.close()
