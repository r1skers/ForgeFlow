from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build realistic 1D heat-PDE prediction dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ForgeFlowApps/heat1d_realistic_predict/config/generate.json"),
        help="Path to dataset generation config JSON.",
    )
    return parser


def resolve_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError("config must be a JSON object")
    return payload


def gaussian_profile(x: NDArray[np.float64], center: float, width: float, amplitude: float) -> NDArray[np.float64]:
    safe_width = max(width, 1e-6)
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * safe_width**2))


def simulate_truth(
    cfg: dict[str, Any], rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], dict[str, float]]:
    grid_cfg = cfg["grid"]
    time_cfg = cfg["time"]
    dyn_cfg = cfg["dynamics"]
    init_cfg = cfg["initial_condition"]

    nx = int(grid_cfg["nx"])
    dx = float(grid_cfg["dx"])
    steps = int(time_cfg["steps"])
    dt = float(time_cfg["dt"])
    if nx < 3:
        raise ValueError("grid.nx must be >= 3")
    if dx <= 0 or dt <= 0:
        raise ValueError("grid.dx and time.dt must be > 0")
    if steps <= 1:
        raise ValueError("time.steps must be > 1")

    x = np.arange(nx, dtype=float) * dx

    kappa_mean = float(dyn_cfg["kappa_mean"])
    kappa_variation = float(dyn_cfg["kappa_variation"])
    advection_v = float(dyn_cfg["advection_v"])
    process_noise_std = float(dyn_cfg["process_noise_std"])
    source_amp = float(dyn_cfg["source_amp"])
    source_freq_hz = float(dyn_cfg["source_freq_hz"])
    source_center_x = float(dyn_cfg["source_center_x"])
    source_width = float(dyn_cfg["source_width"])

    phase = 2.0 * math.pi * (x / max(float(nx * dx), 1e-9))
    kappa = kappa_mean * (1.0 + kappa_variation * np.sin(phase))
    kappa = np.maximum(kappa, 1e-6)

    u_true = np.zeros((steps + 1, nx), dtype=float)
    u_true[0, :] = float(init_cfg["baseline"])
    for peak in init_cfg["peaks"]:
        u_true[0, :] += gaussian_profile(
            x=x,
            center=float(peak["center_x"]),
            width=float(peak["width"]),
            amplitude=float(peak["amplitude"]),
        )

    source_profile = np.exp(-((x - source_center_x) ** 2) / (2.0 * max(source_width, 1e-6) ** 2))
    source_profile = source_profile / max(float(np.max(np.abs(source_profile))), 1e-9)

    for n in range(steps):
        current = u_true[n, :]
        left = np.roll(current, 1)
        right = np.roll(current, -1)

        laplacian = (left - 2.0 * current + right) / (dx * dx)
        if advection_v >= 0.0:
            gradient = (current - left) / dx
        else:
            gradient = (right - current) / dx

        forcing = source_amp * math.sin(2.0 * math.pi * source_freq_hz * (n * dt))
        process_noise = rng.normal(loc=0.0, scale=process_noise_std, size=nx)

        u_true[n + 1, :] = current + dt * (kappa * laplacian - advection_v * gradient + forcing * source_profile)
        u_true[n + 1, :] += process_noise

    diff_limit = (dx * dx) / (2.0 * float(np.max(kappa)))
    adv_limit = float("inf") if abs(advection_v) < 1e-12 else dx / abs(advection_v)
    stable_dt_limit = min(diff_limit, adv_limit)
    stability = {
        "cfl_limit_dt": stable_dt_limit,
        "stable_cfl": float(dt <= stable_dt_limit),
        "diff_limit_dt": diff_limit,
        "adv_limit_dt": adv_limit,
    }

    return x, kappa, u_true, stability


def apply_observation_model(
    u_true: NDArray[np.float64], cfg: dict[str, Any], rng: np.random.Generator
) -> tuple[NDArray[np.float64], NDArray[np.bool_], NDArray[np.bool_], NDArray[np.bool_], NDArray[np.float64]]:
    obs = u_true.copy()
    steps_plus_one, nx = obs.shape

    gaussian_std = float(cfg["gaussian_std"])
    heavy_tail_std = float(cfg["heavy_tail_std"])
    heavy_tail_prob = float(cfg["heavy_tail_prob"])
    missing_prob = float(cfg["missing_prob"])
    missing_block_prob = float(cfg["missing_block_prob"])
    missing_block_len_min = int(cfg["missing_block_len_min"])
    missing_block_len_max = int(cfg["missing_block_len_max"])
    outlier_prob = float(cfg["outlier_prob"])
    outlier_scale = float(cfg["outlier_scale"])
    stuck_prob = float(cfg["stuck_prob"])
    stuck_len_min = int(cfg["stuck_len_min"])
    stuck_len_max = int(cfg["stuck_len_max"])
    drift_std = float(cfg["drift_std"])
    quantization_step = float(cfg["quantization_step"])
    clip_min = float(cfg["clip_min"])
    clip_max = float(cfg["clip_max"])

    obs += rng.normal(loc=0.0, scale=gaussian_std, size=obs.shape)

    heavy_mask = rng.random(obs.shape) < heavy_tail_prob
    heavy_noise = rng.standard_t(df=3, size=obs.shape) * heavy_tail_std
    obs += heavy_mask * heavy_noise

    drift_end = rng.normal(loc=0.0, scale=drift_std, size=nx)
    drift_factor = np.linspace(0.0, 1.0, steps_plus_one, dtype=float)
    drift = drift_factor[:, None] * drift_end[None, :]
    obs += drift

    outlier_mask = rng.random(obs.shape) < outlier_prob
    outlier_sigma = max(float(np.std(u_true)), 1e-6) * outlier_scale
    obs += outlier_mask * rng.normal(loc=0.0, scale=outlier_sigma, size=obs.shape)

    stuck_mask = np.zeros(obs.shape, dtype=bool)
    for sensor in range(nx):
        t = 0
        while t < steps_plus_one:
            if rng.random() < stuck_prob:
                length = int(rng.integers(low=stuck_len_min, high=stuck_len_max + 1))
                end = min(steps_plus_one, t + max(length, 1))
                hold = obs[t - 1, sensor] if t > 0 else obs[t, sensor]
                obs[t:end, sensor] = hold
                stuck_mask[t:end, sensor] = True
                t = end
            else:
                t += 1

    missing_mask = rng.random(obs.shape) < missing_prob
    for sensor in range(nx):
        t = 0
        while t < steps_plus_one:
            if rng.random() < missing_block_prob:
                length = int(rng.integers(low=missing_block_len_min, high=missing_block_len_max + 1))
                end = min(steps_plus_one, t + max(length, 1))
                missing_mask[t:end, sensor] = True
                t = end
            else:
                t += 1

    obs = np.clip(obs, clip_min, clip_max)
    if quantization_step > 0.0:
        obs = np.round(obs / quantization_step) * quantization_step

    obs[missing_mask] = np.nan
    return obs, missing_mask, outlier_mask, stuck_mask, drift


def forward_fill_nan(obs: NDArray[np.float64]) -> NDArray[np.float64]:
    filled = obs.copy()
    steps_plus_one, nx = filled.shape
    global_mean = float(np.nanmean(obs)) if not math.isnan(float(np.nanmean(obs))) else 0.0

    for sensor in range(nx):
        last_value = math.nan
        for t in range(steps_plus_one):
            value = filled[t, sensor]
            if math.isnan(float(value)):
                if not math.isnan(last_value):
                    filled[t, sensor] = last_value
            else:
                last_value = float(value)

        sensor_col = filled[:, sensor]
        valid = sensor_col[~np.isnan(sensor_col)]
        if valid.size == 0:
            filled[:, sensor] = global_mean
        else:
            first_valid = float(valid[0])
            sensor_col[np.isnan(sensor_col)] = first_valid
            filled[:, sensor] = sensor_col

    filled = np.where(np.isnan(filled), global_mean, filled)
    return filled


def write_trajectory_csv(
    path: Path,
    u_true: NDArray[np.float64],
    u_obs: NDArray[np.float64],
    missing_mask: NDArray[np.bool_],
    outlier_mask: NDArray[np.bool_],
    stuck_mask: NDArray[np.bool_],
    drift: NDArray[np.float64],
    x: NDArray[np.float64],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "step",
                "sensor_i",
                "x",
                "u_true",
                "u_obs",
                "is_missing",
                "is_outlier",
                "is_stuck",
                "drift",
            ]
        )
        steps_plus_one, nx = u_true.shape
        for t in range(steps_plus_one):
            for i in range(nx):
                obs_value = u_obs[t, i]
                writer.writerow(
                    [
                        t,
                        i,
                        float(x[i]),
                        float(u_true[t, i]),
                        "" if np.isnan(obs_value) else float(obs_value),
                        int(missing_mask[t, i]),
                        int(outlier_mask[t, i]),
                        int(stuck_mask[t, i]),
                        float(drift[t, i]),
                    ]
                )


def build_supervised_rows(
    x: NDArray[np.float64],
    u_true: NDArray[np.float64],
    u_obs: NDArray[np.float64],
    u_obs_filled: NDArray[np.float64],
    missing_mask: NDArray[np.bool_],
    outlier_mask: NDArray[np.bool_],
    stuck_mask: NDArray[np.bool_],
    train_until_step: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, int]]:
    steps_plus_one, nx = u_true.shape
    total_steps = steps_plus_one - 1
    if not (0 < train_until_step < total_steps):
        raise ValueError("split.train_until_step must be between 1 and steps-1")

    train_rows: list[dict[str, str]] = []
    infer_rows: list[dict[str, str]] = []
    skipped_missing_target = 0

    for t in range(total_steps):
        t_norm = float(t) / float(total_steps)
        for i in range(nx):
            target_obs = u_obs[t + 1, i]
            if np.isnan(target_obs):
                skipped_missing_target += 1
                continue

            left_i = (i - 1) % nx
            right_i = (i + 1) % nx
            row = {
                "step_t": str(t),
                "sensor_i": str(i),
                "x": f"{float(x[i]):.6f}",
                "x_norm": f"{float(i) / float(max(nx - 1, 1)):.6f}",
                "t_norm": f"{t_norm:.6f}",
                "obs_center": f"{float(u_obs_filled[t, i]):.6f}",
                "obs_left": f"{float(u_obs_filled[t, left_i]):.6f}",
                "obs_right": f"{float(u_obs_filled[t, right_i]):.6f}",
                "miss_center": str(int(missing_mask[t, i])),
                "miss_left": str(int(missing_mask[t, left_i])),
                "miss_right": str(int(missing_mask[t, right_i])),
                "outlier_center": str(int(outlier_mask[t, i])),
                "stuck_center": str(int(stuck_mask[t, i])),
                "y": f"{float(target_obs):.6f}",
            }
            if t <= train_until_step:
                train_rows.append(row)
            else:
                infer_rows.append(row)

    stats = {
        "train_rows": len(train_rows),
        "infer_rows": len(infer_rows),
        "skipped_missing_target": skipped_missing_target,
    }
    return train_rows, infer_rows, stats


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "step_t",
        "sensor_i",
        "x",
        "x_norm",
        "t_norm",
        "obs_center",
        "obs_left",
        "obs_right",
        "miss_center",
        "miss_left",
        "miss_right",
        "outlier_center",
        "stuck_center",
        "y",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[3]
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    cfg = load_config(config_path)

    rng = np.random.default_rng(int(cfg["random_seed"]))
    x, kappa, u_true, stability = simulate_truth(cfg, rng)
    u_obs, missing_mask, outlier_mask, stuck_mask, drift = apply_observation_model(
        u_true=u_true,
        cfg=cfg["observation"],
        rng=rng,
    )
    u_obs_filled = forward_fill_nan(u_obs)

    train_rows, infer_rows, row_stats = build_supervised_rows(
        x=x,
        u_true=u_true,
        u_obs=u_obs,
        u_obs_filled=u_obs_filled,
        missing_mask=missing_mask,
        outlier_mask=outlier_mask,
        stuck_mask=stuck_mask,
        train_until_step=int(cfg["split"]["train_until_step"]),
    )
    if not train_rows or not infer_rows:
        raise RuntimeError("generated train/infer rows are empty; adjust split or corruption rates")

    outputs = cfg["outputs"]
    train_csv = resolve_path(project_root, str(outputs["train_csv"]))
    infer_csv = resolve_path(project_root, str(outputs["infer_csv"]))
    true_traj_csv = resolve_path(project_root, str(outputs["true_trajectory_csv"]))
    obs_traj_csv = resolve_path(project_root, str(outputs["obs_trajectory_csv"]))
    manifest_json = resolve_path(project_root, str(outputs["manifest_json"]))

    write_rows(train_csv, train_rows)
    write_rows(infer_csv, infer_rows)
    write_trajectory_csv(
        path=true_traj_csv,
        u_true=u_true,
        u_obs=u_true,
        missing_mask=np.zeros_like(missing_mask, dtype=bool),
        outlier_mask=np.zeros_like(outlier_mask, dtype=bool),
        stuck_mask=np.zeros_like(stuck_mask, dtype=bool),
        drift=np.zeros_like(drift, dtype=float),
        x=x,
    )
    write_trajectory_csv(
        path=obs_traj_csv,
        u_true=u_true,
        u_obs=u_obs,
        missing_mask=missing_mask,
        outlier_mask=outlier_mask,
        stuck_mask=stuck_mask,
        drift=drift,
        x=x,
    )

    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "config": str(config_path),
        "train_csv": str(train_csv),
        "infer_csv": str(infer_csv),
        "true_trajectory_csv": str(true_traj_csv),
        "obs_trajectory_csv": str(obs_traj_csv),
        "grid_nx": int(cfg["grid"]["nx"]),
        "steps": int(cfg["time"]["steps"]),
        "train_until_step": int(cfg["split"]["train_until_step"]),
        "kappa_min": float(np.min(kappa)),
        "kappa_max": float(np.max(kappa)),
        "missing_ratio": float(np.mean(missing_mask)),
        "outlier_ratio": float(np.mean(outlier_mask)),
        "stuck_ratio": float(np.mean(stuck_mask)),
        "stable_cfl": bool(stability["stable_cfl"]),
        "cfl_limit_dt": float(stability["cfl_limit_dt"]),
        "dt": float(cfg["time"]["dt"]),
        "train_rows": row_stats["train_rows"],
        "infer_rows": row_stats["infer_rows"],
        "skipped_missing_target": row_stats["skipped_missing_target"],
    }
    with manifest_json.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=True, indent=2)

    print(f"[build] config={config_path}")
    print(f"[build] stable_cfl={manifest['stable_cfl']}")
    print(f"[build] cfl_limit_dt={manifest['cfl_limit_dt']:.6f}")
    print(f"[build] dt={manifest['dt']:.6f}")
    print(f"[build] train_rows={manifest['train_rows']}")
    print(f"[build] infer_rows={manifest['infer_rows']}")
    print(f"[build] missing_ratio={manifest['missing_ratio']:.4f}")
    print(f"[build] outlier_ratio={manifest['outlier_ratio']:.4f}")
    print(f"[build] stuck_ratio={manifest['stuck_ratio']:.4f}")
    print(f"[build] train_csv={train_csv}")
    print(f"[build] infer_csv={infer_csv}")
    print(f"[build] manifest={manifest_json}")


if __name__ == "__main__":
    main()
