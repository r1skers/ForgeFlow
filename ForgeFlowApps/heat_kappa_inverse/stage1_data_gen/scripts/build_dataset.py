import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np

RawRow = dict[str, float | str | int]

OBS_FEATURE_FIELDS = (
    "l2_t1",
    "l2_t2",
    "mean_abs_t1",
    "mean_abs_t2",
    "max_abs_t1",
    "max_abs_t2",
    "decay_rate_l2",
)


def _load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("config root must be an object")
    return payload


def _resolve(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return project_root / path


def _initial_grid(nx: int, ny: int) -> np.ndarray:
    x = np.arange(nx, dtype=float) / float(nx)
    y = np.arange(ny, dtype=float) / float(ny)
    xx, yy = np.meshgrid(x, y)
    return np.sin(2.0 * math.pi * xx) * np.sin(2.0 * math.pi * yy)


def _step_periodic(grid: np.ndarray, alpha_x: float, alpha_y: float) -> np.ndarray:
    lap_x = np.roll(grid, -1, axis=1) - 2.0 * grid + np.roll(grid, 1, axis=1)
    lap_y = np.roll(grid, -1, axis=0) - 2.0 * grid + np.roll(grid, 1, axis=0)
    return grid + alpha_x * lap_x + alpha_y * lap_y


def _simulate_snapshots(
    *,
    nx: int,
    ny: int,
    dt: float,
    dx: float,
    dy: float,
    kappa: float,
    t1: float,
    t2: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    step_t1 = max(1, int(round(t1 / dt)))
    step_t2 = max(step_t1 + 1, int(round(t2 / dt)))
    actual_t1 = step_t1 * dt
    actual_t2 = step_t2 * dt

    grid = _initial_grid(nx, ny)
    alpha_x = kappa * dt / (dx * dx)
    alpha_y = kappa * dt / (dy * dy)
    snapshot_t1 = grid.copy()

    for step in range(1, step_t2 + 1):
        grid = _step_periodic(grid, alpha_x, alpha_y)
        if step == step_t1:
            snapshot_t1 = grid.copy()

    return snapshot_t1, grid, actual_t1, actual_t2


def _extract_features(grid_t1: np.ndarray, grid_t2: np.ndarray, delta_t: float) -> dict[str, float]:
    eps = 1e-12
    l2_t1 = float(math.sqrt(float(np.mean(np.square(grid_t1)))))
    l2_t2 = float(math.sqrt(float(np.mean(np.square(grid_t2)))))
    mean_abs_t1 = float(np.mean(np.abs(grid_t1)))
    mean_abs_t2 = float(np.mean(np.abs(grid_t2)))
    max_abs_t1 = float(np.max(np.abs(grid_t1)))
    max_abs_t2 = float(np.max(np.abs(grid_t2)))
    ratio = max(l2_t2 / max(l2_t1, eps), eps)
    decay_rate_l2 = float(-math.log(ratio) / delta_t)
    return {
        "l2_t1": l2_t1,
        "l2_t2": l2_t2,
        "mean_abs_t1": mean_abs_t1,
        "mean_abs_t2": mean_abs_t2,
        "max_abs_t1": max_abs_t1,
        "max_abs_t2": max_abs_t2,
        "decay_rate_l2": decay_rate_l2,
    }


def _sample_range(rng: random.Random, lo: float, hi: float) -> float:
    if hi <= lo:
        raise ValueError(f"invalid range: [{lo}, {hi}]")
    return rng.uniform(lo, hi)


def _build_split_rows(
    *,
    split: str,
    size: int,
    kappa_lo: float,
    kappa_hi: float,
    nx: int,
    ny: int,
    dt: float,
    t1_min: float,
    t1_max: float,
    t2_min: float,
    t2_max: float,
    rng: random.Random,
    include_y: bool,
) -> list[RawRow]:
    rows: list[RawRow] = []
    dx = 1.0 / float(nx)
    dy = 1.0 / float(ny)

    for idx in range(size):
        kappa = _sample_range(rng, kappa_lo, kappa_hi)
        t1 = _sample_range(rng, t1_min, t1_max)
        t2 = _sample_range(rng, max(t2_min, t1 + dt), t2_max)

        grid_t1, grid_t2, actual_t1, actual_t2 = _simulate_snapshots(
            nx=nx,
            ny=ny,
            dt=dt,
            dx=dx,
            dy=dy,
            kappa=kappa,
            t1=t1,
            t2=t2,
        )
        delta_t = actual_t2 - actual_t1
        feats = _extract_features(grid_t1, grid_t2, delta_t)

        row: RawRow = {
            "sample_id": idx,
            "split": split,
            "kappa": kappa,
            "y": kappa if include_y else "",
            "noise_std": 0.0,
            "t1": actual_t1,
            "t2": actual_t2,
            "delta_t": delta_t,
            "l2_t1": feats["l2_t1"],
            "l2_t2": feats["l2_t2"],
            "mean_abs_t1": feats["mean_abs_t1"],
            "mean_abs_t2": feats["mean_abs_t2"],
            "max_abs_t1": feats["max_abs_t1"],
            "max_abs_t2": feats["max_abs_t2"],
            "decay_rate_l2": feats["decay_rate_l2"],
        }
        rows.append(row)
    return rows


def _apply_observation_noise(rows: list[RawRow], noise_std: float, rng: random.Random) -> list[RawRow]:
    if noise_std < 0:
        raise ValueError("noise_std must be >= 0")
    noisy_rows: list[RawRow] = []
    for row in rows:
        noisy = dict(row)
        noisy["noise_std"] = float(noise_std)
        for key in OBS_FEATURE_FIELDS:
            base = float(noisy[key])
            scale = max(abs(base), 1e-9)
            noisy[key] = base + rng.gauss(0.0, noise_std * scale)
        noisy_rows.append(noisy)
    return noisy_rows


def _format_row(row: RawRow) -> dict[str, str]:
    formatted: dict[str, str] = {}
    for key, value in row.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.12f}"
        else:
            formatted[key] = str(value)
    return formatted


def _format_rows(rows: list[RawRow]) -> list[dict[str, str]]:
    return [_format_row(row) for row in rows]


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "split",
        "kappa",
        "y",
        "noise_std",
        "t1",
        "t2",
        "delta_t",
        "l2_t1",
        "l2_t2",
        "mean_abs_t1",
        "mean_abs_t2",
        "max_abs_t1",
        "max_abs_t2",
        "decay_rate_l2",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _noise_tag(noise_std: float) -> str:
    return f"{noise_std:.2f}".replace(".", "p")


def _noisy_output_path(base_csv: Path, noise_std: float) -> Path:
    tag = _noise_tag(noise_std)
    return base_csv.with_name(f"{base_csv.stem}_noise_{tag}{base_csv.suffix}")


def _write_manifest(
    *,
    path: Path,
    seed: int,
    nx: int,
    ny: int,
    dt: float,
    cfl_limit: float,
    train_rows: int,
    infer_id_rows: int,
    infer_ood_rows: int,
    noise_levels: list[float],
    noisy_files: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "seed",
                "nx",
                "ny",
                "dt",
                "cfl_limit",
                "train_rows",
                "infer_id_rows",
                "infer_ood_rows",
                "noise_levels",
                "noisy_files",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "seed": seed,
                "nx": nx,
                "ny": ny,
                "dt": f"{dt:.12f}",
                "cfl_limit": f"{cfl_limit:.12f}",
                "train_rows": train_rows,
                "infer_id_rows": infer_id_rows,
                "infer_ood_rows": infer_ood_rows,
                "noise_levels": ",".join(f"{level:.2f}" for level in noise_levels),
                "noisy_files": noisy_files,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build random-kappa heat inverse datasets.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ForgeFlowApps/heat_kappa_inverse/stage1_data_gen/config/generate.json"),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[4]
    payload = _load_config(_resolve(project_root, str(args.config)))

    seed = int(payload["seed"])
    rng = random.Random(seed)

    grid_payload = payload["grid"]
    nx = int(grid_payload["nx"])
    ny = int(grid_payload["ny"])
    if nx <= 0 or ny <= 0:
        raise ValueError("grid nx/ny must be > 0")

    sim_payload = payload["simulation"]
    dt = float(sim_payload["dt"])
    t1_min = float(sim_payload["t1_min"])
    t1_max = float(sim_payload["t1_max"])
    t2_min = float(sim_payload["t2_min"])
    t2_max = float(sim_payload["t2_max"])
    if not (0.0 < t1_min < t1_max < t2_max):
        raise ValueError("invalid t1/t2 bounds")

    noise_levels = sorted(
        {float(level) for level in payload.get("noise_levels", [0.01, 0.03]) if float(level) > 0.0}
    )

    ranges = payload["kappa_ranges"]
    train_lo, train_hi = float(ranges["train"][0]), float(ranges["train"][1])
    id_lo, id_hi = float(ranges["infer_id"][0]), float(ranges["infer_id"][1])
    ood_lo, ood_hi = float(ranges["infer_ood"][0]), float(ranges["infer_ood"][1])

    max_kappa = max(train_hi, id_hi, ood_hi)
    dx = 1.0 / float(nx)
    dy = 1.0 / float(ny)
    cfl_limit = min(dx * dx, dy * dy) / (4.0 * max_kappa)
    if dt > cfl_limit:
        raise ValueError(f"dt={dt} exceeds CFL limit={cfl_limit}")

    sizes = payload["sizes"]
    n_train = int(sizes["train"])
    n_infer_id = int(sizes["infer_id"])
    n_infer_ood = int(sizes["infer_ood"])

    train_rows = _build_split_rows(
        split="train",
        size=n_train,
        kappa_lo=train_lo,
        kappa_hi=train_hi,
        nx=nx,
        ny=ny,
        dt=dt,
        t1_min=t1_min,
        t1_max=t1_max,
        t2_min=t2_min,
        t2_max=t2_max,
        rng=rng,
        include_y=False,
    )
    infer_id_rows = _build_split_rows(
        split="infer_id",
        size=n_infer_id,
        kappa_lo=id_lo,
        kappa_hi=id_hi,
        nx=nx,
        ny=ny,
        dt=dt,
        t1_min=t1_min,
        t1_max=t1_max,
        t2_min=t2_min,
        t2_max=t2_max,
        rng=rng,
        include_y=True,
    )
    infer_ood_rows = _build_split_rows(
        split="infer_ood",
        size=n_infer_ood,
        kappa_lo=ood_lo,
        kappa_hi=ood_hi,
        nx=nx,
        ny=ny,
        dt=dt,
        t1_min=t1_min,
        t1_max=t1_max,
        t2_min=t2_min,
        t2_max=t2_max,
        rng=rng,
        include_y=True,
    )

    paths = payload["paths"]
    train_csv = _resolve(project_root, str(paths["train_csv"]))
    infer_id_csv = _resolve(project_root, str(paths["infer_id_csv"]))
    infer_ood_csv = _resolve(project_root, str(paths["infer_ood_csv"]))
    manifest_csv = _resolve(project_root, str(paths["manifest_csv"]))

    _write_rows(train_csv, _format_rows(train_rows))
    _write_rows(infer_id_csv, _format_rows(infer_id_rows))
    _write_rows(infer_ood_csv, _format_rows(infer_ood_rows))

    noisy_outputs: list[Path] = []
    for level in noise_levels:
        id_noise_rng = random.Random(seed + 101 + int(level * 1_000_000))
        ood_noise_rng = random.Random(seed + 202 + int(level * 1_000_000))
        infer_id_noisy = _apply_observation_noise(infer_id_rows, level, id_noise_rng)
        infer_ood_noisy = _apply_observation_noise(infer_ood_rows, level, ood_noise_rng)

        infer_id_noisy_csv = _noisy_output_path(infer_id_csv, level)
        infer_ood_noisy_csv = _noisy_output_path(infer_ood_csv, level)
        _write_rows(infer_id_noisy_csv, _format_rows(infer_id_noisy))
        _write_rows(infer_ood_noisy_csv, _format_rows(infer_ood_noisy))
        noisy_outputs.extend([infer_id_noisy_csv, infer_ood_noisy_csv])

    _write_manifest(
        path=manifest_csv,
        seed=seed,
        nx=nx,
        ny=ny,
        dt=dt,
        cfl_limit=cfl_limit,
        train_rows=len(train_rows),
        infer_id_rows=len(infer_id_rows),
        infer_ood_rows=len(infer_ood_rows),
        noise_levels=noise_levels,
        noisy_files=len(noisy_outputs),
    )

    print(f"[kappa-data] train_rows={len(train_rows)}")
    print(f"[kappa-data] infer_id_rows={len(infer_id_rows)}")
    print(f"[kappa-data] infer_ood_rows={len(infer_ood_rows)}")
    print(f"[kappa-data] noise_levels={','.join(f'{level:.2f}' for level in noise_levels)}")
    print(f"[kappa-data] cfl_limit={cfl_limit:.12f}")
    print(f"[kappa-data] train_csv={train_csv.resolve()}")
    print(f"[kappa-data] infer_id_csv={infer_id_csv.resolve()}")
    print(f"[kappa-data] infer_ood_csv={infer_ood_csv.resolve()}")
    for noisy_csv in noisy_outputs:
        print(f"[kappa-data] noisy_csv={noisy_csv.resolve()}")
    print(f"[kappa-data] manifest_csv={manifest_csv.resolve()}")


if __name__ == "__main__":
    main()

