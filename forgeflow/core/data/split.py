import random

from forgeflow.interfaces import FeatureMatrix, SplitStats


def split_train_val(
    x: FeatureMatrix,
    y: FeatureMatrix,
    train_ratio: float = 0.8,
    shuffle: bool = False,
    seed: int | None = 42,
) -> tuple[FeatureMatrix, FeatureMatrix, FeatureMatrix, FeatureMatrix, SplitStats]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of samples")

    total_samples = len(x)
    if total_samples < 2:
        raise ValueError("at least 2 samples are required for train/val split")

    train_samples = int(total_samples * train_ratio)
    train_samples = min(max(train_samples, 1), total_samples - 1)
    val_samples = total_samples - train_samples

    if shuffle:
        indices = list(range(total_samples))
        random.Random(seed).shuffle(indices)
        ordered_x = [x[idx] for idx in indices]
        ordered_y = [y[idx] for idx in indices]
    else:
        ordered_x = x
        ordered_y = y

    x_train = ordered_x[:train_samples]
    y_train = ordered_y[:train_samples]
    x_val = ordered_x[train_samples:]
    y_val = ordered_y[train_samples:]

    stats: SplitStats = {
        "total_samples": total_samples,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "train_ratio_pct": int(train_ratio * 100),
    }
    return x_train, y_train, x_val, y_val, stats
