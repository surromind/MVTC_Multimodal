"""CSV 기반 특징 스케일러."""

from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd


class FeatureScaler:
    """Incremental feature scaler supporting z-score and min-max modes."""

    def __init__(
        self, mode: Literal["zscore", "minmax"] = "zscore", eps: float = 1e-6
    ) -> None:
        self.mode = mode
        self.eps = float(eps)
        self.num_features: int | None = None
        self._count: int = 0
        self._sum: np.ndarray | None = None
        self._sumsq: np.ndarray | None = None
        self._min: np.ndarray | None = None
        self._max: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.min: np.ndarray | None = None
        self.max: np.ndarray | None = None
        self.fitted: bool = False

    def _prepare_stats(self, num_features: int) -> None:
        if self.num_features is None:
            self.num_features = num_features
            self._sum = np.zeros(num_features, dtype=np.float64)
            self._sumsq = np.zeros(num_features, dtype=np.float64)
            self._min = np.full(num_features, np.inf, dtype=np.float64)
            self._max = np.full(num_features, -np.inf, dtype=np.float64)
        elif self.num_features != num_features:
            raise ValueError("입력 feature 수가 이전과 일치하지 않습니다.")

    def update(self, arr: np.ndarray) -> None:
        if arr.ndim != 2:
            raise ValueError("입력은 (L, F) 2차원 배열이어야 합니다.")
        if arr.size == 0:
            return
        self._prepare_stats(arr.shape[1])
        assert self._sum is not None and self._sumsq is not None
        assert self._min is not None and self._max is not None
        self._count += int(arr.shape[0])
        self._sum += arr.sum(axis=0, dtype=np.float64)
        self._sumsq += (arr.astype(np.float64) ** 2).sum(axis=0)
        self._min = np.minimum(self._min, arr.min(axis=0).astype(np.float64))
        self._max = np.maximum(self._max, arr.max(axis=0).astype(np.float64))

    def finalize(self) -> None:
        if self._count == 0 or self.num_features is None:
            raise RuntimeError("스케일러를 학습할 데이터가 없습니다.")
        assert self._sum is not None and self._sumsq is not None
        mean = self._sum / float(self._count)
        var = (self._sumsq / float(self._count)) - mean**2
        var = np.maximum(var, self.eps)
        std = np.sqrt(var)
        self.mean = mean.astype(np.float32)
        self.std = std.astype(np.float32)
        if self._min is not None:
            self.min = self._min.astype(np.float32)
        if self._max is not None:
            max_candidates = self._max.astype(np.float32)
            if self.min is not None:
                max_candidates = np.maximum(max_candidates, self.min + self.eps)
            self.max = max_candidates
        self.fitted = True

    def fit_arrays(self, arrays: Iterable[np.ndarray]) -> None:
        for arr in arrays:
            self.update(arr)
        self.finalize()

    def fit_directory(self, directory: Path, pattern: str = "*.csv") -> None:
        paths = sorted(directory.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {directory}")
        for csv_path in paths:
            df = pd.read_csv(csv_path)
            arr = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
            if arr.size == 0:
                continue
            self.update(arr)
        self.finalize()

    def transform(self, arr: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("스케일러가 아직 학습되지 않았습니다.")
        if arr.ndim != 2:
            raise ValueError("입력은 (L, F) 2차원 배열이어야 합니다.")
        if self.num_features is not None and arr.shape[1] != self.num_features:
            raise ValueError("feature 수가 스케일러와 다릅니다.")
        if self.mode == "zscore":
            assert self.mean is not None and self.std is not None
            return (arr - self.mean) / (self.std + self.eps)
        assert self.min is not None and self.max is not None
        return (arr - self.min) / (self.max - self.min + self.eps)
