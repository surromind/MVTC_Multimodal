"""멀티모달 샘플 로더 구현."""

from pathlib import Path
from typing import Iterable, Optional, TypedDict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from app.utils.path import DatasetPaths


class SampleKeyDict(TypedDict):
    """디렉터리/파일명에서 추출한 객체 ID와 샘플 인덱스."""

    obj_id: int
    sample_idx: int


def sample_key_from_filename(name: str) -> SampleKeyDict:
    """``{obj_id}_{idx}.ext`` 패턴의 파일명에서 키 정보를 파싱한다."""

    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 2:
        raise ValueError(f"잘못된 파일명 형식: {name}")
    return {"obj_id": int(parts[0]), "sample_idx": int(parts[1])}


class MultimodalSampleDict(TypedDict, total=False):
    """멀티모달 샘플의 경로·텍스트·촉각 텐서를 담는 딕셔너리."""

    key: SampleKeyDict
    image_path: Path
    text: str
    tactile_path: Path
    tactile_data: torch.Tensor | None


class MultimodalDataset(Dataset[MultimodalSampleDict]):
    def __init__(
        self,
        paths: DatasetPaths,
        csv_normalizer: Optional["TactileNormalizer"] = None,
        sequence_length: int | None = None,
    ) -> None:
        """필요한 경로와 전처리기를 받아 샘플 인덱스를 미리 구축한다."""

        self.paths = paths
        self.csv_normalizer = csv_normalizer
        self.sequence_length = sequence_length
        self.text_cache = self._load_texts(paths.text_dir)
        self.samples = self._index_samples()

    def _load_texts(self, text_dir: Path) -> dict[int, str]:
        """텍스트 파일을 미리 읽어 객체 ID → 설명 문장 매핑을 만든다."""

        texts: dict[int, str] = {}
        for txt_path in sorted(text_dir.glob("*.txt")):
            try:
                obj_id = int(txt_path.stem)
            except ValueError as exc:
                raise ValueError(
                    f"텍스트 파일명에서 숫자를 추출할 수 없습니다: {txt_path}"
                ) from exc
            texts[obj_id] = txt_path.read_text(encoding="utf-8").strip()
        return texts

    def _index_samples(self) -> list[MultimodalSampleDict]:
        """이미지·텍스트·CSV를 키 기준으로 매칭해 샘플 리스트를 생성한다."""

        samples: list[MultimodalSampleDict] = []
        csv_lookup: dict[tuple[int, int], Path] = {}
        for csv_path in sorted(self.paths.tactile_dir.glob("*.csv")):
            key = sample_key_from_filename(csv_path.name)
            csv_lookup[(key["obj_id"], key["sample_idx"])] = csv_path

        for img_dir in sorted(self.paths.image_dir.glob("*/")):
            if not img_dir.is_dir():
                continue
            try:
                obj_id = int(img_dir.name)
            except ValueError:
                continue
            text = self.text_cache.get(obj_id)
            if not text:
                continue
            for img_path in sorted(img_dir.glob("*.jpg")):
                key = sample_key_from_filename(img_path.name)
                csv_path = csv_lookup.get((key["obj_id"], key["sample_idx"]))
                if csv_path is None:
                    continue
                samples.append(
                    {
                        "key": key,
                        "image_path": img_path,
                        "text": text,
                        "tactile_path": csv_path,
                    }
                )
        if not samples:
            raise RuntimeError("유효한 멀티모달 샘플을 찾지 못했습니다.")
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> MultimodalSampleDict:
        """인덱스에 해당하는 샘플을 반환하며 촉각 CSV를 텐서로 변환한다."""

        sample = self.samples[index]
        if self.csv_normalizer is None:
            return sample
        tactile = self.csv_normalizer.load(sample["tactile_path"], self.sequence_length)
        return {
            "key": sample["key"],
            "image_path": sample["image_path"],
            "text": sample["text"],
            "tactile_path": sample["tactile_path"],
            "tactile_data": tactile,
        }


class TactileNormalizer:
    """CSV 데이터를 텐서로 변환하고 선택적으로 정규화한다."""

    def __init__(
        self,
        scaler: object | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.scaler = scaler
        self.device = device

    def load(
        self,
        csv_path: Path,
        sequence_length: int | None = None,
    ) -> torch.Tensor:
        """CSV 파일을 읽어 텐서로 변환하고 필요 시 정규화·길이 절단을 수행."""

        df = pd.read_csv(csv_path)
        values = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float32)
        if sequence_length is not None and values.shape[0] >= sequence_length:
            values = values[:sequence_length]
        if self.scaler is not None:
            values = self.scaler.transform(values)
        tensor = torch.from_numpy(values)
        if self.device is not None:
            tensor = tensor.to(self.device)
        return tensor


def collate_samples(batch: Iterable[MultimodalSampleDict]) -> None:
    """DataLoader에서 사용할 collate 함수는 학습 코드에서 별도 구현한다."""

    raise NotImplementedError("collate_samples는 학습 루프에서 정의해 주세요.")
