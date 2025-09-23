"""임베딩을 생성·저장·시각화하는 유틸 함수 모음."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, NamedTuple

import numpy as np

from app.core import (
    ConcatenationFusion,
    EmbeddingVisualizer,
    MultimodalSampleDict,
)

class EmbeddingBatch(NamedTuple):
    """임베딩 배열과 파생 지표를 함께 보관하는 명명된 튜플."""

    reference: np.ndarray
    tactile: np.ndarray
    concatenated: np.ndarray
    labels: np.ndarray
    keys: np.ndarray
    num_samples: int
    reference_dim: int
    tactile_dim: int
    concat_dim: int


def collect_embeddings(
    fusion: ConcatenationFusion,
    dataset: Iterable[MultimodalSampleDict],
) -> EmbeddingBatch:
    """데이터셋을 순회하며 모든 임베딩을 수집한다."""

    references: list[np.ndarray] = []
    tactile_latents: list[np.ndarray] = []
    concatenated: list[np.ndarray] = []
    labels: list[int] = []
    keys: list[str] = []

    for sample in dataset:
        if sample["tactile_data"] is None:
            raise RuntimeError("정규화된 촉각 텐서를 로드하지 못했습니다.")

        # 1) 이미지/텍스트/촉각을 융합 임베딩으로 변환
        embedding = fusion.encode(
            image_path=str(sample["image_path"]),
            text=sample["text"],
            tactile_sequence=sample["tactile_data"],
        )

        # 2) 모달리티별 배열과 라벨 정보를 누적
        references.append(embedding["reference"])
        tactile_latents.append(embedding["tactile"])
        concatenated.append(embedding["concatenated"])
        labels.append(int(sample["key"]["obj_id"]))
        keys.append(
            f"{sample['key']['obj_id']}_{sample['key']['sample_idx']}"
        )

    if not references:
        raise RuntimeError("데이터셋에서 유효한 샘플을 찾지 못했습니다.")

    reference_arr = np.stack(references, axis=0)
    tactile_arr = np.stack(tactile_latents, axis=0)
    concat_arr = np.stack(concatenated, axis=0)
    label_arr = np.array(labels, dtype=np.int32)
    key_arr = np.array(keys, dtype=np.str_)

    return EmbeddingBatch(
        reference=reference_arr,
        tactile=tactile_arr,
        concatenated=concat_arr,
        labels=label_arr,
        keys=key_arr,
        num_samples=int(reference_arr.shape[0]),
        reference_dim=int(reference_arr.shape[-1]),
        tactile_dim=int(tactile_arr.shape[-1]),
        concat_dim=int(concat_arr.shape[-1]),
    )


def save_embeddings(batch: EmbeddingBatch, output_path: Path) -> None:
    """임베딩 배치를 ``npz`` 압축 파일로 저장한다."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        reference=batch.reference,
        tactile=batch.tactile,
        concatenated=batch.concatenated,
        labels=batch.labels,
        keys=batch.keys,
    )


def build_metadata(batch: EmbeddingBatch, config_path: Path) -> dict[str, object]:
    """임베딩 배치로부터 요약 정보를 계산해 메타데이터 딕셔너리를 만든다."""

    return {
        "config": str(config_path.resolve()),
        "num_samples": batch.num_samples,
        "reference_dim": batch.reference_dim,
        "tactile_dim": batch.tactile_dim,
        "concat_dim": batch.concat_dim,
        "labels": sorted({int(lbl) for lbl in batch.labels}),
    }


def write_metadata(metadata: dict[str, object], metadata_path: Path) -> None:
    """메타데이터를 JSON 파일로 기록한다."""

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))


def visualize_embeddings(
    batch: EmbeddingBatch,
    figure_dir: Path,
    stem: str,
    *,
    tsne_perplexity: float,
    umap_n_neighbors: int,
    umap_min_dist: float,
    seed: int,
) -> None:
    """임베딩을 차원 축소하여 PNG 시각화 파일을 생성한다."""

    figure_dir.mkdir(parents=True, exist_ok=True)
    EmbeddingVisualizer.save_all(
        concat=batch.concatenated,
        labels=batch.labels,
        output_dir=figure_dir,
        stem=stem,
        tsne_perplexity=tsne_perplexity,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        seed=seed,
    )
