"""임베딩을 t-SNE/UMAP으로 시각화해 저장하는 도우미 함수들."""

from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from umap import UMAP 


class VisualizationResultDict(TypedDict, total=False):
    """시각화 결과 파일 경로를 담는 딕셔너리."""

    tsne_path: Path | None
    umap_path: Path | None


def _scatter(coords: np.ndarray, labels: np.ndarray, title: str, path: Path) -> None:
    """공통 산점도 렌더링 루틴."""

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    unique_labels = np.unique(labels)
    colors = plt.get_cmap("tab10", len(unique_labels))
    for idx, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=20,
            alpha=0.75,
            color=colors(idx),
            label=f"ID {label}",
        )
    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_tsne(
    concat: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    stem: str,
    perplexity: float,
    seed: int,
) -> Path:
    """t-SNE 결과를 계산해 파일로 저장하고 경로를 반환한다."""

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=seed,
    )
    tsne_coords = tsne.fit_transform(concat)
    path = output_dir / f"{stem}_tsne.png"
    _scatter(tsne_coords, labels, "Concat Embeddings (t-SNE)", path)
    return path


def save_umap(
    concat: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    stem: str,
    n_neighbors: int,
    min_dist: float,
    seed: int,
) -> Path | None:  # noqa: D401
    """UMAP 결과를 저장하고 UMAP이 없으면 ``None``을 반환한다."""

    if UMAP is None:
        return None
    reducer = UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=seed,
    )
    umap_coords = reducer.fit_transform(concat)
    path = output_dir / f"{stem}_umap.png"
    _scatter(umap_coords, labels, "Concat Embeddings (UMAP)", path)
    return path


def save_all(
    concat: np.ndarray,
    labels: np.ndarray,
    output_dir: Path,
    stem: str,
    tsne_perplexity: float,
    umap_n_neighbors: int,
    umap_min_dist: float,
    seed: int,
) -> VisualizationResultDict:
    """t-SNE/UMAP을 모두 실행하고 각 결과 경로를 묶어 반환한다."""

    tsne_path = save_tsne(
        concat=concat,
        labels=labels,
        output_dir=output_dir,
        stem=stem,
        perplexity=tsne_perplexity,
        seed=seed,
    )
    umap_path = save_umap(
        concat=concat,
        labels=labels,
        output_dir=output_dir,
        stem=stem,
        n_neighbors=umap_n_neighbors,
        min_dist=umap_min_dist,
        seed=seed,
    )
    return {"tsne_path": tsne_path, "umap_path": umap_path}


