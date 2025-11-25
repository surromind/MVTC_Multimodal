"""CLIP/BERT 참조 임베딩과 촉각 잠재벡터를 단순 결합한 베이스라인 생성 스크립트."""

import argparse
from pathlib import Path

from app.utils import build_dataset
from app.utils.embeddings import (
    build_metadata,
    collect_embeddings,
    save_embeddings,
    visualize_embeddings,
    write_metadata,
)

from app.core import ConcatenationFusion, load_project_config


def build_parser() -> argparse.ArgumentParser:
    """명령행 인자를 정의하는 ArgumentParser를 생성한다."""

    parser = argparse.ArgumentParser(description="Generate concat baseline embeddings")
    parser.add_argument(
        "--config",
        type=str,
        default="app/config/default.yaml",
        help="설정 파일 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/concat_embeddings.npz",
        help="저장할 npz 파일 경로",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="outputs/concat_embeddings_meta.json",
        help="샘플 메타데이터를 저장할 JSON 경로",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="사용할 디바이스(cuda 또는 cpu). 지정하지 않으면 config/device를 따릅니다.",
    )
    parser.add_argument(
        "--figure-dir",
        type=str,
        default=None,
        help="시각화 결과를 저장할 디렉터리 (기본값은 output과 동일한 폴더)",
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=float,
        default=20.0,
        help="t-SNE perplexity 값",
    )
    parser.add_argument(
        "--umap-n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors 값",
    )
    parser.add_argument(
        "--umap-min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist 값",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="차원 축소용 랜덤 시드",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="설정 시 t-SNE/UMAP 시각화를 생략합니다.",
    )
    return parser


def main() -> None:
    """설정을 로드하고 임베딩 생성 → 저장 → 시각화 순으로 실행한다."""

    # 1) 사용자 입력 파싱
    parser = build_parser()
    args = parser.parse_args()

    # 2) 프로젝트 설정 및 모델 준비
    cfg = load_project_config(Path(args.config))

    fusion = ConcatenationFusion(
        model_paths=cfg["models"],
        training_cfg=cfg["training"],
        device=args.device,
    )

    # 3) 데이터셋 구성 후 임베딩 수집
    dataset = build_dataset(cfg, fusion.device)

    batch = collect_embeddings(fusion, dataset)

    # 4) 임베딩 및 메타데이터 저장
    output_path = Path(args.output)
    save_embeddings(batch, output_path)

    metadata = build_metadata(batch, Path(args.config))
    metadata_path = Path(args.metadata)
    write_metadata(metadata, metadata_path)

    print(f"저장 완료: {output_path} ({metadata['num_samples']} samples)")
    print(f"메타데이터: {metadata_path}")

    if args.no_visualization:
        return

    # 5) 군집화 확인을 위한 시각화 수행
    figure_dir = Path(args.figure_dir) if args.figure_dir else output_path.parent
    visualize_embeddings(
        batch,
        figure_dir,
        stem=output_path.stem,
        tsne_perplexity=args.tsne_perplexity,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
