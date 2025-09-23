"""설정 기반으로 멀티모달 데이터셋을 구축하는 헬퍼."""

import torch

from app.core import (
    FeatureScaler,
    MultimodalDataset,
    ProjectConfigDict,
    TactileNormalizer,
)


def build_dataset(cfg: ProjectConfigDict, device: torch.device) -> MultimodalDataset:
    """프로젝트 설정과 디바이스를 받아 :class:`MultimodalDataset`을 생성한다."""

    apply_scaler = bool(cfg["training"].get("apply_scaler", True))
    scaler: FeatureScaler | None = None
    if apply_scaler:
        # 1) 촉각 CSV에 적용할 스케일러 학습
        mode = cfg["training"].get("scaler_mode", "zscore")
        scaler = FeatureScaler(mode=mode)
        scaler.fit_directory(cfg["paths"].tactile_dir)

    # 2) 정규화기를 구성해 CSV → 텐서 변환 시 사용
    normalizer = TactileNormalizer(scaler=scaler, device=device)
    sequence_length = cfg["training"].get("sequence_length")

    # 3) 멀티모달 데이터셋 인스턴스 반환
    return MultimodalDataset(
        paths=cfg["paths"],
        csv_normalizer=normalizer,
        sequence_length=sequence_length,
    )
