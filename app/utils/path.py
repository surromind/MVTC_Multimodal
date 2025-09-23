"""모듈 검색 경로와 설정 경로를 관리하는 유틸리티."""

import sys
from pathlib import Path
from typing import Iterable, Mapping, TypedDict


def _ensure_path(path: Path) -> None:
    """경로가 존재하면 ``sys.path`` 맨 앞에 추가한다."""

    if not path.exists():
        return
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


def configure_paths(extra_paths: Iterable[Path] | None = None) -> None:
    """프로젝트 루트 및 서브 디렉터리를 ``sys.path``에 등록한다."""

    project_root = Path(__file__).resolve().parents[2]
    default_paths = [
        project_root,
        project_root / "app",
        project_root / "app/core",
        project_root / "app/utils",
    ]
    for default_path in default_paths:
        _ensure_path(default_path)

    if extra_paths is not None:
        for path in extra_paths:
            _ensure_path(path)


class DatasetPaths:
    """데이터셋 관련 경로를 속성으로 보관하는 단순 래퍼."""

    def __init__(
        self,
        *,
        base_dir: Path,
        image_dir: Path,
        text_dir: Path,
        tactile_dir: Path,
        outputs_dir: Path,
        tactile_scaler: Path | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.image_dir = image_dir
        self.text_dir = text_dir
        self.tactile_dir = tactile_dir
        self.outputs_dir = outputs_dir
        self.tactile_scaler = tactile_scaler

    def ensure(self) -> None:
        """출력 디렉터리가 존재하지 않으면 생성한다."""
        self.outputs_dir.mkdir(parents=True, exist_ok=True)


def make_dataset_paths(cfg: Mapping[str, object]) -> "DatasetPaths":
    """설정 딕셔너리에서 :class:`DatasetPaths` 인스턴스를 생성한다."""
    base_dir = Path(cfg.get("base_dir", "") or "").resolve()
    return DatasetPaths(
        base_dir=base_dir,
        image_dir=(base_dir / (cfg.get("image_dir", "") or "")).resolve(),
        text_dir=(base_dir / (cfg.get("text_dir", "") or "")).resolve(),
        tactile_dir=(base_dir / (cfg.get("tactile_dir", "") or "")).resolve(),
        outputs_dir=Path(cfg.get("outputs_dir", "./outputs") or "./outputs").resolve(),
        tactile_scaler=(
            Path(cfg["tactile_scaler"]).resolve() if cfg.get("tactile_scaler") else None
        ),
    )


class ModelPathsDict(TypedDict):
    """모델 문자열과 체크포인트 경로를 묶어 보관하는 딕셔너리."""

    clip_model: str
    bert_model: str
    tactile_checkpoint: Path | None


def make_model_paths(cfg: Mapping[str, object]) -> "ModelPathsDict":
    """설정 딕셔너리에서 :class:`ModelPathsDict`를 생성한다."""
    checkpoint = cfg.get("tactile_checkpoint")
    tactile_ckpt = Path(checkpoint).resolve() if checkpoint else None
    return ModelPathsDict(
        clip_model=str(cfg.get("clip_model", "openai/clip-vit-base-patch32")),
        bert_model=str(cfg.get("bert_model", "bert-base-uncased")),
        tactile_checkpoint=tactile_ckpt,
    )

