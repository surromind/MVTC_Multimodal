"""프로젝트 설정 로더."""

from pathlib import Path
from typing import Any, Mapping, TypedDict

import yaml

from app.utils.path import (
    DatasetPaths,
    ModelPathsDict,
    make_dataset_paths,
    make_model_paths,
)


class ProjectConfigDict(TypedDict):
    """프로젝트 설정 필드를 담는 딕셔너리."""

    paths: DatasetPaths
    models: ModelPathsDict
    training: dict[str, Any]


def load_project_config(path: Path | str) -> ProjectConfigDict:
    """YAML 설정 파일을 읽어 :class:`ProjectConfigDict`를 생성한다."""

    cfg_path = Path(path)
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, Mapping):
        raise TypeError("설정 파일은 최상위에 매핑 구조를 가져야 합니다.")

    # 1) 서브 섹션을 dict 형태로 안전하게 추출하고 기본값 적용
    paths = make_dataset_paths(raw.get("paths", {}))
    models = make_model_paths(raw.get("models", {}))
    training = dict(raw.get("training", {}))

    # 2) 출력 경로 생성 보장
    paths.ensure()

    # 3) 프로젝트 설정 객체로 래핑
    return ProjectConfigDict(paths=paths, models=models, training=training)
