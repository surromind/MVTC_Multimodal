"""CLIP/BERT 임베딩과 촉각 잠재벡터를 결합하는 단순 융합 모듈."""

from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from .embedders import VTCEmbedder
from app.utils.path import ModelPathsDict
from .tactile import load_tactile_encoder


class ConcatenationFusion:
    """이미지·텍스트·촉각 임베딩을 단순 결합해 반환하는 베이스라인."""

    def __init__(
        self,
        model_paths: ModelPathsDict,
        training_cfg: Mapping[str, Any],
        device: str | None = None,
    ) -> None:
        """모델 체크포인트와 설정으로 융합기를 구성한다."""

        requested_device = device or training_cfg.get("device", "auto")
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)

        if model_paths["tactile_checkpoint"] is None:
            raise ValueError("tactile_checkpoint 경로가 설정되어야 합니다.")

        self.embedder = VTCEmbedder(
            clip_model_id=model_paths["clip_model"],
            bert_model_id=model_paths["bert_model"],
            device=str(self.device),
        )
        self.tactile = load_tactile_encoder(
            model_paths["tactile_checkpoint"],
            device=str(self.device),
            sequence_length=training_cfg.get("sequence_length"),
        )

    @torch.no_grad()
    def encode(
        self,
        image_path: str,
        text: str,
        tactile_sequence: torch.Tensor,
        l2norm_parts: bool = True,
    ) -> dict[str, np.ndarray]:
        """단일 샘플을 받아 참조·촉각·결합 벡터를 생성한다.

        Parameters
        ----------
        image_path, text, tactile_sequence
            각 모달리티의 원시 입력.
        l2norm_parts
            부분 임베딩을 정규화할지 여부.
        """

        # 1) 이미지/텍스트 임베딩 추출
        outputs = self.embedder.encode(image_path, text, l2norm=True)
        if outputs.get("fused") is None:
            raise RuntimeError("참조 임베딩을 생성하지 못했습니다.")
        ref_vec = torch.from_numpy(outputs["fused"]).float().to(self.device)
        ref_vec = ref_vec.squeeze(0)

        # 2) 촉각 시퀀스를 잠재벡터로 변환
        tactile_latent = self.tactile.encode(tactile_sequence, requires_grad=False)
        tactile_vec = tactile_latent.squeeze(0)

        # 3) 필요 시 각 벡터를 정규화 후 결합
        if l2norm_parts:
            ref_vec = F.normalize(ref_vec, dim=-1)
            tactile_vec = F.normalize(tactile_vec, dim=-1)

        concat_vec = torch.cat([ref_vec, tactile_vec], dim=-1)

        return {
            "reference": ref_vec.detach().cpu().numpy(),
            "tactile": tactile_vec.detach().cpu().numpy(),
            "concatenated": concat_vec.detach().cpu().numpy(),
        }
