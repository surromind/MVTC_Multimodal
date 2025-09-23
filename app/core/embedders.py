"""이미지·텍스트 모달리티 임베딩 래퍼 모음."""

from typing import Sequence, TypedDict

import numpy as np
import torch
from PIL import Image
from transformers import BertModel, BertTokenizer, CLIPModel, CLIPProcessor


class EmbedOutputsDict(TypedDict, total=False):
    """각 모달리티 임베딩 결과를 담는 딕셔너리."""

    image: torch.Tensor | None
    text: torch.Tensor | None
    fused: np.ndarray | None


class BaseEmbedder:
    """Hugging Face 모델의 디바이스 선택을 공통 처리하는 베이스 클래스."""

    def __init__(self, model_id: str, device: str | None = None) -> None:
        """모델 ID와 디바이스 문자열을 받아 torch.device를 결정한다."""

        selected = device or "auto"
        if selected == "auto":
            selected = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model_id
        self.device = torch.device(selected)


class CLIPEmbedder(BaseEmbedder):
    def __init__(self, model_id: str, device: str | None = None) -> None:
        """CLIP 모델과 프로세서를 로드해 이미지/텍스트 임베딩을 제공."""

        super().__init__(model_id, device)
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(self.model_id)

    @torch.no_grad()
    def encode_image(self, image_path: str | Image.Image) -> torch.Tensor:
        """이미지를 RGB로 정규화해 CLIP 이미지 임베딩을 반환한다."""

        if isinstance(image_path, Image.Image):
            image = image_path.convert("RGB")
        else:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        return features / features.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_text(self, text: str | Sequence[str]) -> torch.Tensor:
        """텍스트 입력을 받아 CLIP 텍스트 임베딩을 반환한다."""

        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(
            self.device
        )
        features = self.model.get_text_features(
            **{k: inputs[k] for k in ["input_ids", "attention_mask"]}
        )
        return features / features.norm(dim=-1, keepdim=True)


class BERTEmbedder(BaseEmbedder):
    def __init__(self, model_id: str, device: str | None = None) -> None:
        """사전학습된 BERT를 로드해 텍스트 임베딩을 생성한다."""

        super().__init__(model_id, device)
        self.model = BertModel.from_pretrained(self.model_id).to(self.device).eval()
        self.tokenizer = BertTokenizer.from_pretrained(self.model_id)

    @torch.no_grad()
    def encode_text(self, text: str, mode: str = "cls") -> torch.Tensor:
        """텍스트를 토크나이징한 뒤 CLS 혹은 평균 풀링 임베딩을 반환한다."""

        inputs = self.tokenizer(
            text, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        outputs = self.model(**inputs)
        if mode == "cls":
            feats = outputs.pooler_output
        else:
            feats = outputs.last_hidden_state.mean(dim=1)
        return feats / feats.norm(dim=-1, keepdim=True)


class VTCEmbedder:
    """Vision-Text-Tactile 융합에 사용되는 이미지·텍스트 임베더 묶음."""

    def __init__(
        self,
        clip_model_id: str,
        bert_model_id: str,
        device: str | None = None,
    ) -> None:
        """CLIP과 BERT 임베더를 초기화한다."""

        self.clip = CLIPEmbedder(clip_model_id, device)
        self.bert = BERTEmbedder(bert_model_id, device)

    @torch.no_grad()
    def encode(self, image_path: str, text: str, l2norm: bool = True) -> EmbedOutputsDict:
        """이미지와 텍스트 임베딩을 각각 구한 뒤 결합한다."""

        # 1) 각 모달리티 임베딩 추출
        img_emb = self.clip.encode_image(image_path)
        txt_emb = self.bert.encode_text(text)

        # 2) numpy 배열로 결합하고 필요 시 L2 정규화
        fused = torch.cat([img_emb, txt_emb], dim=-1).detach().cpu().numpy()
        if l2norm:
            fused = fused / (np.linalg.norm(fused) + 1e-12)
        return {"image": img_emb, "text": txt_emb, "fused": fused}
