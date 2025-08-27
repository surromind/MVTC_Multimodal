from typing import Literal

import torch
from transformers import BertModel, BertTokenizer

from app.core.embedder.base_embedder import BaseEmbedder


class BERT_Embedder(BaseEmbedder):
    def __init__(self, model_path):
        super().__init__(model_path)  # model_id, device 설정
        self.model = BertModel.from_pretrained(self.model_id).to(self.device).eval()
        self.tokenizer = BertTokenizer.from_pretrained(self.model_id)

    def get_img_embedding(self, image_path: str) -> None:
        ValueError("BERT_Embedder: get_img_embedding is not supported")
        return False

    def get_text_embedding(self, text: str, return_type: Literal["cls", "mean"] = "cls") -> torch.Tensor:
        """
        입력 텍스트에 대한 BERT 임베딩을 추출하는 함수.

        Args:
            text (str): 임베딩을 추출할 입력 문장.
            return_type (Literal["cls", "mean"], optional):
                임베딩 추출 방식 선택.
                - "cls": [CLS] 토큰의 출력 벡터(pooler_output)를 반환.
                - "mean": 전체 토큰 hidden state의 평균 벡터를 반환.
                기본값은 "cls".

        Returns:
            torch.Tensor: 지정된 방식으로 추출된 텍스트 임베딩 텐서.
                - shape: [1, hidden_dim]
        """
        input = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**input)
            cls_embeddings = outputs.pooler_output  # [CLS] 임베딩
            mean_embeddings = outputs.last_hidden_state.mean(dim=1)  # 토큰 평균 임베딩

        if return_type == "cls":
            return cls_embeddings
        elif return_type == "mean":
            return mean_embeddings
