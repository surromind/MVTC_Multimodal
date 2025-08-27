from abc import ABC, abstractmethod

import torch


class BaseEmbedder(ABC):
    def __init__(self, model_path: str):
        self.model_id = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @abstractmethod
    def get_img_embedding(self, img_path: str) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_text_embedding(self, text: str) -> torch.Tensor:
        raise NotImplementedError
