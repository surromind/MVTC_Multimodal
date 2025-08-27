from re import M

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.core.embedder.base_embedder import BaseEmbedder
from app.utils.logger import logger


class CLIP_Embedder(BaseEmbedder):
    def __init__(self, model_path):
        logger.info("Initializing CLIP_Embedder")
        super().__init__(model_path)  # model_id, device 설정
        self.model = CLIPModel.from_pretrained(self.model_id).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(self.model_id)

    def get_img_embedding(self, image_path: str) -> torch.Tensor:
        logger.info(f"CLIP : Getting image embedding from {image_path}")
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=img, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            z_img = self.model.get_image_features(**{k: inputs[k] for k in ["pixel_values"]})
            z_img = z_img / z_img.norm(dim=-1, keepdim=True)
            return z_img

    def get_text_embedding(self, text: str) -> torch.Tensor:
        logger.info(f"CLIP : Getting text embedding from {text}")
        inputs = self.processor(text=text, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            z_txt = self.model.get_text_features(**{k: inputs[k] for k in ["input_ids", "attention_mask"]})
            z_txt = z_txt / z_txt.norm(dim=-1, keepdim=True)
            return z_txt
