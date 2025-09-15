from typing import Literal

import numpy as np

from app.core.embedder.bert_embedder import BERT_Embedder
from app.core.embedder.clip_embedder import CLIP_Embedder
from config import logger


class VTC_Embedder:
    def __init__(self, clip_model_path: str, bert_model_path: str):
        logger.info(
            f"Initializing VTC_Embedder with clip_model_path: {clip_model_path} and bert_model_path: {bert_model_path}"
        )
        try:
            self.clip_embedder = CLIP_Embedder(clip_model_path)
            self.bert_embedder = BERT_Embedder(bert_model_path)
        except Exception as e:
            logger.error(f"Error initializing VTC_Embedder: {str(e)}")
            raise ValueError(f"Error initializing VTC_Embedder: {str(e)}")

    def concat_img_txt_embedding(
        self,
        image_embedding: np.ndarray,
        text_embedding: np.ndarray,
        adj_embedding: Literal[np.ndarray, None] = None,
        l2norm: bool = True,
    ) -> np.ndarray:

        if adj_embedding is not None:
            logger.info("Concatenating image, text and adj embeddings")
            concat_embedding = np.concatenate(
                [
                    image_embedding.detach().cpu().numpy(),
                    text_embedding.detach().cpu().numpy(),
                    adj_embedding.detach().cpu().numpy(),
                ],
                axis=0,
            )
        else:
            logger.info("Concatenating image and text embeddings")
            concat_embedding = np.concatenate(
                [
                    image_embedding.detach().cpu().numpy(),
                    text_embedding.detach().cpu().numpy(),
                ],
                axis=0,
            )

        if l2norm:
            concat_embedding = self.run_l2_normalize(concat_embedding)
        return concat_embedding

    def run_l2_normalize(self, x: np.ndarray) -> np.ndarray:
        return x / (np.linalg.norm(x) + 1e-12)

    def get_vtc_embedding(
        self,
        image_path: str,
        text: str,
    ) -> np.ndarray:
        logger.info("Getting VTC embedding")
        try:
            image_embedding = self.clip_embedder.get_img_embedding(image_path)
            text_embedding = self.bert_embedder.get_text_embedding(text)
            return self.concat_img_txt_embedding(image_embedding, text_embedding)
        except Exception as e:
            logger.error(f"Error getting VTC embedding from {image_path} {e}")
            return None
