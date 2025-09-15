import argparse
import os

import torch

from config import logger
from core.vtc_embedder import VTC_Embedder
from utils.txt_utils import load_txt
from utils.npy_utils import save_npy
from utils.models import download_models
from config.path import TEST_DATA_DIR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clip_model_path",
        type=str,
        default="./models/openai/clip-vit-large-patch14",
    )
    parser.add_argument(
        "--bert_model_path",
        type=str,
        default="./models/google-bert/bert-base-multilingual-cased",
    )
    parser.add_argument("--vectors_dir", type=str, default="./vectors")
    args = parser.parse_args()

    # CUDA 사용 가능 여부 확인
    if torch.cuda.is_available():
        logger.info("CUDA is available")
    else:
        logger.info("CUDA is not available")

    for model_name in [
        "openai/clip-vit-large-patch14",
        "google-bert/bert-base-multilingual-cased",
    ]:
        if not os.path.exists(f"./models/{model_name}"):
            download_models(model_name)

    # TODO : 테스트 데이터 경로 수정
    logger.info("Starting MVTC Multimodal Embedding")
    vtc_embedder = VTC_Embedder(args.clip_model_path, args.bert_model_path)

    labels = load_txt("./test_data/labels.txt")
    vtc_embedding = vtc_embedder.get_vtc_embedding(
        image_path="./test_data/black_foam.jpg", text=" ".join(labels)
    )
    save_npy(vtc_embedding, "./test_data/vtc_embedding.npy")
