import os
from huggingface_hub import snapshot_download

from config.path import MODEL_DIR
from config import logger


def download_models(model_name: str):
    logger.info(f"Downloading model: {model_name}")
    snapshot_download(
        repo_id=model_name,
        local_dir=os.path.join(MODEL_DIR, model_name),
        local_dir_use_symlinks=False,
    )
