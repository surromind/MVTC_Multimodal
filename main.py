import argparse
import os

from natsort import natsorted

from app.core.vtc_embedder import VTC_Embedder
from app.utils.json_utils import load_json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_model_path", type=str, default="./models/clip-vit-large-patch14")
    parser.add_argument("--bert_model_path", type=str, default="./models/bert-base-multilingual-cased")
    parser.add_argument("--vectors_dir", type=str, default="./vectors")
    args = parser.parse_args()

    vtc_embedder = VTC_Embedder(args.clip_model_path, args.bert_model_path)

    image_dir = "./test_datasets/Filtered_PHAC2/Images"
    text_json_path = "./test_datasets/Filtered_PHAC2/object_adjective_text_dict.json"
    text_dict = load_json(text_json_path)

    for image_path in natsorted(os.listdir(image_dir)):
        object = "_".join(image_path.split("_")[:-2])
        text = text_dict[object]
        vtc_embedding = vtc_embedder.get_vtc_embedding(os.path.join(image_dir, image_path), text)
