## MVTC_Multimodal
## VTC(Vision&Tactile Cognition)
- The VTC project develops a multimodal generative model that takes **images** and **tactile modality vectors** as input to generate **tactile sensor signals**.

## Dataset
- This project experimentally uses the **PHAC2 dataset** for tactile modality experiments.  
- The dataset is applied for **research and validation purposes only**.  
- Please ensure that you have the appropriate usage rights and follow the dataset’s license and guidelines.


## Settings
### Build a docker image
```
docker container run --gpus all --name={container_name} --restart=always -it {image_name} /bin/bash
```

### Install packages with pipfile
```
./setup-dev.sh
pipenv shell
```

### PyTorch
```
pipenv run pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

### Embedding Model Requirements
- This project uses BERT for text embeddings and CLIP for image embeddings.
- Please download the models manually from Hugging Face before running the code
```
import os
from huggingface_hub import snapshot_download

local_dir = "/home/MVTC_Multimodal/models/"
model_LIST = ["google-bert/bert-base-multilingual-cased", "openai/clip-vit-large-patch14"]
for model_name in model_list :
    snapshot_download(
        repo_id=model_name,
        local_dir=os.path.join(local_dir, model_name.spilt("/")[-1]),
        local_dir_use_symlinks=False,
    )
```
