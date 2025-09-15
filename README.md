## MVTC_Multimodal
## VTC(Vision&Tactile Cognition)
- The VTC project develops a multimodal generative model that takes **images** and **tactile modality vectors** as input to generate **tactile sensor signals**.

## Settings
### Build
```
./scripts/build.sh
./scripts/run.sh
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
