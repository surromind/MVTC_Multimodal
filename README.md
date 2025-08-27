## MVTC_Multimodal
## VTC(Vision&Tactile Cognition)
- The VTC project develops a multimodal generative model that takes **images** and **tactile modality vectors** as input to generate **tactile sensor signals**.

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
