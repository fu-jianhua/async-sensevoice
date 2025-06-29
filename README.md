# async-sensevoice

ASR Server for SenseVoice

## Installation

install torch with specific cuda version

```
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

clone SenseVoice and install SenseVoice requirements and this repo requirements

```
pip install -r SenseVoice/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
pip install -r ./requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```