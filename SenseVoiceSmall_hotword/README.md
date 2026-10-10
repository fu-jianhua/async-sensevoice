---
frameworks:
- Pytorch
license: Apache License 2.0
tasks:
- auto-speech-recognition

#model-type:
##如 gpt、phi、llama、chatglm、baichuan 等
#- gpt

#domain:
##如 nlp、cv、audio、multi-modal
#- nlp

#language:
##语言代码列表 https://help.aliyun.com/document_detail/215387.html?spm=a2c4g.11186623.0.0.9f8d7467kni6Aa
#- cn 

#metrics:
##如 CIDEr、Blue、ROUGE 等
#- CIDEr

#tags:
##各种自定义，包括 pretrained、fine-tuned、instruction-tuned、RL-tuned 等训练方法和其他
#- pretrained

#tools:
##如 vllm、fastchat、llamacpp、AdaSeq 等
#- vllm
---


# Highlights
在官方SenseVoiceSmall基础上，加装热词模块训练，并导出onnx模型。

训练数据为1w小时普通话，200小时粤语，1w小时英文，因达到训练目标提前停止（示例音频达摩院热词成功转换为打磨院）,大约训练了0.3轮（单卡4090训练约24小时）。

因训练不足，可能出现某些热词效果不好的情况。

如果想要在官方docker上运行，需要等官方加装热词模块。
## 模型说明
model.onnx为sensevoice_small主体，即未ctc_lo前的encode_out。

sensevoice_model_hot_emb.onnx为热词编码模块，如果热词相同，可缓存避免多次运行浪费资源。

sensevoice_model_hot_module.onnx为热词增强模块。

sensevoice_model_nohot_module.onnx为原版ctc_lo，model.onnx输出的encode_out经过此模块后输出即为原版sensevoice_small。

```python
#pip install kaldi_native_fbank
from pathlib import Path
from modelscope import snapshot_download
model_dir = snapshot_download('dengcunqin/SenseVoiceSmall_hotword')
import sys
sys.path.append(model_dir)

from sensevoice_bin_hot import SenseVoiceSmall
from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess

model = SenseVoiceSmall(model_dir, batch_size=10, quantize=False)

wav_or_scp = [model_dir+"/asr_example.wav".format(Path.home(), model_dir)]

res = model(wav_or_scp,hotwords_str='',hotwords_score=1.0)
print('SenseVoiceSmall output:',[rich_transcription_postprocess(i) for i in res])

res = model(wav_or_scp,hotwords_str='打磨院',hotwords_score=1.0)
print('SenseVoiceSmall_hotword output:',[rich_transcription_postprocess(i) for i in res])


wav_or_scp = [model_dir+"/A2_0.wav".format(Path.home(), model_dir)]

res = model(wav_or_scp,hotwords_str='',hotwords_score=1.0)
print('SenseVoiceSmall output:',[rich_transcription_postprocess(i) for i in res])

res = model(wav_or_scp,hotwords_str='秀妹',hotwords_score=1.0)
print('SenseVoiceSmall_hotword output:',[rich_transcription_postprocess(i) for i in res])
```
### 微信交流
Funasr群找小邓

### 微信小程序
<img src="https://www.modelscope.cn/models/dengcunqin/speech_seaco_paraformer_large_asr_nat-zh-cantonese-en-16k-common-vocab11666-pytorch/resolve/master/%E5%BE%AE%E4%BF%A1%E5%B0%8F%E7%A8%8B%E5%BA%8F%E7%A0%81.jpg" alt="微信小程序"  width="500" />

