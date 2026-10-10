# #pip install kaldi_native_fbank
# from pathlib import Path
# from modelscope import snapshot_download
# model_dir = snapshot_download('dengcunqin/SenseVoiceSmall_hotword')
# import sys
# sys.path.append(model_dir)

# from sensevoice_bin_hot import SenseVoiceSmall
# from funasr_onnx.utils.postprocess_utils import rich_transcription_postprocess

# model = SenseVoiceSmall(model_dir, batch_size=10, quantize=False)

# wav_or_scp = [model_dir+"/asr_example.wav".format(Path.home(), model_dir)]

# res = model(wav_or_scp,hotwords_str='',hotwords_score=1.0)
# print('SenseVoiceSmall output:',[rich_transcription_postprocess(i) for i in res])

# res = model(wav_or_scp,hotwords_str='打磨院',hotwords_score=1.0)
# print('SenseVoiceSmall_hotword output:',[rich_transcription_postprocess(i) for i in res])


# wav_or_scp = [model_dir+"/A2_0.wav".format(Path.home(), model_dir)]

# res = model(wav_or_scp,hotwords_str='',hotwords_score=1.0)
# print('SenseVoiceSmall output:',[rich_transcription_postprocess(i) for i in res])

# res = model(wav_or_scp,hotwords_str='秀妹',hotwords_score=1.0)
# print('SenseVoiceSmall_hotword output:',[rich_transcription_postprocess(i) for i in res])

import sounddevice as sd
import numpy as np
import time
import torch
import sys
from pathlib import Path
from modelscope import snapshot_download

from silero_vad import load_silero_vad, VADIterator
from funasr.utils.postprocess_utils import rich_transcription_postprocess


print(f'cuda is available: {torch.cuda.is_available()}')

CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
PREV_CHUNKS = int(0.2 * SAMPLE_RATE / CHUNK_SIZE) 

# 下载并加载 SenseVoiceSmall 热词版
model_dir = snapshot_download('dengcunqin/SenseVoiceSmall_hotword')
sys.path.append(model_dir)

from sensevoice_bin_hot import SenseVoiceSmall
model = SenseVoiceSmall(model_dir, batch_size=10, quantize=False)

# 热词配置
HOTWORDS = "打疫苗|依依二二|秀妹|安安|华智未来|大摩院"
HOTWORDS_SCORE = 1.0

# 查看当前录音设备
print(sd.query_devices())

def test_sensevoice():
    vad = load_silero_vad()
    log = open("log.txt", "w", encoding="utf-8")
    vad_iter = VADIterator(vad, threshold=0.3)

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32") as stream:
        audio_chunks = []
        prev_audio_chunks = []
        speaking = False

        while True:
            samples, _ = stream.read(CHUNK_SIZE)
            samples = samples[:, 0]  # 单声道
            speech_dict = vad_iter(samples)

            if speech_dict:
                if not speaking and "start" in speech_dict:
                    speaking = True

            if speaking:
                audio_chunks.append(samples)
            else:
                prev_audio_chunks.append(samples)
                if len(prev_audio_chunks) > PREV_CHUNKS:
                    prev_audio_chunks.pop(0)

            if speech_dict and "end" in speech_dict:
                speaking = False
                audio = np.concatenate(prev_audio_chunks + audio_chunks)
                audio_chunks = []

                st = time.time()
                res = model(audio * 32768, hotwords_str=HOTWORDS, hotwords_score=HOTWORDS_SCORE)
                text = [rich_transcription_postprocess(i) for i in res]
                print(f"sensevoice ret={text} timeusage={int((time.time()-st)*1000)}ms")
                log.write(f"sensevoice ret={text} timeusage={int((time.time()-st)*1000)}ms\n")
                log.flush()

if __name__ == "__main__":
    test_sensevoice()
