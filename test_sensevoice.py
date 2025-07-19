'''
silero VAD流式断句
sensevoice 非流式ASR
'''
import sounddevice as sd
from funasr import AutoModel
from silero_vad import load_silero_vad,VADIterator
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import torch 
import numpy as np
import time
print(f'cuda is available: {torch.cuda.is_available()}')

CHANNELS=1
SAMPLE_RATE=16000
CHUNK_SIZE=512
PREV_CHUNKS=int(0.2*SAMPLE_RATE/CHUNK_SIZE) 

# 查看当前录音设备
print(sd.query_devices())

def test_sensevoice():
    model=AutoModel(model='iic/SenseVoiceSmall',trust_remote_code=True,remote_code="./model.py",device="cuda:0")
    vad=load_silero_vad()
    
    log=open("log.txt","w",encoding='utf-8')
    vad_iter=VADIterator(vad,threshold=0.3)
    with sd.InputStream(samplerate=SAMPLE_RATE,channels=CHANNELS,dtype='float32') as stream:
        audio_chunks=[]
        prev_audio_chunks=[]
        
        speaking=False
        while True:
            samples,_=stream.read(CHUNK_SIZE)
            samples=samples[:,0]
            speech_dict=vad_iter(samples)
            if speech_dict:
                if not speaking and 'start' in speech_dict:
                    speaking=True
            if speaking:
                audio_chunks.append(samples)
            else:
                prev_audio_chunks.append(samples)
                if len(prev_audio_chunks)>PREV_CHUNKS:
                    prev_audio_chunks.pop(0)
            if speech_dict and 'end' in speech_dict:
                speaking=False
                audio=np.concatenate(prev_audio_chunks+audio_chunks)
                audio_chunks=[]
                st=time.time()
                result=model.inference(audio*32768,language='zh',disable_pbar=True)
                text=rich_transcription_postprocess(result[0]['text'])
                print(f'sensevoice ret={text} timeusage={int((time.time()-st)*1000)}ms')
                log.write(f'sensevoice ret={text} timeusage={int((time.time()-st)*1000)}ms\n')
                log.flush()

if __name__=='__main__':
    test_sensevoice()