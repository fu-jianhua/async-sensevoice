'''
本地系统音频捕获 + WebSocket 服务
使用 silero VAD 流式断句 + sensevoice 非流式ASR
'''
import asyncio
import websockets
import json
import sounddevice as sd
from funasr import AutoModel
from silero_vad import load_silero_vad, VADIterator
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import torch 
import numpy as np
import time
import threading
from queue import Queue

print(f'cuda is available: {torch.cuda.is_available()}')

CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK_SIZE = 512
PREV_CHUNKS = int(0.2 * SAMPLE_RATE / CHUNK_SIZE)

# VAD 优化参数
VAD_THRESHOLD = 0.3  # 降低阈值，减少误判
SPEECH_END_DELAY = 0.3  # 语音结束延迟0.3秒，减少短句延迟
MIN_SPEECH_DURATION = 0.2  # 最小语音持续时间0.2秒，支持更短的语音

class LocalAudioService:
    def __init__(self):
        # 初始化模型
        self.model = AutoModel(model='iic/SenseVoiceSmall', trust_remote_code=True, 
                              remote_code="./model.py", device="cuda:0")
        self.vad = load_silero_vad()
        self.vad_iter = VADIterator(self.vad, threshold=VAD_THRESHOLD)
        
        # WebSocket 客户端管理
        self.websocket_clients = set()
        self.client_lock = asyncio.Lock()
        
        # 音频处理状态
        self.audio_chunks = []
        self.prev_audio_chunks = []
        self.speaking = False
        self.is_capturing = False
        
        # VAD 优化状态
        self.speech_start_time = 0
        self.speech_end_timer = None
        
        # 防重复机制
        self.last_asr_text = ""
        self.last_asr_time = 0
        self.asr_cooldown = 0.3  # 0.3秒冷却时间，减少连续短句的延迟
        
        # 消息队列 - 用于从非异步上下文发送消息到异步上下文
        self.message_queue = Queue()
        
        # 日志文件
        self.log = open("audio_service_log.txt", "w", encoding='utf-8')
        
        print("音频服务初始化完成")
    
    async def add_client(self, websocket):
        """添加 WebSocket 客户端"""
        async with self.client_lock:
            self.websocket_clients.add(websocket)
            print(f"客户端连接，当前连接数: {len(self.websocket_clients)}")
    
    async def remove_client(self, websocket):
        """移除 WebSocket 客户端"""
        async with self.client_lock:
            self.websocket_clients.discard(websocket)
            print(f"客户端断开，当前连接数: {len(self.websocket_clients)}")
    
    async def broadcast_to_clients(self, message):
        """广播消息给所有客户端"""
        if not self.websocket_clients:
            return
        
        async with self.client_lock:
            disconnected_clients = set()
            for client in self.websocket_clients:
                try:
                    await client.send(json.dumps(message))
                except websockets.exceptions.ConnectionClosed:
                    disconnected_clients.add(client)
                except Exception as e:
                    print(f"发送消息失败: {e}")
                    disconnected_clients.add(client)
            
            # 清理断开的连接
            for client in disconnected_clients:
                self.websocket_clients.discard(client)
    
    async def process_message_queue(self):
        """处理消息队列中的消息"""
        while True:
            try:
                # 非阻塞方式获取消息
                message = self.message_queue.get_nowait()
                await self.broadcast_to_clients(message)
            except:
                # 队列为空，等待一小段时间
                await asyncio.sleep(0.01)
    
    def process_asr(self, audio):
        """处理 ASR（在独立线程中运行）"""
        try:
            st = time.time()
            result = self.model.inference(audio * 32768, language='zh', disable_pbar=True)
            text = rich_transcription_postprocess(result[0]['text'])
            processing_time = int((time.time() - st) * 1000)
            
            # 防重复处理 - 对短句更友好
            current_time = time.time()
            if (text == self.last_asr_text and 
                current_time - self.last_asr_time < self.asr_cooldown):
                # 如果文本很短（可能是短句），减少冷却时间
                if len(text) <= 5:
                    short_cooldown = 0.1  # 短句0.1秒冷却
                    if current_time - self.last_asr_time < short_cooldown:
                        print(f'检测到重复短句ASR结果，忽略: {text}')
                        self.log.write(f'检测到重复短句ASR结果，忽略: {text}\n')
                        self.log.flush()
                        return
                else:
                    print(f'检测到重复ASR结果，忽略: {text}')
                    self.log.write(f'检测到重复ASR结果，忽略: {text}\n')
                    self.log.flush()
                    return
            
            print(f'sensevoice ret={text} timeusage={processing_time}ms')
            self.log.write(f'sensevoice ret={text} timeusage={processing_time}ms\n')
            self.log.flush()
            
            # 更新防重复状态
            self.last_asr_text = text
            self.last_asr_time = current_time
            
            # 将消息放入队列，由异步上下文处理
            self.message_queue.put({
                'type': 'asr_result',
                'text': text,
                'processing_time': processing_time
            })
            
        except Exception as e:
            print(f"ASR 处理失败: {e}")
            self.log.write(f"ASR 处理失败: {e}\n")
            self.log.flush()
    
    def start_audio_capture(self):
        """开始音频捕获"""
        if self.is_capturing:
            return
        
        self.is_capturing = True
        print("开始音频捕获...")
        
        try:
            # 使用非回调方式，直接读取音频数据
            with sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype='float32',
                blocksize=CHUNK_SIZE
            ) as stream:
                print("音频流已启动，按 Ctrl+C 停止")
                
                # 直接读取音频数据
                while self.is_capturing:
                    try:
                        samples, _ = stream.read(CHUNK_SIZE)
                        samples = samples[:, 0]  # 取第一个通道
                        
                        # VAD 检测
                        speech_dict = self.vad_iter(samples)
                        
                        # 处理语音开始
                        if speech_dict and 'start' in speech_dict:
                            if not self.speaking:
                                self.speaking = True
                                self.speech_start_time = time.time()
                                print("检测到语音开始")
                                # 将消息放入队列，由异步上下文处理
                                self.message_queue.put({
                                    'type': 'vad_status',
                                    'status': 'speech_start'
                                })
                            else:
                                # 如果正在说话时又检测到语音开始，取消之前的延迟处理
                                if self.speech_end_timer:
                                    self.speech_end_timer.cancel()
                                    self.speech_end_timer = None
                                    print("取消语音结束延迟处理，继续说话")
                        
                        # 处理语音结束
                        if speech_dict and 'end' in speech_dict:
                            if self.speaking:
                                # 检查最小语音持续时间
                                speech_duration = time.time() - self.speech_start_time
                                if speech_duration >= MIN_SPEECH_DURATION:
                                    # 根据语音长度动态调整延迟
                                    # 短语音使用更短的延迟，长语音使用标准延迟
                                    if speech_duration < 1.0:
                                        dynamic_delay = 0.2  # 短语音0.2秒延迟
                                    elif speech_duration < 3.0:
                                        dynamic_delay = 0.3  # 中等语音0.3秒延迟
                                    else:
                                        dynamic_delay = SPEECH_END_DELAY  # 长语音使用标准延迟
                                    
                                    # 设置延迟处理语音结束
                                    if self.speech_end_timer:
                                        self.speech_end_timer.cancel()
                                    
                                    self.speech_end_timer = threading.Timer(
                                        dynamic_delay, 
                                        self.process_speech_end
                                    )
                                    self.speech_end_timer.start()
                                    print(f"语音结束延迟处理，持续时间: {speech_duration:.2f}s，延迟: {dynamic_delay}s")
                                else:
                                    print(f"语音太短，忽略: {speech_duration:.2f}s")
                                    self.speaking = False
                                    self.audio_chunks = []
                        
                        # 收集音频数据
                        if self.speaking:
                            self.audio_chunks.append(samples)
                        else:
                            self.prev_audio_chunks.append(samples)
                            if len(self.prev_audio_chunks) > PREV_CHUNKS:
                                self.prev_audio_chunks.pop(0)
                                
                    except Exception as e:
                        print(f"音频读取错误: {e}")
                        break
                        
        except Exception as e:
            print(f"音频捕获失败: {e}")
            self.is_capturing = False
    
    def process_speech_end(self):
        """延迟处理语音结束"""
        if self.speaking and self.audio_chunks:
            self.speaking = False
            print("处理语音结束")
            
            # 合并音频进行 ASR
            audio = np.concatenate(self.prev_audio_chunks + self.audio_chunks)
            self.audio_chunks = []
            
            # 在新线程中处理 ASR，避免阻塞音频读取
            threading.Thread(target=self.process_asr, args=(audio,)).start()
    
    def stop_audio_capture(self):
        """停止音频捕获"""
        self.is_capturing = False
        print("音频捕获已停止")

async def websocket_handler(websocket):
    """WebSocket 连接处理器"""
    await audio_service.add_client(websocket)
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                print(f"收到客户端消息: {data}")
                
                # 处理客户端命令
                if data.get('type') == 'ping':
                    await websocket.send(json.dumps({'type': 'pong'}))
                elif data.get('type') == 'start_capture':
                    if not audio_service.is_capturing:
                        threading.Thread(target=audio_service.start_audio_capture).start()
                    await websocket.send(json.dumps({'type': 'capture_started'}))
                elif data.get('type') == 'stop_capture':
                    audio_service.stop_audio_capture()
                    await websocket.send(json.dumps({'type': 'capture_stopped'}))
                    
            except json.JSONDecodeError:
                print(f"无效的 JSON 消息: {message}")
            except Exception as e:
                print(f"处理消息失败: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        print("WebSocket 连接已关闭")
    except Exception as e:
        print(f"WebSocket 错误: {e}")
    finally:
        await audio_service.remove_client(websocket)

async def main():
    """主函数"""
    global audio_service
    audio_service = LocalAudioService()
    
    # 启动 WebSocket 服务器
    print("启动 WebSocket 服务器...")
    server = await websockets.serve(websocket_handler, "localhost", 8765)
    print("WebSocket 服务器已启动，监听端口 8765")
    print("等待客户端连接...")
    
    # 启动消息队列处理任务
    message_task = asyncio.create_task(audio_service.process_message_queue())
    
    # 保持服务器运行
    try:
        await server.wait_closed()
    finally:
        message_task.cancel()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"服务器错误: {e}")
