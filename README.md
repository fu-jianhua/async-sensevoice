# Async SenseVoice 音频服务

基于 SenseVoice 的本地音频捕获和语音识别服务，使用 silero VAD 进行语音活动检测。

## 功能特性

- 🎤 **本地音频捕获**: 使用 sounddevice 直接捕获系统音频
- 🗣️ **实时 VAD**: 使用 silero VAD 进行语音活动检测
- 📝 **高精度 ASR**: 使用 SenseVoice 模型进行语音识别
- 🔌 **WebSocket 服务**: 实时向前端推送识别结果
- ⚡ **低延迟**: 本地处理，无网络传输延迟

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 启动音频服务

```bash
# 方式1: 使用启动脚本（推荐）
python start_service.py

# 方式2: 直接运行服务
python audio_service.py
```

### 2. 前端集成

在 HTML 页面中引入音频处理模块：

```html
<script src="audio-handler.js"></script>
```

### 3. 服务配置

- **WebSocket 端口**: 8765
- **音频采样率**: 16000Hz
- **音频通道**: 1 (单声道)
- **VAD 阈值**: 0.3

## 文件结构

```
async-sensevoice/
├── audio_service.py      # 主服务文件
├── start_service.py      # 启动脚本
├── test_sensevoice.py    # 测试文件
├── requirements.txt      # 依赖列表
└── README.md            # 说明文档
```

## API 接口

### WebSocket 消息格式

#### 客户端发送消息

```json
// 开始音频捕获
{"type": "start_capture"}

// 停止音频捕获
{"type": "stop_capture"}

// 心跳包
{"type": "ping"}
```

#### 服务器发送消息

```json
// ASR 识别结果
{
    "type": "asr_result",
    "text": "识别的文本内容",
    "processing_time": 150
}

// VAD 状态
{
    "type": "vad_status",
    "status": "speech_start"
}

// 服务状态
{
    "type": "capture_started"
}
```

## 性能优化

- 使用多线程处理 ASR，避免阻塞音频回调
- 音频数据缓冲管理，减少内存占用
- WebSocket 连接池管理，支持多客户端

## 故障排除

### 1. 音频设备问题

```bash
# 检查音频设备
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### 2. 依赖问题

```bash
# 重新安装依赖
pip install --upgrade -r requirements.txt
```

### 3. 端口占用

```bash
# 检查端口占用
netstat -an | grep 8765
```

## 注意事项

- 确保系统有可用的音频输入设备
- 首次运行会自动下载模型文件
- 建议在安静的环境中使用以获得最佳识别效果