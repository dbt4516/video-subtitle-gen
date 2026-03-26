# video-subtitle-gen

Local video-to-subtitle tool powered by [whisper.cpp](https://github.com/ggml-org/whisper.cpp). Extracts audio from video files and generates SRT subtitles, entirely offline.

## Features

- Automatic language detection (99 languages supported)
- VAD (Voice Activity Detection) to reduce hallucinations in silent segments
- Subtitle merging for more natural, longer subtitle lines
- Runs fully local on macOS (Apple Silicon optimized)

## Prerequisites

```bash
brew install ffmpeg whisper-cpp
```

Download a whisper model (place in `~/Downloads/whisper-models/` or current directory):

```bash
# Recommended: good balance of speed and accuracy
curl -L -o ~/Downloads/whisper-models/ggml-large-v3-turbo.bin \
  https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin
```

(Optional) Download VAD model to reduce hallucinations:

```bash
curl -L -o ~/Downloads/whisper-models/ggml-silero-v5.1.2.bin \
  https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin
```

## Usage

```bash
# Basic usage (auto-detect language, VAD enabled)
python3 transcribe.py video.mp4

# Specify language
python3 transcribe.py video.mp4 --lang zh

# Use a different model
python3 transcribe.py video.mp4 -m ggml-medium.bin

# Disable VAD
python3 transcribe.py video.mp4 --no-vad

# Custom subtitle merging
python3 transcribe.py video.mp4 --merge-gap 3.0 --merge-max 45

# No merging (keep original short segments)
python3 transcribe.py video.mp4 --no-merge

# Output to a different directory
python3 transcribe.py video.mp4 -o ./subtitles/
```

## Available Models

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| `ggml-tiny.bin` | 75 MB | Fastest | Low |
| `ggml-base.bin` | 142 MB | Fast | Fair |
| `ggml-small.bin` | 466 MB | Medium | Good |
| `ggml-medium.bin` | 1.5 GB | Slow | Great |
| `ggml-large-v3-turbo.bin` | 1.6 GB | Medium | Great |
| `ggml-large-v3.bin` | 3.1 GB | Slowest | Best |

Models with `.en` suffix (e.g. `ggml-medium.en.bin`) are English-only but more accurate for English.

## License

MIT
