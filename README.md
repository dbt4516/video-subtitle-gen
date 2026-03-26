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
# Single file
python3 transcribe.py video.mp4

# Batch: process all .mp4 and .ts files in a directory
python3 transcribe.py /path/to/videos/

# Batch with skip (don't re-process files that already have .srt)
python3 transcribe.py /path/to/videos/ --skip-existing

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
python3 transcribe.py /path/to/videos/ -o ./subtitles/
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


## Subtitle Translation

Translate SRT files between 200+ languages using [NLLB-200](https://huggingface.co/facebook/nllb-200-distilled-600M), fully offline.

```bash
# Setup (one-time)
python3 -m venv .venv && source .venv/bin/activate
pip install transformers sentencepiece torch

# Translate Japanese to Chinese
python3 translate.py subtitle.srt --from ja --to zh

# Batch translate a directory
python3 translate.py /path/to/srts/ --from en --to zh

# Skip already translated files by checking for .zh.srt
python3 translate.py /path/to/srts/ --from ja --to zh
```

Supported languages: zh, ja, en, ko, fr, de, es, ru, pt, ar, th, vi, id (and 200+ more via NLLB code).

## License

MIT
