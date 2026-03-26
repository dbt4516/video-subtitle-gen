#!/usr/bin/env python3
"""
使用 whisper.cpp (whisper-cli) 从视频中提取音频并生成 SRT 字幕文件。

依赖：
  - ffmpeg: brew install ffmpeg
  - whisper-cpp: brew install whisper-cpp
  - whisper 模型: 从 https://huggingface.co/ggerganov/whisper.cpp 下载
  - VAD 模型(可选): 从 https://huggingface.co/ggml-org/whisper-vad 下载

用法：
  python3 transcribe.py video.mp4
  python3 transcribe.py video.mp4 -m ~/models/ggml-large-v3-turbo.bin
  python3 transcribe.py video.mp4 --lang zh --no-vad
  python3 transcribe.py video.mp4 --merge-gap 3.0 --merge-max 45
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time

DEFAULT_MODEL = "ggml-large-v3-turbo.bin"
DEFAULT_VAD_MODEL = "ggml-silero-v5.1.2.bin"
MODEL_SEARCH_PATHS = [
    ".",
    os.path.expanduser("~/Downloads/whisper-models"),
    os.path.expanduser("~/.local/share/whisper-models"),
    "/usr/local/share/whisper-cpp/models",
]


def find_file(filename, search_paths):
    """在搜索路径中查找文件"""
    if os.path.isabs(filename) and os.path.exists(filename):
        return filename
    for path in search_paths:
        full = os.path.join(path, filename)
        if os.path.exists(full):
            return full
    return None


def find_whisper_cli():
    """查找 whisper-cli 可执行文件"""
    path = shutil.which("whisper-cli")
    if path:
        return path
    for candidate in ["/opt/homebrew/bin/whisper-cli", "/usr/local/bin/whisper-cli"]:
        if os.path.exists(candidate):
            return candidate
    return None


def extract_audio(video_path, audio_path):
    """用 ffmpeg 从视频中提取音频，转为 whisper 要求的 16kHz mono WAV"""
    print(f"[1/3] 提取音频: {video_path}")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ar", "16000",
        "-ac", "1",
        "-c:a", "pcm_s16le",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg 错误:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    print(f"   音频已保存: {audio_path}")


def transcribe(audio_path, whisper_cli, model_path, vad_model_path, language, threads, output_dir, output_name):
    """用 whisper-cli 进行语音识别并输出 SRT 字幕"""
    print("[2/3] 语音识别中...")
    output_prefix = os.path.join(output_dir, output_name)
    cmd = [
        whisper_cli,
        "-m", model_path,
        "-f", audio_path,
        "-l", language,
        "-t", str(threads),
        "-osrt",
        "-of", output_prefix,
    ]
    if vad_model_path:
        cmd += ["--vad", "-vm", vad_model_path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"whisper-cli 错误:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    srt_file = output_prefix + ".srt"
    if not os.path.exists(srt_file):
        print(f"未找到输出文件，whisper-cli 输出:\n{result.stdout}\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    return srt_file


def srt_time_to_sec(t):
    """将 SRT 时间 '00:01:02,340' 转为秒数"""
    t = t.replace(",", ".")
    parts = t.split(":")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])


def sec_to_srt_time(s):
    """将秒数转为 SRT 时间格式 '00:01:02,340'"""
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:06.3f}".replace(".", ",")


def parse_srt(srt_path):
    """解析 SRT 文件，返回 [(start_sec, end_sec, text), ...]"""
    entries = []
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    for block in content.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        time_line = lines[1]
        text = " ".join(lines[2:]).strip()
        if not text:
            continue
        try:
            start_str, end_str = time_line.split(" --> ")
            entries.append((srt_time_to_sec(start_str.strip()), srt_time_to_sec(end_str.strip()), text))
        except (ValueError, IndexError):
            continue
    return entries


def merge_entries(entries, gap_sec, max_duration_sec):
    """合并相邻字幕条目，使每条字幕更长"""
    if not entries:
        return entries

    merged = []
    cur_start, cur_end, cur_text = entries[0]

    for start, end, text in entries[1:]:
        if (start - cur_end) <= gap_sec and (end - cur_start) <= max_duration_sec:
            cur_end = end
            cur_text = cur_text + " " + text
        else:
            merged.append((cur_start, cur_end, cur_text))
            cur_start, cur_end, cur_text = start, end, text

    merged.append((cur_start, cur_end, cur_text))
    return merged


def write_srt(entries, output_path):
    """将条目写回 SRT 格式"""
    with open(output_path, "w", encoding="utf-8") as f:
        for i, (start, end, text) in enumerate(entries, 1):
            f.write(f"{i}\n")
            f.write(f"{sec_to_srt_time(start)} --> {sec_to_srt_time(end)}\n")
            f.write(f"{text}\n\n")


def main():
    parser = argparse.ArgumentParser(description="视频转 SRT 字幕（基于 whisper.cpp）")
    parser.add_argument("video", help="输入视频文件路径")
    parser.add_argument("-m", "--model", default=DEFAULT_MODEL, help=f"whisper 模型文件路径或名称 (默认: {DEFAULT_MODEL})")
    parser.add_argument("-l", "--lang", default="auto", help="语言代码，如 zh/en/ja/auto (默认: auto)")
    parser.add_argument("-t", "--threads", type=int, default=8, help="线程数 (默认: 8)")
    parser.add_argument("-o", "--output-dir", help="输出目录 (默认: 与视频同目录)")
    parser.add_argument("--no-vad", action="store_true", help="禁用 VAD (Voice Activity Detection)")
    parser.add_argument("--vad-model", default=DEFAULT_VAD_MODEL, help=f"VAD 模型文件路径或名称 (默认: {DEFAULT_VAD_MODEL})")
    parser.add_argument("--no-merge", action="store_true", help="禁用字幕合并")
    parser.add_argument("--merge-gap", type=float, default=2.0, help="合并间隔阈值，秒 (默认: 2.0)")
    parser.add_argument("--merge-max", type=float, default=30.0, help="合并后单条最大时长，秒 (默认: 30.0)")
    args = parser.parse_args()

    # 检查视频文件
    video_path = os.path.abspath(args.video)
    if not os.path.exists(video_path):
        print(f"视频文件不存在: {video_path}", file=sys.stderr)
        sys.exit(1)

    # 查找 whisper-cli
    whisper_cli = find_whisper_cli()
    if not whisper_cli:
        print("未找到 whisper-cli，请安装: brew install whisper-cpp", file=sys.stderr)
        sys.exit(1)

    # 查找模型
    model_path = find_file(args.model, MODEL_SEARCH_PATHS)
    if not model_path:
        print(f"未找到模型: {args.model}", file=sys.stderr)
        print(f"搜索路径: {MODEL_SEARCH_PATHS}", file=sys.stderr)
        print("下载: https://huggingface.co/ggerganov/whisper.cpp", file=sys.stderr)
        sys.exit(1)

    # 查找 VAD 模型
    vad_model_path = None
    if not args.no_vad:
        vad_model_path = find_file(args.vad_model, MODEL_SEARCH_PATHS)
        if not vad_model_path:
            print(f"警告: 未找到 VAD 模型 {args.vad_model}，将禁用 VAD", file=sys.stderr)

    # 输出路径
    output_dir = args.output_dir or os.path.dirname(video_path)
    output_name = os.path.splitext(os.path.basename(video_path))[0]

    # 提取音频 → 转录 → 合并
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        audio_path = tmp.name

    try:
        extract_audio(video_path, audio_path)

        t0 = time.time()
        srt_file = transcribe(audio_path, whisper_cli, model_path, vad_model_path, args.lang, args.threads, output_dir, output_name)
        elapsed = time.time() - t0
        print(f"   识别耗时: {elapsed:.1f} 秒")

        if not args.no_merge:
            print(f"[3/3] 合并字幕（间隔 <{args.merge_gap}s, 单条最长 {args.merge_max}s）...")
            entries = parse_srt(srt_file)
            before_count = len(entries)
            entries = merge_entries(entries, args.merge_gap, args.merge_max)
            after_count = len(entries)
            write_srt(entries, srt_file)
            print(f"   合并完成: {before_count} 条 → {after_count} 条")
        else:
            entries = parse_srt(srt_file)

        # 预览
        print("\n--- 字幕预览 ---")
        for start, end, text in entries[:5]:
            print(f"[{sec_to_srt_time(start)} → {sec_to_srt_time(end)}] {text}")
        if len(entries) > 5:
            print(f"... (共 {len(entries)} 条)")

        print(f"\n字幕文件: {srt_file}")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    print("完成!")


if __name__ == "__main__":
    main()
