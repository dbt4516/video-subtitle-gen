"""
VideoMAE 视频动作检测 + 片段剪辑工具
本地离线运行，数据不离机

用法:
    python3 detect.py <视频路径或目录> <动作1> [动作2] [动作3] ...

示例:
    python3 detect.py video.mp4 "eating food" "drinking" 

可用动作列表:
    python3 detect.py --list
"""

import sys
import os
import subprocess
import warnings
import numpy as np

# 强制实时输出，不缓冲
sys.stdout.reconfigure(line_buffering=True)

warnings.filterwarnings("ignore")


# ── 参数配置 ──────────────────────────────────────────────
STRIDE = 5          # 每隔几秒采样一次（秒）
CONFIDENCE = 0.12   # 命中置信度阈值
PADDING = 3.0       # 片段前后各补几秒
MERGE_GAP = 10      # 两个命中点间隔小于此值则合并（秒）
MIN_DUR = 3.0       # 最短片段时长（秒）
NUM_FRAMES = 16     # 每次送入模型的帧数
# ─────────────────────────────────────────────────────────


def load_model():
    import torch
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"设备: {device}")
    print("加载模型中...")
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    model = model.to(device)
    model.eval()
    print("模型加载完成\n")
    return processor, model, device


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".flv", ".wmv", ".m4v", ".webm"}


def collect_videos(path):
    """返回路径下所有视频文件列表（单文件直接返回，目录则递归收集）"""
    if os.path.isfile(path):
        return [path]
    videos = []
    for root, _, files in os.walk(path):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS:
                videos.append(os.path.join(root, f))
    return videos


def get_video_duration(video_path):
    import av
    with av.open(video_path) as c:
        return float(c.duration) / 1_000_000


def get_frames(video_path, start_sec, n=16):
    import av
    frames = []
    with av.open(video_path) as c:
        s = c.streams.video[0]
        c.seek(int(start_sec * 1_000_000))
        for frame in c.decode(s):
            if float(frame.pts * s.time_base) >= start_sec:
                frames.append(frame.to_ndarray(format="rgb24"))
            if len(frames) >= n:
                break
    while len(frames) < n:
        if frames:
            frames.append(frames[-1])
        else:
            break
    return frames


def make_segments(hit_times, duration):
    if not hit_times:
        return []
    segments, s, e = [], hit_times[0], hit_times[0]
    for ht in hit_times[1:]:
        if ht - e <= MERGE_GAP:
            e = ht
        else:
            segments.append((max(0, s - PADDING), min(duration, e + PADDING)))
            s, e = ht, ht
    segments.append((max(0, s - PADDING), min(duration, e + PADDING)))
    return [(s, e) for s, e in segments if e - s >= MIN_DUR]


def export_clips(video_path, segments, output_dir, slug):
    os.makedirs(output_dir, exist_ok=True)
    clip_paths = []
    for i, (s, e) in enumerate(segments):
        out = os.path.join(output_dir, f"{slug}_clip{i+1:02d}_{s:.0f}s-{e:.0f}s.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-ss", str(s), "-to", str(e), "-i", video_path, "-c", "copy", out],
            capture_output=True,
        )
        size_kb = os.path.getsize(out) // 1024
        print(f"    片段{i+1}: {s:.0f}s ~ {e:.0f}s ({e-s:.0f}秒)  {size_kb}KB")
        clip_paths.append(out)
    return clip_paths


def merge_clips(clip_paths, merged_path):
    if len(clip_paths) == 1:
        import shutil
        shutil.copy(clip_paths[0], merged_path)
    else:
        lst = "/tmp/_detect_merge_list.txt"
        with open(lst, "w") as f:
            for p in clip_paths:
                f.write(f"file '{p}'\n")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", merged_path],
            capture_output=True,
        )
    size_mb = os.path.getsize(merged_path) / 1024 / 1024
    print(f"    合并文件: {os.path.basename(merged_path)} ({size_mb:.1f}MB)")


def print_all_labels():
    from transformers import VideoMAEForVideoClassification
    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
    labels = sorted(model.config.id2label.values())
    print(f"共 {len(labels)} 个可识别动作:\n")
    for i, label in enumerate(labels):
        print(f"  {label}")


def detect_one(video_path, target_ids, processor, model, device, base_dir):
    """处理单个视频，出错时抛出异常由上层捕获"""
    import torch

    duration = get_video_duration(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_base = os.path.join(os.path.dirname(video_path), f"{video_name}_clips")

    print(f"视频: {os.path.basename(video_path)}  时长: {duration:.0f}秒")
    print(f"置信度阈值: {CONFIDENCE:.0%}  采样间隔: {STRIDE}s\n")

    hits = {label: [] for label in target_ids}
    t = 0.0
    total = int(duration / STRIDE)
    step = 0

    while t + 4 <= duration:
        frames = get_frames(video_path, t, NUM_FRAMES)
        if len(frames) == NUM_FRAMES:
            inputs = {k: v.to(device) for k, v in processor(frames, return_tensors="pt").items()}
            with torch.no_grad():
                out = model(**inputs)
            probs = torch.softmax(out.logits, dim=-1)[0]

            found = []
            for label, tid in target_ids.items():
                p = probs[tid].item()
                if p >= CONFIDENCE:
                    hits[label].append(t)
                    found.append(f"{label}({p:.0%})")

            if found:
                print(f"  [{t:6.0f}s] {' | '.join(found)}")
            else:
                print(f"  [{t:6.0f}s] {step}/{total}", end="\r")
        t += STRIDE
        step += 1

    # 收集所有命中时间点（合并所有动作），按时间排序去重
    all_hits = sorted(set(t for ht in hits.values() for t in ht))

    print(f"\n\n── 检测结果 ──")
    for label, ht in hits.items():
        print(f"【{label}】命中 {len(ht)} 个时间点")

    all_segments = make_segments(all_hits, duration)
    if not all_segments:
        print("\n未检测到目标片段")
        return

    print(f"\n共 {len(all_segments)} 个片段（所有动作合并）:")
    tmp_dir = os.path.join(output_base, "_tmp")
    all_clips = export_clips(video_path, all_segments, tmp_dir, "clip")

    output_dir = os.path.join(base_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{video_name}_output.mp4")
    merge_clips(all_clips, output_file)

    # 清理临时目录
    import shutil
    shutil.rmtree(output_base)

    print(f"输出文件: {output_file}")


def detect(input_path, targets):
    if not os.path.exists(input_path):
        print(f"错误: 路径不存在 {input_path}")
        sys.exit(1)

    processor, model, device = load_model()

    # 验证 target 名称（只做一次）
    all_labels = set(model.config.id2label.values())
    target_ids = {}
    for target in targets:
        matches = [v for v in all_labels if target.lower() in v.lower()]
        if not matches:
            print(f"警告: 未找到动作 '{target}'，跳过")
            continue
        best = min(matches, key=len)
        tid = [k for k, v in model.config.id2label.items() if v == best][0]
        target_ids[best] = tid
        if best != target:
            print(f"'{target}' → 匹配到 '{best}'")

    if not target_ids:
        print("没有有效的目标动作，退出")
        sys.exit(1)

    print(f"目标动作: {list(target_ids.keys())}\n")

    videos = collect_videos(input_path)
    if not videos:
        print(f"未找到视频文件: {input_path}")
        sys.exit(1)

    print(f"共找到 {len(videos)} 个视频\n{'─' * 50}")

    # 以 input_path 所在目录（单文件）或 input_path 本身（目录）作为基准
    base_dir = input_path if os.path.isdir(input_path) else os.path.dirname(input_path)
    success_dir = os.path.join(base_dir, "success")
    fail_dir = os.path.join(base_dir, "fail")
    os.makedirs(success_dir, exist_ok=True)
    os.makedirs(fail_dir, exist_ok=True)

    skipped = []
    for i, video_path in enumerate(videos):
        print(f"\n[{i+1}/{len(videos)}] {os.path.basename(video_path)}")
        try:
            detect_one(video_path, target_ids, processor, model, device, base_dir)
            dst = os.path.join(success_dir, os.path.basename(video_path))
            os.rename(video_path, dst)
            print(f"  → 移动至 success/")
        except Exception as e:
            print(f"  ⚠ 跳过（{e}）")
            skipped.append((video_path, str(e)))
            dst = os.path.join(fail_dir, os.path.basename(video_path))
            os.rename(video_path, dst)
            print(f"  → 移动至 fail/")

    print(f"\n{'─' * 50}")
    print(f"完成: {len(videos) - len(skipped)}/{len(videos)} 个视频处理成功")
    if skipped:
        print("失败的文件:")
        for path, reason in skipped:
            print(f"  {os.path.basename(path)}: {reason}")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    if sys.argv[1] == "--list":
        print_all_labels()
        sys.exit(0)

    if len(sys.argv) < 3:
        print("用法: python3 detect.py <视频路径> <动作1> [动作2] ...")
        print("查看所有动作: python3 detect.py --list")
        sys.exit(1)

    input_path = sys.argv[1]
    targets = sys.argv[2:]
    detect(input_path, targets)


if __name__ == "__main__":
    main()
