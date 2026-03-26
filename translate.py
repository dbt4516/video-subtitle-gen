#!/usr/bin/env python3
"""
使用 NLLB-200 将 SRT 字幕文件翻译为目标语言。完全本地离线运行。
支持 200+ 语言互译，包括日→中、英→中、中→英等。

依赖（在 venv 中安装）：
  pip install transformers sentencepiece torch

用法：
  python3 translate.py subtitle.srt --from ja --to zh
  python3 translate.py /path/to/srts/ --from ja --to zh
  python3 translate.py subtitle.srt --from en --to zh --batch-size 16
"""

import argparse
import glob
import os
import sys
import time

# NLLB 使用 BCP-47 风格的语言代码
LANG_TO_NLLB = {
    "zh": "zho_Hans",
    "zh-tw": "zho_Hant",
    "ja": "jpn_Jpan",
    "en": "eng_Latn",
    "ko": "kor_Hang",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "ru": "rus_Cyrl",
    "pt": "por_Latn",
    "ar": "arb_Arab",
    "th": "tha_Thai",
    "vi": "vie_Latn",
    "id": "ind_Latn",
}

DEFAULT_MODEL = "facebook/nllb-200-distilled-600M"


def get_nllb_code(lang):
    """将简短语言代码转为 NLLB 代码"""
    code = LANG_TO_NLLB.get(lang)
    if not code:
        print(f"不支持的语言代码: {lang}", file=sys.stderr)
        print(f"支持的语言: {', '.join(sorted(LANG_TO_NLLB.keys()))}", file=sys.stderr)
        sys.exit(1)
    return code


def load_model(model_name, src_lang):
    """加载 NLLB 模型和 tokenizer"""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    print(f"加载模型: {model_name}")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    print(f"模型加载完成 ({time.time() - t0:.1f} 秒)")
    return model, tokenizer


def translate_texts(texts, model, tokenizer, tgt_lang_code, batch_size=8):
    """批量翻译文本列表"""
    results = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        translated = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang_code),
            max_new_tokens=512,
        )
        results.extend(tokenizer.batch_decode(translated, skip_special_tokens=True))
    return results


def parse_srt(srt_path):
    """解析 SRT 文件，返回 [(index, time_line, text), ...]"""
    entries = []
    with open(srt_path, "r", encoding="utf-8") as f:
        content = f.read()

    for block in content.strip().split("\n\n"):
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue
        index = lines[0]
        time_line = lines[1]
        text = "\n".join(lines[2:])
        entries.append((index, time_line, text))
    return entries


def write_srt(entries, output_path):
    """将翻译后的条目写回 SRT 格式"""
    with open(output_path, "w", encoding="utf-8") as f:
        for index, time_line, text in entries:
            f.write(f"{index}\n{time_line}\n{text}\n\n")


def find_srt_files(input_path):
    """根据输入路径返回 SRT 文件列表"""
    input_path = os.path.abspath(input_path)
    if os.path.isfile(input_path):
        return [input_path]
    if os.path.isdir(input_path):
        files = glob.glob(os.path.join(input_path, "*.srt"))
        files += glob.glob(os.path.join(input_path, "*.SRT"))
        return sorted(set(files))
    print(f"路径不存在: {input_path}", file=sys.stderr)
    sys.exit(1)


def translate_srt(srt_path, model, tokenizer, tgt_lang, tgt_lang_code, batch_size, output_dir):
    """翻译单个 SRT 文件"""
    entries = parse_srt(srt_path)
    if not entries:
        print(f"  空文件，跳过")
        return None

    texts = [e[2] for e in entries]

    t0 = time.time()
    translated = translate_texts(texts, model, tokenizer, tgt_lang_code, batch_size)
    elapsed = time.time() - t0

    out_entries = [(e[0], e[1], t) for e, t in zip(entries, translated)]

    basename = os.path.basename(srt_path)
    name, _ = os.path.splitext(basename)
    out_name = f"{name}.{tgt_lang}.srt"
    out_dir = output_dir or os.path.dirname(srt_path)
    out_path = os.path.join(out_dir, out_name)

    write_srt(out_entries, out_path)
    print(f"  翻译完成: {len(entries)} 条, 耗时 {elapsed:.1f} 秒 → {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="SRT 字幕翻译（基于 NLLB-200，支持 200+ 语言，完全离线）")
    parser.add_argument("input", help="SRT 文件或包含 SRT 的目录")
    parser.add_argument("--from", dest="src_lang", required=True, help=f"源语言: {', '.join(sorted(LANG_TO_NLLB.keys()))}")
    parser.add_argument("--to", dest="tgt_lang", required=True, help=f"目标语言: {', '.join(sorted(LANG_TO_NLLB.keys()))}")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"HuggingFace 模型名 (默认: {DEFAULT_MODEL})")
    parser.add_argument("--batch-size", type=int, default=8, help="翻译批次大小 (默认: 8)")
    parser.add_argument("-o", "--output-dir", help="输出目录 (默认: 与源文件同目录)")
    args = parser.parse_args()

    src_code = get_nllb_code(args.src_lang)
    tgt_code = get_nllb_code(args.tgt_lang)

    srt_files = find_srt_files(args.input)
    if not srt_files:
        print("未找到 SRT 文件", file=sys.stderr)
        sys.exit(1)

    try:
        model, tokenizer = load_model(args.model, src_code)
    except Exception as e:
        print(f"加载模型失败: {e}", file=sys.stderr)
        sys.exit(1)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    total = len(srt_files)
    success = 0
    for i, srt_path in enumerate(srt_files, 1):
        print(f"\n[{i}/{total}] {os.path.basename(srt_path)}")
        try:
            result = translate_srt(srt_path, model, tokenizer, args.tgt_lang, tgt_code, args.batch_size, args.output_dir)
            if result:
                success += 1
        except Exception as e:
            print(f"  错误: {e}", file=sys.stderr)

    print(f"\n完成: {success}/{total} 个文件翻译成功")


if __name__ == "__main__":
    main()
