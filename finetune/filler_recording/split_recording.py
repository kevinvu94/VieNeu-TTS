"""Cắt một file thu âm dài thành từng clip theo câu, đặt tên clipXXX.wav theo kịch bản.

Nguyên tắc: người đọc nghỉ giữa hai câu *lâu hơn* chỗ ngắt hơi trong câu. Script tìm các
khoảng lặng, lấy N-1 khoảng lặng dài nhất làm ranh giới giữa N câu (N = số dòng trong
metadata_fillers.csv), rồi cắt. Không cần ASR.

    uv run python finetune/filler_recording/split_recording.py thu_am.wav --dry-run
    uv run python finetune/filler_recording/split_recording.py thu_am.wav

File vào có thể là wav/mp3/m4a/flac... (cần ffmpeg cho định dạng khác wav).
Nếu thu thành nhiều đợt thì cắt từng đợt: --start 1 --count 20 (câu 1-20), --start 21 --count 20 ...
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
DEFAULT_META = HERE / "metadata_fillers.csv"
DEFAULT_OUT = HERE.parent / "dataset" / "raw_audio"
HOP_S = 0.02
MIN_GAP_S = 0.25   # chỉ xét khoảng lặng từ 0.25 giây trở lên
PAD_S = 0.2        # chừa lại tối đa 0.2 giây im lặng ở hai đầu mỗi clip


def load_audio(path: Path):
    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    except Exception:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "in.wav"
            subprocess.run(["ffmpeg", "-y", "-i", str(path), "-ac", "1", str(tmp)],
                           check=True, capture_output=True)
            data, sr = sf.read(str(tmp), dtype="float32", always_2d=True)
    return data.mean(axis=1), sr


def find_gaps(x: np.ndarray, sr: int, min_gap: float = MIN_GAP_S):
    hop = int(sr * HOP_S)
    n = len(x) // hop
    frames = x[: n * hop].reshape(n, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    lo, hi = np.percentile(db, 5), np.percentile(db, 95)
    thr = lo + 0.25 * (hi - lo)
    silent = db < thr
    gaps, start = [], None
    for i, s in enumerate(silent):
        if s and start is None:
            start = i
        elif not s and start is not None:
            gaps.append((start, i))
            start = None
    if start is not None:
        gaps.append((start, n))
    out = []
    for a, b in gaps:
        if a == 0 or b == n:       # bỏ khoảng lặng đầu/cuối file
            continue
        if (b - a) * HOP_S >= min_gap:
            out.append((a * hop, b * hop))
    return out, lo, hi


# ---------------------------------------------------------------- ASR alignment
import difflib
import re
import unicodedata


def _norm_tokens(text: str):
    text = unicodedata.normalize("NFC", text.lower())
    return re.findall(r"[^\W_]+", text)


def asr_words(x: np.ndarray, sr: int, model_id: str, max_win: float = 18.0):
    """Chạy Whisper trên từng cụm câu (cắt tại chỗ im lặng), trả về [(token, start_s, end_s)]."""
    import librosa
    import torch
    from transformers import pipeline

    y = librosa.resample(x, orig_sr=sr, target_sr=16000) if sr != 16000 else x
    y = y.astype("float32")
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = pipeline("automatic-speech-recognition", model=model_id, device=dev,
                    dtype=torch.float16 if dev != "cpu" else torch.float32)

    # cụm = các đoạn có tiếng gộp lại, tối đa max_win giây, ranh giới nằm giữa một khoảng lặng
    gaps, _, _ = find_gaps(x, sr, 0.3)
    mids = [(a + b) // 2 for a, b in gaps]
    edges, last = [0], 0
    for i, m in enumerate(mids):
        nxt = mids[i + 1] if i + 1 < len(mids) else len(x)
        if (nxt - last) / sr > max_win and (m - last) / sr > 1.0:
            edges.append(m)
            last = m
    edges.append(len(x))
    words = []
    for a, b in zip(edges[:-1], edges[1:]):
        seg = y[int(a / sr * 16000): int(b / sr * 16000)]
        if len(seg) < 3200:
            continue
        out = pipe({"raw": seg, "sampling_rate": 16000}, return_timestamps="word",
                   generate_kwargs={"language": "vi", "task": "transcribe"})
        off = a / sr
        for ch in out["chunks"]:
            t0, t1 = ch["timestamp"]
            if t0 is None:
                continue
            if t1 is None:
                t1 = t0 + 0.3
            for tok in _norm_tokens(ch["text"]):
                words.append((tok, off + float(t0), off + float(t1)))
    return words


def align_sentences(words, sentences, slack: int = 20):
    """Khớp tuần tự từ ASR với từng câu kịch bản (theo đúng thứ tự).
    Trả về [(start_s, end_s, n_matched, n_total)] cho mỗi câu (None nếu không khớp)."""
    asr = [w[0] for w in words]
    ptr, spans = 0, []
    for t in sentences:
        toks = _norm_tokens(t)
        total = len(toks)
        best = None
        for s0 in range(ptr, min(len(asr), ptr + slack)):
            sm = difflib.SequenceMatcher(None, asr[s0:s0 + total + 3], toks, autojunk=False)
            blocks = [b for b in sm.get_matching_blocks() if b.size]
            cnt = sum(b.size for b in blocks)
            if best is None or cnt > best[0]:      # '>' giữ lại vị trí sớm nhất khi hoà
                best = (cnt, s0, blocks)
        if best and best[0] >= max(2, 0.5 * total):
            idx = [best[1] + b.a + j for b in best[2] for j in range(b.size)]
            spans.append((min(words[i][1] for i in idx), max(words[i][2] for i in idx), len(idx), total))
            ptr = max(idx) + 1
        else:
            spans.append(None)
    return spans


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", help="file thu âm dài")
    ap.add_argument("--meta", default=str(DEFAULT_META), help="metadata_fillers.csv")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="thư mục ghi clip")
    ap.add_argument("--start", type=int, default=1, help="câu đầu tiên có trong file (1-based)")
    ap.add_argument("--count", type=int, default=0, help="số câu trong file (mặc định: tới hết kịch bản)")
    ap.add_argument("--asr", action="store_true", help="dùng Whisper để khớp lời với kịch bản (chính xác hơn)")
    ap.add_argument("--asr-model", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--dry-run", action="store_true", help="chỉ báo cáo, không ghi file")
    args = ap.parse_args()

    rows = [l.rstrip("\n").split("|", 1) for l in open(args.meta, encoding="utf-8") if l.strip()]
    sel = rows[args.start - 1:]
    if args.count:
        sel = sel[: args.count]
    n = len(sel)

    x, sr = load_audio(Path(args.audio))
    pad = int(PAD_S * sr)
    hop = int(sr * HOP_S)
    gaps, lo, hi = find_gaps(x, sr)
    frames_db = 20 * np.log10(np.sqrt((x[: (len(x) // hop) * hop].reshape(-1, hop) ** 2).mean(axis=1) + 1e-12))
    voiced = np.nonzero(frames_db >= lo + 0.25 * (hi - lo))[0]
    first_voice, last_voice = voiced[0] * hop, (voiced[-1] + 1) * hop
    print(f"File dài {len(x)/sr:.1f}s, {sr} Hz; cần {n} câu.")
    info = [""] * n
    final = [None] * n   # (s2, e2) đã gồm phần đệm

    if args.asr:
        import json
        cache = Path(args.audio).with_suffix(".asr.json")
        if cache.exists():
            words = [tuple(w) for w in json.loads(cache.read_text(encoding="utf-8"))]
        else:
            words = asr_words(x, sr, args.asr_model)
            cache.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
        print(f"Whisper nghe được {len(words)} từ.")
        spans = align_sentences(words, [t for _, t in sel])
        small, _, _ = find_gaps(x, sr, 0.12)
        # Câu Whisper không nghe ra (ảo giác/nghe sót): chia đoạn giữa hai câu đã khớp theo khoảng lặng dài nhất
        est = set()
        k = 0
        while k < n:
            if spans[k] is not None:
                k += 1
                continue
            j = k
            while j + 1 < n and spans[j + 1] is None:
                j += 1
            r0 = spans[k - 1][1] * sr if k > 0 else first_voice
            r1 = spans[j + 1][0] * sr if j + 1 < n else last_voice
            m = j - k + 1
            inside = [g for g in small if g[0] >= r0 - 0.1 * sr and g[1] <= r1 + 0.1 * sr]
            v0, v1 = r0, r1
            # bỏ khoảng lặng sát hai đầu đoạn (giữa câu đã khớp và câu cần tìm)
            if inside and inside[0][0] <= r0 + 0.8 * sr:
                v0 = inside[0][1]
                inside = inside[1:]
            if inside and inside[-1][1] >= r1 - 0.8 * sr:
                v1 = inside[-1][0]
                inside = inside[:-1]
            if len(inside) >= m - 1:
                # chọn m-1 khoảng lặng sao cho độ dài các câu sát nhất với tỉ lệ số từ
                wts = [len(_norm_tokens(sel[i][1])) for i in range(k, j + 1)]
                best_c, best_cost = None, None
                from itertools import combinations
                if len(inside) > 24:
                    inside = sorted(sorted(inside, key=lambda g: g[1] - g[0], reverse=True)[:24])
                for comb in combinations(range(len(inside)), m - 1):
                    gs = [inside[i] for i in comb]
                    edges = [v0] + [c for g in gs for c in (g[0], g[1])] + [v1]
                    durs = [(edges[2 * t + 1] - edges[2 * t]) / sr for t in range(m)]
                    voiced_total = sum(durs)
                    cost = sum(((d - voiced_total * wt / sum(wts)) / max(1.0, voiced_total * wt / sum(wts))) ** 2
                               for d, wt in zip(durs, wts)) - 0.3 * sum((g[1] - g[0]) / sr for g in gs)
                    if best_cost is None or cost < best_cost:
                        best_c, best_cost = edges, cost
                if best_c is not None:
                    for t, idx in enumerate(range(k, j + 1)):
                        spans[idx] = (best_c[2 * t] / sr, best_c[2 * t + 1] / sr, 0, wts[t])
                        est.add(idx)
            k = j + 1
        centers = []   # khoảng lặng dùng để cắt giữa câu k và k+1
        for k in range(n - 1):
            if spans[k] is None or spans[k + 1] is None:
                centers.append(None)
                continue
            e, st = spans[k][1] * sr, spans[k + 1][0] * sr
            w0, w1 = e - 0.4 * sr, st + 0.4 * sr
            cand = [g for g in small if g[1] > w0 and g[0] < w1]
            if cand:
                centers.append(max(cand, key=lambda g: g[1] - g[0]))
            else:
                m = int((e + st) / 2)
                centers.append((m, m))
        for k in range(n):
            if spans[k] is None:
                info[k] = "  ⚠ KHÔNG KHỚP với bản nghe được"
                continue
            prev_c = centers[k - 1] if k > 0 else None
            next_c = centers[k] if k < n - 1 else None
            s = prev_c[1] if prev_c else (first_voice if k == 0 else int(max(0, spans[k][0] - 0.3) * sr))
            e = next_c[0] if next_c else (last_voice if k == n - 1 else int(spans[k][1] * sr + 0.3 * sr))
            s2 = s - pad if not prev_c else max(s - pad, (prev_c[0] + prev_c[1]) // 2)
            e2 = e + pad if not next_c else min(e + pad, (next_c[0] + next_c[1]) // 2)
            final[k] = (max(0, s2), min(len(x), e2))
            info[k] = f"  khớp {spans[k][2]}/{spans[k][3]} từ"
            if k in est:
                info[k] = "  ⚠ ƯỚC LƯỢNG theo khoảng lặng (Whisper không nghe ra) - hãy nghe kiểm tra"
            elif spans[k][2] < 0.7 * spans[k][3]:
                info[k] += "  ⚠ khớp thấp"
    else:
        print(f"Tìm thấy {len(gaps)} khoảng lặng ≥ {MIN_GAP_S}s.")
        if len(gaps) < n - 1:
            sys.exit(f"Không đủ khoảng lặng ({len(gaps)} < {n-1}).")
        order = sorted(range(len(gaps)), key=lambda i: gaps[i][1] - gaps[i][0], reverse=True)
        cuts = [gaps[i] for i in sorted(order[: n - 1])]
        mi = (gaps[order[n - 2]][1] - gaps[order[n - 2]][0]) / sr if n > 1 else 0
        mo = (gaps[order[n - 1]][1] - gaps[order[n - 1]][0]) / sr if len(order) >= n else 0
        print(f"Khoảng nghỉ giữa câu ngắn nhất: {mi:.2f}s | khoảng ngắt hơi trong câu dài nhất: {mo:.2f}s")
        if mi < mo * 1.3:
            print("⚠ Hai mức này quá gần nhau: dùng --asr để cắt chính xác hơn.")
        starts = [first_voice] + [b for (_, b) in cuts]
        ends = [a for (a, _) in cuts] + [last_voice]
        for k, (s, e) in enumerate(zip(starts, ends)):
            final[k] = (max(0, s - pad), min(len(x), e + pad))

    outdir = Path(args.out)
    if not args.dry_run:
        outdir.mkdir(parents=True, exist_ok=True)
    warn = 0
    for k, (name, text) in enumerate(sel):
        if final[k] is None:
            print(f"{name}   ----  {text[:50]}{info[k]}")
            warn += 1
            continue
        s2, e2 = final[k]
        dur = (e2 - s2) / sr
        flag = info[k]
        if dur < 1.5 or dur > 15:
            flag += "  ⚠ độ dài bất thường"
        if "⚠" in flag:
            warn += 1
        print(f"{name}  {dur:5.1f}s  {text[:50]}{flag}")
        if not args.dry_run:
            sf.write(str(outdir / name), x[s2:e2], sr)
    print(f"\n{'(dry-run, chưa ghi file) ' if args.dry_run else f'Đã xử lý {n} clip, ghi vào {outdir}. '}"
          f"{warn} clip cần kiểm tra." )


if __name__ == "__main__":
    main()
