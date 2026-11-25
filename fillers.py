"""Chèn từ đệm (filler) tiếng Việt vào văn bản trước khi đưa cho TTS.

Ví dụ:
    from fillers import add_fillers
    add_fillers("Cái này dễ lắm. Bạn thử xem.", seed=1)
"""
import random
import re

START_FILLERS = ["à", "ừ"]
END_FILLERS = ["nha", "nhá"]

_SENT_RE = re.compile(r"[^.!?\n]+[.!?]*|\n+")
_END_WORDS = {"nha", "nhá", "nhé", "mà", "ấy", "cơ", "nhỉ", "đấy", "đó", "thôi"}
_START_WORDS = ("à", "ừ", "ờ", "nói chung", "cơ mà", "thực ra", "vâng", "dạ")


def _lower_first(sentence: str) -> str:
    """Hạ chữ cái đầu câu, trừ khi từ đầu là viết tắt/tên riêng kiểu IELTS, CEO."""
    first = sentence.split(" ", 1)[0]
    if len(first) > 1 and first[1:].isupper():
        return sentence
    return sentence[:1].lower() + sentence[1:]


def add_fillers(
    text: str,
    p_start: float = 0.1,
    p_end: float = 0.2,
    min_words: int = 5,
    seed: int | None = None,
) -> str:
    """Chèn filler đầu/cuối câu theo xác suất.

    - Chỉ thêm filler cuối câu cho câu kết thúc bằng dấu chấm (không phải câu hỏi/cảm thán).
    - Bỏ qua câu ngắn (< min_words từ) và câu đã có sẵn filler.
    - Không chèn filler đầu câu ở hai câu liên tiếp để khỏi lặp.
    """
    rng = random.Random(seed)
    out: list[str] = []
    prev_started = False
    pieces = _SENT_RE.findall(text)
    for idx, piece in enumerate(pieces):
        if piece.startswith("\n") or not piece.strip():
            out.append(piece)
            prev_started = False
            continue
        lead = piece[: len(piece) - len(piece.lstrip())]
        body = piece.strip()
        m = re.match(r"^(.*?)([.!?]*)$", body, re.S)
        core, punct = m.group(1), m.group(2)
        words = core.split()
        started = False
        if len(words) >= min_words and core:
            low = core.lower()
            # filler đầu câu (bỏ qua câu đầu tiên của đoạn để mở bài vẫn trang trọng)
            if (
                idx > 0
                and not prev_started
                and not low.startswith(_START_WORDS)
                and rng.random() < p_start
            ):
                f = rng.choice(START_FILLERS)
                core = f"{f}, {_lower_first(core)}"
                core = core[:1].upper() + core[1:]
                started = True
            # filler cuối câu
            if punct == "." and words[-1].lower() not in _END_WORDS and rng.random() < p_end:
                core = f"{core} {rng.choice(END_FILLERS)}"
        prev_started = started
        out.append(f"{lead}{core}{punct}")
    return "".join(out) if "\n" in text else " ".join(s.strip() for s in out if s.strip())
