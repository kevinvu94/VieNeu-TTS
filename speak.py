"""Đọc văn bản bằng giọng đã fine-tune LoRA.

Ví dụ:
    uv run python speak.py "Xin chào các bạn."
    uv run python speak.py --file bai_doc.txt --out outputs/bai_doc.wav
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "finetune" / "output" / "my_voice_fillers" / "merged"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("text", nargs="?", help="văn bản cần đọc")
    ap.add_argument("--file", help="đọc văn bản từ file .txt (UTF-8)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "speak.wav"), help="file .wav kết quả")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="thư mục model đã merge")
    ap.add_argument("--voice", default="Giọng của tôi", help="tên giọng đã đóng gói bằng make_voice.py")
    ap.add_argument("--fillers", action="store_true", help="tự chèn từ đệm (à, ừ, nha, nhá, ...) cho tự nhiên")
    ap.add_argument("--p-start", type=float, default=0.1, help="xác suất filler đầu câu (mặc định 0.1)")
    ap.add_argument("--p-end", type=float, default=0.2, help="xác suất filler cuối câu (mặc định 0.2)")
    ap.add_argument("--seed", type=int, default=None, help="seed để chèn filler lặp lại được")
    args = ap.parse_args()

    if args.file:
        text = Path(args.file).read_text(encoding="utf-8").strip()
    else:
        text = (args.text or "").strip()
    if not text:
        ap.error("cần truyền văn bản hoặc --file")
    if not Path(args.model).is_dir():
        sys.exit(f"Không thấy model: {args.model}")

    if args.fillers:
        from fillers import add_fillers

        text = add_fillers(text, p_start=args.p_start, p_end=args.p_end, seed=args.seed)
        print(f"Văn bản sau khi chèn filler:\n{text}\n")

    from vieneu import Vieneu

    tts = Vieneu(mode="v3turbo", backbone_repo=args.model)
    audio = tts.infer(text, voice=args.voice)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tts.save(audio, str(out))
    print(f"Đã lưu: {out}")


if __name__ == "__main__":
    main()
