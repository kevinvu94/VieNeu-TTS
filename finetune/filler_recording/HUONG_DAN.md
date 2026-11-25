# Thu âm bộ dữ liệu filler (bước 2)

`metadata_fillers.csv` có 120 câu (clip077 → clip196), nối tiếp 76 clip hiện có.
Kịch bản đọc dễ theo dõi nằm ở `KICH_BAN_THU_AM.md`. Mỗi filler xuất hiện nhiều lần ở đầu hoặc cuối câu, có xen câu thường để cân bằng.

## Cách thu
1. Cùng micro, cùng phòng, cùng khoảng cách với 76 clip cũ (phải cùng một giọng/chất âm thanh).
2. Mỗi dòng đọc thành **một file riêng** `clipXXX.wav` (3–10 giây). Đọc **đúng từng chữ** trong câu, kể cả filler.
3. Đọc như đang nói chuyện: filler đọc nhẹ, nhanh, không nhấn mạnh, đừng đọc như đọc bài.
4. Chừa ~0.3 giây im lặng đầu và cuối; tránh tiếng thở mạnh, tiếng click.
5. Chép các file vào `finetune/dataset/raw_audio/`.
6. Nối metadata: `cat finetune/filler_recording/metadata_fillers.csv >> finetune/dataset/metadata.csv`
   (kiểm tra file cũ có kết thúc bằng xuống dòng trước khi nối).

## Thu một file dài rồi để máy cắt
Đọc lần lượt các câu trong `KICH_BAN_THU_AM.md`, **nghỉ ~1.5–2 giây giữa hai câu** (ngắt hơi trong câu thì ngắn thôi), rồi:
```bash
uv run python finetune/filler_recording/split_recording.py thu_am.wav --dry-run   # xem thử
uv run python finetune/filler_recording/split_recording.py thu_am.wav             # cắt thật vào dataset/raw_audio
```
Thu nhiều đợt thì thêm `--start 21 --count 20` cho đợt 2, v.v. Nếu đọc sai/đọc lại một câu, thu lại câu đó riêng.

## Train lại (khi đã thu xong)
```bash
uv run python finetune/prepare_dataset.py --dataset-dir finetune/dataset
uv run python finetune/train_lora.py --data finetune/dataset/train.parquet --run my_voice_fillers --epochs 20 --merge
uv run python finetune/make_voice.py --audio <1 clip của bạn> --name "Giọng của tôi" --out finetune/output/my_voice_fillers/merged
```
Rồi chạy `speak.py --model finetune/output/my_voice_fillers/merged`. Nếu muốn thêm filler tự động,
dùng kèm `--fillers`; nhớ dùng tỷ lệ thấp (`--p-start 0.15 --p-end 0.2`) để khỏi nghe lặp.
Muốn thêm filler khác thì viết thêm ≥ 8–10 câu cho mỗi filler đó.
