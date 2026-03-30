# TATA Steel – YOLOv5 Seg + PaddleOCR (Side View)

This project runs **YOLOv5 segmentation + SORT tracking + PaddleOCR** on a video stream to:

- **Count** detected classes when they cross a configurable counting line
- **Read wagon IDs** (11-digit) via OCR for the detected `wagon_id` class
- **Save** an annotated output video + CSV + summary JSON + OCR debug crops

## What to run

Main script:

- `paddleocr/paddle_ocr.py`

Your saved command (from `paddleocr/command.txt`):

```bash
python paddle_ocr.py --img 640 --weights "D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Manfucturing_sector\TATA_steel\enginnering\model\tata_steel_side_view.pt" --project D:\bhanu\tata_steel\top_view\side_view_ --name side_view_paddle_ocr --source "D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Manfucturing_sector\TATA_steel\enginnering\input_videos\side_view.mp4" --conf 0.60
```

Run it from the `paddleocr/` folder:

```bash
cd paddleocr
python paddle_ocr.py --weights "<path-to-weights>.pt" --source "<path-to-video-or-camera>"
```

## CLI arguments (most important)

- `--weights`: path to YOLOv5 **segmentation** model (`.pt`)
- `--source`: video path OR camera index as string (example: `"0"`)
- `--axis`: counting direction (`x` or `y`)  
- `--imgsz`: inference size (default `640`)
- `--conf-thres`: confidence threshold
- `--iou-thres`: NMS IoU threshold
- `--device`: device string (YOLOv5 style)
- `--project`, `--name`: output folder
- `--ocr-class`: optional override for which class to OCR (auto-detects `wagon_id` / `wagon_number` / `number` / `id`)

## Outputs

The run creates a new folder under:

- `<project>/<name>`, auto-incremented if it already exists

Inside that folder:

- **`output.mp4`**: annotated video (masks, tracking, count HUD, confirmed wagon panel)
- **`ocr_results.csv`**: wagon OCR results with timestamps and frame number
- **`summary.json`**: class counts + tracker-id → wagon-number map
- **`ocr_debug_crops/`**: saved crops used for OCR (for debugging)

## Notes

- The script **tracks objects** using SORT and counts when a track crosses the line in the configured direction.
- OCR output is normalized to digits and validated as an **11-digit** wagon number.
- Press `Esc` to stop the window playback.

