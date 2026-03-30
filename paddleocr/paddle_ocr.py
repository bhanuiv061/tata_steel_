# import argparse
# import sys
# import csv
# import json
# from datetime import datetime
# from pathlib import Path
# import cv2
# import torch
# import numpy as np
# import pathlib
# from tqdm import tqdm
# import signal
# import re
# from collections import Counter

# # ================= PaddleOCR =================
# from paddleocr import PaddleOCR

# import sys
# import os

# yolov5_path = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Manfucturing_sector\TATA_steel\enginnering\final_code_v1\yolov5-master"
# sys.path.append(yolov5_path)

# # ================= WINDOWS FIX =================
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# # ================= ROOT =================
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))

# # ================= YOLO =================
# from models.common import DetectMultiBackend
# from utils.dataloaders import LoadImages, LoadStreams
# from utils.general import check_img_size, non_max_suppression, scale_boxes
# from utils.segment.general import process_mask
# from utils.torch_utils import select_device, smart_inference_mode
# from sort.sort import Sort

# # ================= OCR INIT =================
# # PaddleOCR: use_angle_cls helps with rotated/slanted text on wagons
# ocr_reader = PaddleOCR(
#     lang="en",
#     device="gpu",  # change to "cpu" if no GPU
#     use_textline_orientation=True
# )

# # ================= CONFIG =================
# LINE_RATIO   = 0.3
# BUFFER_RATIO = 0.06
# ocr_temp = {}
# OCR_CLASS_NAME = None
# MIN_OCR_FRAMES = 3
# BBOX_PAD_RATIO = 0.15
# DEBUG_CROPS    = True
# STOP_REQUESTED = False

# # ================= SIGNAL =================
# def request_stop(sig=None, frame=None):
#     global STOP_REQUESTED
#     STOP_REQUESTED = True
#     print("\n⚠ Stopping safely...")
# signal.signal(signal.SIGINT, request_stop)

# # ================= OCR CLASS AUTO-DETECT =================
# def detect_ocr_class(names, override=None):
#     all_cls = list(names.values())
#     print(f"ℹ  Model classes: {all_cls}")
#     if override:
#         if override in all_cls:
#             print(f"✅ OCR class (override): '{override}'")
#             return override
#         print(f"⚠  --ocr-class '{override}' not in model. Available: {all_cls}")
#     for kw in ["wagon_id", "wagon_number", "number", "id"]:
#         match = [c for c in all_cls if kw in c.lower()]
#         if match:
#             print(f"✅ OCR class (auto '{kw}'): '{match[0]}'")
#             return match[0]
#     print(f"⚠  No OCR class found. OCR disabled.")
#     return None

# # ================= TEXT NORMALISATION =================
# def normalise_ocr_text(text):
#     text = text.upper()
#     for src, dst in [("O","0"),("I","1"),("L","1"),("S","5"),
#                      ("B","8"),("Z","2"),("G","6"),("T","1"),("Q","0")]:
#         text = text.replace(src, dst)
#     return re.sub(r'[^0-9]', '', text)

# def extract_valid_number(raw_text):
#     digits = normalise_ocr_text(raw_text)
#     m = re.findall(r'\d{11}', digits)
#     if m:
#         return m[0]
#     m = re.findall(r'\d{10,12}', digits)
#     if m:
#         num = m[0]
#         if len(num) == 11: return num
#         if len(num) == 12: return num[:11]
#     return ""

# # ================= CROP & PREPROCESS =================
# def expanded_bbox(x1, y1, x2, y2, pad_ratio, frame_w, frame_h):
#     pw = int((x2 - x1) * pad_ratio)
#     ph = int((y2 - y1) * pad_ratio)
#     return (
#         max(x1 - pw, 0),
#         max(y1 - ph, 0),
#         min(x2 + pw, frame_w),
#         min(y2 + ph, frame_h),
#     )

# def is_blurry_crop(raw_crop):
#     gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
#     var  = cv2.Laplacian(gray, cv2.CV_64F).var()
#     return var < 15, var

# def make_ocr_variants(raw_crop):
#     up = cv2.resize(raw_crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
#     variants = []

#     # ===== 1. CLAHE (BEST baseline) =====
#     lab  = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
#     cl   = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
#     lab[:, :, 0] = cl.apply(lab[:, :, 0])
#     clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
#     variants.append(("CLAHE", clahe_bgr))

#     # ===== 2. SHARPEN =====
#     blur   = cv2.GaussianBlur(up, (0, 0), 3)
#     sharp  = cv2.addWeighted(up, 1.8, blur, -0.8, 0)
#     variants.append(("SHARP", sharp))

#     # ===== 3. THRESH + MORPH CLOSE =====
#     gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
#     denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
#     thresh = cv2.adaptiveThreshold(
#         denoised, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         31, 8
#     )
#     kernel = np.ones((2, 2), np.uint8)
#     morph  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#     morph3 = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
#     variants.append(("THRESH_MORPH", morph3))

#     return variants


# # ================= PaddleOCR INFERENCE =================
# def run_paddleocr_on_crop(img_bgr, label=""):
#     """
#     Run PaddleOCR on a BGR numpy image.
#     Returns: (joined_text, best_confidence, raw_results)
#     """
#     # PaddleOCR expects BGR numpy array (same as OpenCV) — no conversion needed
#     result = ocr_reader.ocr(img_bgr, cls=True)

#     # result is a list-of-pages; each page is a list of
#     # [[box_points], (text, confidence)]  OR  None
#     if not result or result[0] is None:
#         return "", 0.0, []

#     kept = []
#     for line in result[0]:
#         if line is None:
#             continue
#         text, conf = line[1]
#         text = text.strip()
#         if conf > 0.3 and text:
#             kept.append((text, conf))

#     if not kept:
#         return "", 0.0, result[0]

#     joined    = "".join(t for t, c in kept)
#     best_conf = max(c for t, c in kept)
#     return joined, best_conf, result[0]


# def extract_text_paddleocr(raw_crop, debug_dir=None, tid=None, frame_no=None):
#     """
#     Identical contract to the old extract_text_easyocr():
#     returns a cleaned 11-digit string or "".
#     """
#     if raw_crop is None or raw_crop.size == 0:
#         return ""

#     blurry, blur_var = is_blurry_crop(raw_crop)
#     if blurry:
#         print(f"      [OCR] skipped blurry raw crop (var={blur_var:.1f})")
#         return ""

#     print(f"      [OCR] trying  blur_var={blur_var:.1f}")

#     variants = make_ocr_variants(raw_crop)
#     best_number = ""
#     best_conf   = 0.0

#     for label, img in variants:
#         if debug_dir and tid is not None:
#             vpath = debug_dir / f"tid{tid}_f{frame_no}_{label}.jpg"
#             cv2.imwrite(str(vpath), img)

#         raw_text, conf, results = run_paddleocr_on_crop(img, label)

#         if results:
#             for line in results:
#                 if line is None:
#                     continue
#                 txt, c = line[1]
#                 print(f"      [OCR {label}] '{txt}'  conf={c:.2f}")

#         number = extract_valid_number(raw_text)
#         if number and conf > best_conf:
#             best_number = number
#             best_conf   = conf
#             print(f"      [OCR {label}] ✔ candidate: {number}  (conf={conf:.2f})")

#     if best_number:
#         print(f"      [OCR] BEST: {best_number}  conf={best_conf:.2f}")
#     else:
#         print(f"      [OCR] no valid 11-digit number found across all strategies")

#     return best_number


# # ================= COUNTING LOGIC =================
# def get_side(pos, line_pos, buffer_px, axis):
#     lo, hi = line_pos - buffer_px, line_pos + buffer_px
#     if axis == "x":
#         if pos < lo: return "left"
#         if pos > hi: return "right"
#     else:
#         if pos < lo: return "top"
#         if pos > hi: return "bottom"
#     return "buffer"

# def crossed_line(prev_side, curr_side, axis):
#     if prev_side is None or prev_side == "buffer" or curr_side == "buffer":
#         return None
#     if prev_side == curr_side:
#         return None
#     if axis == "x":
#         if prev_side == "right" and curr_side == "left":  return "out"
#     else:
#         if prev_side == "bottom" and curr_side == "top":  return "out"
#     return None


# def draw_ocr_panel(frame, ocr_results, ocr_candidates, ocr_done, names, OCR_CLASS_NAME):
#     h, w = frame.shape[:2]

#     FONT    = cv2.FONT_HERSHEY_SIMPLEX
#     PAD_X   = 12
#     PAD_TOP = 10
#     ROW_H   = 36
#     HDR_H   = 26
#     PANEL_W = 280

#     entries = [(tid, num) for tid, num in ocr_results.items()]

#     if not entries:
#         return

#     PANEL_H = min(PAD_TOP + HDR_H + 4 + ROW_H * len(entries) + PAD_TOP, h)
#     px = w - PANEL_W

#     ov = frame.copy()
#     cv2.rectangle(ov, (px, 0), (w, PANEL_H), (15, 15, 15), -1)
#     cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

#     hy = PAD_TOP + HDR_H - 4
#     cv2.putText(frame, "CONFIRMED WAGONS", (px + PAD_X, hy),
#                 FONT, 0.45, (0, 255, 100), 1, cv2.LINE_AA)

#     div_y = PAD_TOP + HDR_H + 2
#     cv2.line(frame, (px + PAD_X, div_y), (w - PAD_X, div_y), (55, 55, 55), 1)

#     for idx, (tid, num) in enumerate(entries):
#         ry = div_y + 6 + ROW_H * idx
#         if ry + ROW_H > h:
#             break

#         if idx % 2 == 0:
#             ro = frame.copy()
#             cv2.rectangle(ro, (px, ry), (w, ry + ROW_H), (30, 30, 30), -1)
#             cv2.addWeighted(ro, 0.35, frame, 0.65, 0, frame)

#         cv2.putText(frame, num, (px + PAD_X, ry + 32),
#                     FONT, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
#         cv2.putText(frame, "Detected", (w - PAD_X - 18, ry + 20),
#                     FONT, 0.45, (0, 200, 80), 1, cv2.LINE_AA)

#     cv2.rectangle(frame, (px, 0), (w - 1, PANEL_H), (70, 70, 70), 1)


# # ================= UTIL =================
# def get_class_color(cls):
#     np.random.seed(abs(hash(cls)) % (2**32))
#     return tuple(int(c) for c in np.random.randint(40, 255, 3))

# def make_save_dir(project, name):
#     base     = Path(project) / name
#     save_dir = base
#     i = 2
#     while save_dir.exists():
#         save_dir = Path(f"{base}{i}")
#         i += 1
#     save_dir.mkdir(parents=True, exist_ok=True)
#     return save_dir

# # ================= DRAWING =================
# def draw_counting_line(frame, axis, line_pos, buffer_px):
#     h, w = frame.shape[:2]
#     if axis == "x":
#         lx  = int(np.clip(line_pos,             0, w - 1))
#         blo = int(np.clip(line_pos - buffer_px, 0, w - 1))
#         bhi = int(np.clip(line_pos + buffer_px, 0, w - 1))
#         cv2.line(frame, (lx, 0), (lx, h - 1), (0, 30, 255), 3)
#         for y in range(0, h, 20):
#             y2 = min(y + 10, h - 1)
#             cv2.line(frame, (blo, y), (blo, y2), (80, 80, 180), 1)
#             cv2.line(frame, (bhi, y), (bhi, y2), (80, 80, 180), 1)
#         cv2.putText(frame, "COUNT LINE", (min(lx + 6, w - 120), 22),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 30, 255), 1, cv2.LINE_AA)
#     else:
#         ly  = int(np.clip(line_pos,             0, h - 1))
#         blo = int(np.clip(line_pos - buffer_px, 0, h - 1))
#         bhi = int(np.clip(line_pos + buffer_px, 0, h - 1))
#         cv2.line(frame, (0, ly), (w - 1, ly), (0, 30, 255), 3)
#         for x in range(0, w, 20):
#             x2 = min(x + 10, w - 1)
#             cv2.line(frame, (x, blo), (x2, blo), (80, 80, 180), 1)
#             cv2.line(frame, (x, bhi), (x2, bhi), (80, 80, 180), 1)
#         cv2.putText(frame, "COUNT LINE", (10, max(ly - 8, 16)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 30, 255), 1, cv2.LINE_AA)


# def draw_count_hud(frame, count_out, class_order):
#     FONT     = cv2.FONT_HERSHEY_SIMPLEX
#     PAD_X    = 12
#     PAD_TOP  = 10
#     ROW_H    = 30
#     HDR_H    = 26
#     COL_NAME = PAD_X
#     COL_OUT  = PAD_X + 175
#     PANEL_W  = COL_OUT + 60
#     PANEL_H  = PAD_TOP + HDR_H + 4 + ROW_H * len(class_order) + PAD_TOP

#     ov = frame.copy()
#     cv2.rectangle(ov, (0, 0), (PANEL_W, PANEL_H), (15, 15, 15), -1)
#     cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

#     hy = PAD_TOP + HDR_H - 4
#     cv2.putText(frame, "CLASS", (COL_NAME, hy), FONT, 0.46, (160,160,160), 1, cv2.LINE_AA)
#     cv2.putText(frame, "Count",   (COL_OUT,  hy), FONT, 0.46, (160,160,160), 1, cv2.LINE_AA)

#     div_y = PAD_TOP + HDR_H + 2
#     cv2.line(frame, (PAD_X, div_y), (PANEL_W - PAD_X, div_y), (55, 55, 55), 1)

#     for idx, cls in enumerate(class_order):
#         ry = div_y + 6 + ROW_H * idx + 18
#         cls_color = get_class_color(cls)
#         if idx % 2 == 0:
#             ro = frame.copy()
#             y0, y1 = div_y + 6 + ROW_H * idx, div_y + 6 + ROW_H * (idx + 1)
#             cv2.rectangle(ro, (0, y0), (PANEL_W, y1), (30, 30, 30), -1)
#             cv2.addWeighted(ro, 0.35, frame, 0.65, 0, frame)
#         cv2.putText(frame, cls,
#                     (COL_NAME, ry), FONT, 0.50, cls_color, 1, cv2.LINE_AA)
#         cv2.putText(frame, str(count_out.get(cls, 0)),
#                     (COL_OUT, ry), FONT, 0.52, (80, 150, 255), 1, cv2.LINE_AA)

#     cv2.rectangle(frame, (0, 0), (PANEL_W, PANEL_H), (70, 70, 70), 1)


# def draw_wagon_label_inside_mask(frame, mask_bool, tracker_id,
#                                   wagon_number=None, ocr_status=None):
#     ys, xs = np.where(mask_bool)
#     if len(xs) == 0:
#         return

#     cx, cy = int(xs.mean()), int(ys.mean())
#     FONT   = cv2.FONT_HERSHEY_SIMPLEX
#     SO     = 1

#     id_label = f"ID:{tracker_id}"
#     (tw, th), _ = cv2.getTextSize(id_label, FONT, 0.6, 2)
#     tx = cx - tw // 2
#     ty = (cy - 22) if wagon_number else ((cy - 10) if ocr_status else (cy + th // 2))

#     cv2.putText(frame, id_label, (tx+SO, ty+SO), FONT, 0.6, (0,0,0),       2, cv2.LINE_AA)
#     cv2.putText(frame, id_label, (tx,    ty),    FONT, 0.6, (255,255,255),  2, cv2.LINE_AA)

#     if ocr_status and not wagon_number:
#         (sw, sh), _ = cv2.getTextSize(ocr_status, FONT, 0.45, 1)
#         sx, sy = cx - sw // 2, ty + sh + 6
#         cv2.putText(frame, ocr_status, (sx+SO, sy+SO), FONT, 0.45, (0,0,0),    1, cv2.LINE_AA)
#         cv2.putText(frame, ocr_status, (sx,    sy),    FONT, 0.45, (0,220,255), 1, cv2.LINE_AA)

#     if wagon_number:
#         (nw, nh), _ = cv2.getTextSize(wagon_number, FONT, 0.65, 2)
#         nx, ny = cx - nw // 2, ty + nh + 8
#         cv2.putText(frame, wagon_number, (nx+SO, ny+SO), FONT, 0.65, (0,0,0),     2, cv2.LINE_AA)
#         cv2.putText(frame, wagon_number, (nx,    ny),    FONT, 0.65, (0,255,255),  2, cv2.LINE_AA)


# # ================= SAVE HELPERS =================
# def init_csv(save_dir):
#     csv_path = save_dir / "ocr_results.csv"
#     f = open(csv_path, "w", newline="", encoding="utf-8")
#     w = csv.writer(f)
#     w.writerow(["timestamp", "tracker_id", "wagon_number", "frame_no"])
#     print(f"📄 OCR CSV  → {csv_path}")
#     return f, w

# def save_summary_json(save_dir, count_out, ocr_results):
#     data = {
#         "run_time":  datetime.now().isoformat(),
#         "count_out": count_out,
#         "ocr_wagon_numbers": {str(tid): num for tid, num in ocr_results.items()}
#     }
#     out = save_dir / "summary.json"
#     with open(out, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2)
#     print(f"📊 Summary  → {out}")


# # ==================================================
# @smart_inference_mode()
# def run(weights, source, imgsz=640, conf_thres=0.25, iou_thres=0.45,
#         device="", project="runs", name="exp", axis="x", ocr_class=None):

#     save_dir  = make_save_dir(project, name)
#     debug_dir = save_dir / "ocr_debug_crops"
#     debug_dir.mkdir(exist_ok=True)
#     print(f"💾 Saving to: {save_dir}")

#     device = select_device(device)
#     model  = DetectMultiBackend(weights, device=device)
#     stride, names = model.stride, model.names
#     imgsz  = check_img_size(imgsz, s=stride)
#     class_order = list(names.values())

#     global OCR_CLASS_NAME
#     OCR_CLASS_NAME = detect_ocr_class(names, override=ocr_class)

#     dataset = LoadStreams(source) if source.isnumeric() else LoadImages(source)

#     vid_writer = None
#     out_video  = save_dir / "output.mp4"

#     tracker   = Sort()
#     count_out = {v: 0 for v in names.values()}

#     last_confirmed_side = {}

#     ocr_candidates  = {}
#     ocr_done        = {}
#     ocr_results     = {}
#     ocr_attempt_no  = {}

#     csv_file, csv_writer = init_csv(save_dir)

#     frame_no  = 0
#     line_pos  = None
#     buffer_px = None

#     for data in tqdm(dataset):

#         if STOP_REQUESTED:
#             break

#         path, im, im0s, vid_cap, _ = data
#         raw   = im0s[0] if isinstance(im0s, list) else im0s
#         frame = raw.copy()
#         frame_no += 1

#         h, w = frame.shape[:2]

#         if axis == "x":
#             line_pos  = int(w * LINE_RATIO)
#             buffer_px = int(w * BUFFER_RATIO)
#         else:
#             line_pos  = int(h * LINE_RATIO)
#             buffer_px = int(h * BUFFER_RATIO)

#         if vid_writer is None:
#             fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
#             fps = fps if fps > 0 else 25
#             fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
#             vid_writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))
#             print(f"🎬 Video → {out_video}  ({w}×{h} @ {fps:.1f} fps)")

#         im_t = torch.from_numpy(im).to(device).float() / 255.0
#         if im_t.ndim == 3:
#             im_t = im_t[None]

#         out_m = model(im_t)
#         if isinstance(out_m, (list, tuple)):
#             pred, proto = out_m[0], out_m[1]
#         else:
#             raise ValueError("Invalid model output")
#         if proto is None:
#             raise ValueError("Segmentation model required")

#         pred = non_max_suppression(pred, conf_thres, iou_thres, nm=32)
#         detections = []

#         if pred[0] is not None:
#             pred[0][:, :4] = scale_boxes(im_t.shape[2:], pred[0][:, :4], frame.shape).round()
#             if isinstance(proto, list): proto = proto[0]
#             if not isinstance(proto, torch.Tensor):
#                 raise TypeError(f"Proto must be tensor, got {type(proto)}")
#             if proto.ndim == 4: proto = proto[0]
#             if proto.ndim != 3:
#                 raise ValueError(f"Unexpected proto shape: {proto.shape}")

#             masks = process_mask(
#                 proto, pred[0][:, 6:], pred[0][:, :4],
#                 frame.shape[:2], upsample=True
#             ).cpu().numpy()

#             for i, (*xyxy, conf, cls) in enumerate(pred[0][:, :6]):
#                 x1, y1, x2, y2 = map(int, xyxy)
#                 detections.append([x1, y1, x2, y2, conf.item(), int(cls), masks[i]])

#         tracks = tracker.update(
#             np.array([d[:5] for d in detections]) if detections else np.empty((0, 5))
#         )

#         for trk in tracks.astype(int):
#             x1, y1, x2, y2, tid = trk

#             det = next((d for d in detections if abs(x1 - d[0]) < 20), None)
#             if det is None:
#                 continue

#             cls_name  = names[det[5]]
#             mask      = det[6]
#             mask_bool = mask > 0.5

#             ys, xs = np.where(mask_bool)
#             if len(xs) == 0:
#                 continue

#             cx, cy = int(xs.mean()), int(ys.mean())
#             pos    = cx if axis == "x" else cy

#             # ---- line crossing ----
#             curr_side = get_side(pos, line_pos, buffer_px, axis)
#             prev_side = last_confirmed_side.get(tid)
#             direction = crossed_line(prev_side, curr_side, axis)

#             if direction == "out":
#                 count_out[cls_name] += 1
#                 print(f"⬅  OUT | {cls_name} | TID:{tid} | frame:{frame_no}")

#                 if tid in ocr_temp and tid not in ocr_results:
#                     final = ocr_temp[tid]
#                     ocr_results[tid] = final

#                     ts = datetime.now().isoformat()
#                     csv_writer.writerow([ts, tid, final, frame_no])
#                     csv_file.flush()

#                     print(f"🔥 FINAL CONFIRMED AFTER CROSSING TID:{tid} → {final}")
#             if curr_side != "buffer":
#                 last_confirmed_side[tid] = curr_side

#             # ---- OCR BLOCK ----
#             ocr_status = None

#             if OCR_CLASS_NAME and cls_name == OCR_CLASS_NAME and tid not in ocr_done:

#                 n_so_far  = len(ocr_candidates.get(tid, []))
#                 ocr_status = f"scanning {n_so_far}/{MIN_OCR_FRAMES}"

#                 ex1, ey1, ex2, ey2 = expanded_bbox(x1, y1, x2, y2, BBOX_PAD_RATIO, w, h)
#                 raw_crop = raw[ey1:ey2, ex1:ex2].copy()

#                 attempt = ocr_attempt_no.get(tid, 0) + 1
#                 ocr_attempt_no[tid] = attempt
#                 if DEBUG_CROPS and attempt <= 8:
#                     cv2.imwrite(
#                         str(debug_dir / f"tid{tid}_f{frame_no}_raw.jpg"),
#                         raw_crop
#                     )

#                 # ✅ PaddleOCR replaces EasyOCR here
#                 number = extract_text_paddleocr(
#                     raw_crop,
#                     debug_dir=debug_dir if (DEBUG_CROPS and attempt <= 4) else None,
#                     tid=tid,
#                     frame_no=frame_no,
#                 )

#                 if number:
#                     ocr_candidates.setdefault(tid, []).append(number)
#                     n = len(ocr_candidates[tid])
#                     ocr_status = f"reading {n}/{MIN_OCR_FRAMES}"
#                     print(f"   ✔ OCR TID:{tid} [{n}/{MIN_OCR_FRAMES}] → {number}")

#                     if n >= MIN_OCR_FRAMES:
#                         most_common, count = Counter(ocr_candidates[tid]).most_common(1)[0]
#                         if count >= 2:
#                             final = most_common
#                             ocr_temp[tid] = final
#                             ocr_done[tid] = True
#                             print(f"🕒 OCR READY (waiting for crossing) TID:{tid} → {final}")
#                             ocr_status       = None
#                             ts = datetime.now().isoformat()
#                             csv_writer.writerow([ts, tid, final, frame_no])
#                             csv_file.flush()
#                             print(f"🔥 CONFIRMED TID:{tid} → {final}  (frame {frame_no})")

#                             cv2.imwrite(
#                                 str(debug_dir / f"CONFIRMED_tid{tid}_{final}.jpg"),
#                                 raw_crop
#                             )
#                         else:
#                             ocr_status = f"no consensus ({n} reads)"
#                             print(f"   ⚠ No consensus yet TID:{tid} — {ocr_candidates[tid]}")

#             # ---- draw mask overlay ----
#             color = get_class_color(cls_name)
#             frame[mask_bool] = frame[mask_bool] * 0.5 + np.array(color) * 0.5

#             cv2.putText(frame, cls_name, (x1, max(y1 - 5, 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

#             draw_wagon_label_inside_mask(
#                 frame, mask_bool, tid,
#                 wagon_number=ocr_results.get(tid),
#                 ocr_status=ocr_status,
#             )

#         draw_counting_line(frame, axis, line_pos, buffer_px)
#         draw_count_hud(frame, count_out, class_order)
#         draw_ocr_panel(frame, ocr_results, ocr_candidates, ocr_done, names, OCR_CLASS_NAME)

#         vid_writer.write(frame)
#         cv2.imshow("FINAL SYSTEM", frame)
#         if cv2.waitKey(1) == 27:
#             break

#     if vid_writer:
#         vid_writer.release()
#     csv_file.close()
#     cv2.destroyAllWindows()
#     save_summary_json(save_dir, count_out, ocr_results)

#     print("\n" + "=" * 60)
#     print(f"  Run complete → {save_dir}")
#     print(f"  ├─ output.mp4              annotated video")
#     print(f"  ├─ ocr_results.csv         wagon numbers + timestamps")
#     print(f"  ├─ summary.json            counts + wagon-number map")
#     print(f"  └─ ocr_debug_crops/        raw crops + all variants per attempt")
#     print("=" * 60)


# # ================= CLI =================
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights",    required=True)
#     parser.add_argument("--source",     required=True)
#     parser.add_argument("--axis",       choices=["x", "y"], default="x")
#     parser.add_argument("--imgsz",      type=int,   default=640)
#     parser.add_argument("--conf-thres", type=float, default=0.25)
#     parser.add_argument("--iou-thres",  type=float, default=0.45)
#     parser.add_argument("--device",     default="")
#     parser.add_argument("--project",    default="runs")
#     parser.add_argument("--name",       default="exp")
#     parser.add_argument("--ocr-class",  default=None,
#                         help="Class to OCR. Auto-detected if not set.")
#     return parser.parse_args()

# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))









































































































# import argparse
# import sys
# import csv
# import json
# from datetime import datetime
# from pathlib import Path
# import cv2
# import torch
# import numpy as np
# import pathlib
# from tqdm import tqdm
# import signal
# import re
# from collections import Counter
# import os
# os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
# os.environ["FLAGS_use_mkldnn"] = "0"          # ← disables oneDNN (fixes the crash)
# os.environ["FLAGS_mkldnn_disable"] = "1"       # ← belt-and-suspenders
# # ================= PaddleOCR =================
# from paddleocr import PaddleOCR

# import sys
# import os
# import sys
# sys.path.insert(0, r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Manfucturing_sector\TATA_steel\enginnering\final_code_v1\yolov5-master")  # YOLOv5 utils first
# sys.path.insert(1, r"C:\Users\admin\.conda\envs\paddle_env\Lib\site-packages\paddleocr")  # PaddleOCR utils second
# yolov5_path = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Manfucturing_sector\TATA_steel\enginnering\final_code_v1\yolov5-master"
# sys.path.append(yolov5_path)

# # ================= WINDOWS FIX =================
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# # ================= ROOT =================
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))

# # ================= YOLO =================
# from models.common import DetectMultiBackend
# from utils.dataloaders import LoadImages, LoadStreams
# from utils.general import check_img_size, non_max_suppression, scale_boxes
# from utils.segment.general import process_mask
# from utils.torch_utils import select_device, smart_inference_mode
# from sort.sort import Sort

# # ================= OCR INIT =================
# # PaddleOCR v3+ API — deprecated params removed:
# #   use_angle_cls  → replaced by use_textline_orientation=True
# #   show_log       → removed (suppress logs via env var or logging module)
# #   use_gpu        → removed (device selection now via CUDA_VISIBLE_DEVICES)
# import logging
# logging.getLogger("ppocr").setLevel(logging.ERROR)   # suppress PaddleOCR verbose logs
# import os
# os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"  # skip slow connectivity check

# ocr_reader = PaddleOCR(
#     lang="en",
#     use_angle_cls=True,
#     use_gpu=False,
#     show_log=False
# )
# # ================= CONFIG =================
# LINE_RATIO   = 0.4
# BUFFER_RATIO = 0.06
# ocr_temp     = {}
# OCR_CLASS_NAME  = None
# MIN_OCR_FRAMES  = 3
# BBOX_PAD_RATIO  = 0.15
# DEBUG_CROPS     = True
# STOP_REQUESTED  = False

# # ================= SIGNAL =================
# def request_stop(sig=None, frame=None):
#     global STOP_REQUESTED
#     STOP_REQUESTED = True
#     print("\n⚠ Stopping safely...")
# signal.signal(signal.SIGINT, request_stop)

# # ================= OCR CLASS AUTO-DETECT =================
# def detect_ocr_class(names, override=None):
#     all_cls = list(names.values())
#     print(f"ℹ  Model classes: {all_cls}")
#     if override:
#         if override in all_cls:
#             print(f"✅ OCR class (override): '{override}'")
#             return override
#         print(f"⚠  --ocr-class '{override}' not in model. Available: {all_cls}")
#     for kw in ["wagon_id", "wagon_number", "number", "id"]:
#         match = [c for c in all_cls if kw in c.lower()]
#         if match:
#             print(f"✅ OCR class (auto '{kw}'): '{match[0]}'")
#             return match[0]
#     print(f"⚠  No OCR class found. OCR disabled.")
#     return None

# # ================= TEXT NORMALISATION =================
# def normalise_ocr_text(text):
#     text = text.upper()
#     for src, dst in [("O","0"),("I","1"),("L","1"),("S","5"),
#                      ("B","8"),("Z","2"),("G","6"),("T","1"),("Q","0")]:
#         text = text.replace(src, dst)
#     return re.sub(r'[^0-9]', '', text)

# def extract_valid_number(raw_text):
#     digits = normalise_ocr_text(raw_text)
#     m = re.findall(r'\d{11}', digits)
#     if m:
#         return m[0]
#     m = re.findall(r'\d{10,12}', digits)
#     if m:
#         num = m[0]
#         if len(num) == 11: return num
#         if len(num) == 12: return num[:11]
#     return ""

# # ================= CROP & PREPROCESS =================
# def expanded_bbox(x1, y1, x2, y2, pad_ratio, frame_w, frame_h):
#     pw = int((x2 - x1) * pad_ratio)
#     ph = int((y2 - y1) * pad_ratio)
#     return (
#         max(x1 - pw, 0),
#         max(y1 - ph, 0),
#         min(x2 + pw, frame_w),
#         min(y2 + ph, frame_h),
#     )

# def is_blurry_crop(raw_crop):
#     gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
#     var  = cv2.Laplacian(gray, cv2.CV_64F).var()
#     return var < 15, var

# def make_ocr_variants(raw_crop):
#     up = cv2.resize(raw_crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
#     variants = []

#     # 1. CLAHE (best baseline)
#     lab  = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
#     cl   = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
#     lab[:, :, 0] = cl.apply(lab[:, :, 0])
#     clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
#     variants.append(("CLAHE", clahe_bgr))

#     # 2. SHARPEN
#     blur  = cv2.GaussianBlur(up, (0, 0), 3)
#     sharp = cv2.addWeighted(up, 1.8, blur, -0.8, 0)
#     variants.append(("SHARP", sharp))

#     # 3. THRESH + MORPH CLOSE
#     gray     = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
#     denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
#     thresh   = cv2.adaptiveThreshold(
#         denoised, 255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY,
#         31, 8
#     )
#     kernel = np.ones((2, 2), np.uint8)
#     morph  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
#     morph3 = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
#     variants.append(("THRESH_MORPH", morph3))

#     return variants


# def run_paddleocr_on_crop(img_bgr, label=""):
#     results = ocr_reader.ocr(img_bgr, cls=True)

#     if not results or not results[0]:
#         return "", 0.0, []

#     texts = []
#     for line in results[0]:
#         txt = line[1][0]
#         conf = line[1][1]
#         texts.append((txt, conf))

#     kept = [(t.strip(), c) for t, c in texts if t.strip() and c > 0.3]

#     if not kept:
#         return "", 0.0, texts

#     joined = "".join(t for t, _ in kept)
#     best_conf = max(c for _, c in kept)

#     return joined, best_conf, texts

# def extract_text_paddleocr(raw_crop, debug_dir=None, tid=None, frame_no=None):
#     if raw_crop is None or raw_crop.size == 0:
#         return ""

#     blurry, blur_var = is_blurry_crop(raw_crop)
#     if blurry:
#         print(f"      [OCR] skipped blurry raw crop (var={blur_var:.1f})")
#         return ""

#     print(f"      [OCR] trying  blur_var={blur_var:.1f}")

#     # ✅ Save exactly what goes into PaddleOCR (for debugging)
#     if debug_dir and tid is not None:
#         cv2.imwrite(str(debug_dir / f"tid{tid}_f{frame_no}_TO_OCR.jpg"), raw_crop)

#     raw_text, conf, results = run_paddleocr_on_crop(raw_crop, label="RAW")

#     if results:
#         for txt, c in results:
#             print(f"      [OCR RAW] '{txt}'  conf={c:.2f}")

#     number = extract_valid_number(raw_text)

#     if number:
#         print(f"      [OCR] BEST: {number}  conf={conf:.2f}")
#     else:
#         print(f"      [OCR] no valid 11-digit number found")

#     return number

# # ================= COUNTING LOGIC =================
# def get_side(pos, line_pos, buffer_px, axis):
#     lo, hi = line_pos - buffer_px, line_pos + buffer_px
#     if axis == "x":
#         if pos < lo: return "left"
#         if pos > hi: return "right"
#     else:
#         if pos < lo: return "top"
#         if pos > hi: return "bottom"
#     return "buffer"

# def crossed_line(prev_side, curr_side, axis):
#     if prev_side is None or prev_side == "buffer" or curr_side == "buffer":
#         return None
#     if prev_side == curr_side:
#         return None
#     if axis == "x":
#         if prev_side == "right" and curr_side == "left": return "out"
#     else:
#         if prev_side == "bottom" and curr_side == "top": return "out"
#     return None


# def draw_ocr_panel(frame, ocr_results, ocr_candidates, ocr_done, names, OCR_CLASS_NAME):
#     h, w = frame.shape[:2]

#     FONT    = cv2.FONT_HERSHEY_SIMPLEX
#     PAD_X   = 12
#     PAD_TOP = 10
#     ROW_H   = 36
#     HDR_H   = 26
#     PANEL_W = 280

#     entries = [(tid, num) for tid, num in ocr_results.items()]
#     if not entries:
#         return

#     PANEL_H = min(PAD_TOP + HDR_H + 4 + ROW_H * len(entries) + PAD_TOP, h)
#     px = w - PANEL_W

#     ov = frame.copy()
#     cv2.rectangle(ov, (px, 0), (w, PANEL_H), (15, 15, 15), -1)
#     cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

#     hy = PAD_TOP + HDR_H - 4
#     cv2.putText(frame, "CONFIRMED WAGONS", (px + PAD_X, hy),
#                 FONT, 0.45, (0, 255, 100), 1, cv2.LINE_AA)

#     div_y = PAD_TOP + HDR_H + 2
#     cv2.line(frame, (px + PAD_X, div_y), (w - PAD_X, div_y), (55, 55, 55), 1)

#     for idx, (tid, num) in enumerate(entries):
#         ry = div_y + 6 + ROW_H * idx
#         if ry + ROW_H > h:
#             break

#         if idx % 2 == 0:
#             ro = frame.copy()
#             cv2.rectangle(ro, (px, ry), (w, ry + ROW_H), (30, 30, 30), -1)
#             cv2.addWeighted(ro, 0.35, frame, 0.65, 0, frame)

#         cv2.putText(frame, num, (px + PAD_X, ry + 32),
#                     FONT, 0.52, (0, 255, 255), 1, cv2.LINE_AA)
#         cv2.putText(frame, "Detected", (w - PAD_X - 18, ry + 20),
#                     FONT, 0.45, (0, 200, 80), 1, cv2.LINE_AA)

#     cv2.rectangle(frame, (px, 0), (w - 1, PANEL_H), (70, 70, 70), 1)

# # ================= UTIL =================
# def get_class_color(cls):
#     np.random.seed(abs(hash(cls)) % (2**32))
#     return tuple(int(c) for c in np.random.randint(40, 255, 3))

# def make_save_dir(project, name):
#     base     = Path(project) / name
#     save_dir = base
#     i = 2
#     while save_dir.exists():
#         save_dir = Path(f"{base}{i}")
#         i += 1
#     save_dir.mkdir(parents=True, exist_ok=True)
#     return save_dir

# # ================= DRAWING =================
# def draw_counting_line(frame, axis, line_pos, buffer_px):
#     h, w = frame.shape[:2]
#     if axis == "x":
#         lx  = int(np.clip(line_pos,             0, w - 1))
#         blo = int(np.clip(line_pos - buffer_px, 0, w - 1))
#         bhi = int(np.clip(line_pos + buffer_px, 0, w - 1))
#         cv2.line(frame, (lx, 0), (lx, h - 1), (0, 30, 255), 3)
#         for y in range(0, h, 20):
#             y2 = min(y + 10, h - 1)
#             cv2.line(frame, (blo, y), (blo, y2), (80, 80, 180), 1)
#             cv2.line(frame, (bhi, y), (bhi, y2), (80, 80, 180), 1)
#         cv2.putText(frame, "COUNT LINE", (min(lx + 6, w - 120), 22),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 30, 255), 1, cv2.LINE_AA)
#     else:
#         ly  = int(np.clip(line_pos,             0, h - 1))
#         blo = int(np.clip(line_pos - buffer_px, 0, h - 1))
#         bhi = int(np.clip(line_pos + buffer_px, 0, h - 1))
#         cv2.line(frame, (0, ly), (w - 1, ly), (0, 30, 255), 3)
#         for x in range(0, w, 20):
#             x2 = min(x + 10, w - 1)
#             cv2.line(frame, (x, blo), (x2, blo), (80, 80, 180), 1)
#             cv2.line(frame, (x, bhi), (x2, bhi), (80, 80, 180), 1)
#         cv2.putText(frame, "COUNT LINE", (10, max(ly - 8, 16)),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 30, 255), 1, cv2.LINE_AA)


# def draw_count_hud(frame, count_out, class_order):
#     FONT     = cv2.FONT_HERSHEY_SIMPLEX
#     PAD_X    = 12
#     PAD_TOP  = 10
#     ROW_H    = 30
#     HDR_H    = 26
#     COL_NAME = PAD_X
#     COL_OUT  = PAD_X + 175
#     PANEL_W  = COL_OUT + 60
#     PANEL_H  = PAD_TOP + HDR_H + 4 + ROW_H * len(class_order) + PAD_TOP

#     ov = frame.copy()
#     cv2.rectangle(ov, (0, 0), (PANEL_W, PANEL_H), (15, 15, 15), -1)
#     cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

#     hy = PAD_TOP + HDR_H - 4
#     cv2.putText(frame, "CLASS", (COL_NAME, hy), FONT, 0.46, (160,160,160), 1, cv2.LINE_AA)
#     cv2.putText(frame, "Count", (COL_OUT,  hy), FONT, 0.46, (160,160,160), 1, cv2.LINE_AA)

#     div_y = PAD_TOP + HDR_H + 2
#     cv2.line(frame, (PAD_X, div_y), (PANEL_W - PAD_X, div_y), (55, 55, 55), 1)

#     for idx, cls in enumerate(class_order):
#         ry = div_y + 6 + ROW_H * idx + 18
#         cls_color = get_class_color(cls)
#         if idx % 2 == 0:
#             ro = frame.copy()
#             y0, y1 = div_y + 6 + ROW_H * idx, div_y + 6 + ROW_H * (idx + 1)
#             cv2.rectangle(ro, (0, y0), (PANEL_W, y1), (30, 30, 30), -1)
#             cv2.addWeighted(ro, 0.35, frame, 0.65, 0, frame)
#         cv2.putText(frame, cls,
#                     (COL_NAME, ry), FONT, 0.50, cls_color, 1, cv2.LINE_AA)
#         cv2.putText(frame, str(count_out.get(cls, 0)),
#                     (COL_OUT, ry), FONT, 0.52, (80, 150, 255), 1, cv2.LINE_AA)

#     cv2.rectangle(frame, (0, 0), (PANEL_W, PANEL_H), (70, 70, 70), 1)


# def draw_wagon_label_inside_mask(frame, mask_bool, tracker_id,
#                                   wagon_number=None, ocr_status=None):
#     ys, xs = np.where(mask_bool)
#     if len(xs) == 0:
#         return

#     cx, cy = int(xs.mean()), int(ys.mean())
#     FONT   = cv2.FONT_HERSHEY_SIMPLEX
#     SO     = 1

#     id_label = f"ID:{tracker_id}"
#     (tw, th), _ = cv2.getTextSize(id_label, FONT, 0.6, 2)
#     tx = cx - tw // 2
#     ty = (cy - 22) if wagon_number else ((cy - 10) if ocr_status else (cy + th // 2))

#     cv2.putText(frame, id_label, (tx+SO, ty+SO), FONT, 0.6, (0,0,0),      2, cv2.LINE_AA)
#     cv2.putText(frame, id_label, (tx,    ty),    FONT, 0.6, (255,255,255), 2, cv2.LINE_AA)

#     if ocr_status and not wagon_number:
#         (sw, sh), _ = cv2.getTextSize(ocr_status, FONT, 0.45, 1)
#         sx, sy = cx - sw // 2, ty + sh + 6
#         cv2.putText(frame, ocr_status, (sx+SO, sy+SO), FONT, 0.45, (0,0,0),    1, cv2.LINE_AA)
#         cv2.putText(frame, ocr_status, (sx,    sy),    FONT, 0.45, (0,220,255), 1, cv2.LINE_AA)

#     if wagon_number:
#         (nw, nh), _ = cv2.getTextSize(wagon_number, FONT, 0.65, 2)
#         nx, ny = cx - nw // 2, ty + nh + 8
#         cv2.putText(frame, wagon_number, (nx+SO, ny+SO), FONT, 0.65, (0,0,0),    2, cv2.LINE_AA)
#         cv2.putText(frame, wagon_number, (nx,    ny),    FONT, 0.65, (0,255,255), 2, cv2.LINE_AA)

# def find_closest_ocr(cx, cy, ocr_temp_boxes):
#     best_tid = None
#     best_dist = 99999

#     for tid, (ox, oy) in ocr_temp_boxes.items():
#         d = (cx - ox)**2 + (cy - oy)**2
#         if d < best_dist:
#             best_dist = d
#             best_tid = tid

#     return best_tid

# # ================= SAVE HELPERS =================
# def init_csv(save_dir):
#     csv_path = save_dir / "ocr_results.csv"
#     f = open(csv_path, "w", newline="", encoding="utf-8")
#     w = csv.writer(f)
#     w.writerow(["timestamp", "tracker_id", "wagon_number", "frame_no"])
#     print(f"📄 OCR CSV  → {csv_path}")
#     return f, w

# def save_summary_json(save_dir, count_out, ocr_results):
#     data = {
#         "run_time":         datetime.now().isoformat(),
#         "count_out":        count_out,
#         "ocr_wagon_numbers": {str(tid): num for tid, num in ocr_results.items()}
#     }
#     out = save_dir / "summary.json"
#     with open(out, "w", encoding="utf-8") as f:
#         json.dump(data, f, indent=2)
#     print(f"📊 Summary  → {out}")


# # ==================================================
# @smart_inference_mode()
# def run(weights, source, imgsz=640, conf_thres=0.25, iou_thres=0.45,
#         device="", project="runs", name="exp", axis="x", ocr_class=None):

#     save_dir  = make_save_dir(project, name)
#     debug_dir = save_dir / "ocr_debug_crops"
#     debug_dir.mkdir(exist_ok=True)
#     print(f"💾 Saving to: {save_dir}")

#     device = select_device(device)
#     model  = DetectMultiBackend(weights, device=device)
#     stride, names = model.stride, model.names
#     imgsz  = check_img_size(imgsz, s=stride)
#     class_order = list(names.values())

#     global OCR_CLASS_NAME
#     OCR_CLASS_NAME = detect_ocr_class(names, override=ocr_class)

#     dataset = LoadStreams(source) if source.isnumeric() else LoadImages(source)

#     vid_writer = None
#     out_video  = save_dir / "output.mp4"

#     tracker   = Sort()
#     count_out = {v: 0 for v in names.values()}

#     last_confirmed_side = {}
#     ocr_candidates      = {}
#     ocr_done            = {}
#     ocr_results         = {}
#     ocr_attempt_no      = {}

#     csv_file, csv_writer = init_csv(save_dir)

#     frame_no  = 0
#     line_pos  = None
#     buffer_px = None

#     for data in tqdm(dataset):

#         if STOP_REQUESTED:
#             break

#         path, im, im0s, vid_cap, _ = data
#         raw   = im0s[0] if isinstance(im0s, list) else im0s
#         frame = raw.copy()
#         frame_no += 1

#         h, w = frame.shape[:2]

#         if axis == "x":
#             line_pos  = int(w * LINE_RATIO)
#             buffer_px = int(w * BUFFER_RATIO)
#         else:
#             line_pos  = int(h * LINE_RATIO)
#             buffer_px = int(h * BUFFER_RATIO)

#         if vid_writer is None:
#             fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
#             fps = fps if fps > 0 else 25
#             fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
#             vid_writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))
#             print(f"🎬 Video → {out_video}  ({w}×{h} @ {fps:.1f} fps)")

#         im_t = torch.from_numpy(im).to(device).float() / 255.0
#         if im_t.ndim == 3:
#             im_t = im_t[None]

#         out_m = model(im_t)
#         if isinstance(out_m, (list, tuple)):
#             pred, proto = out_m[0], out_m[1]
#         else:
#             raise ValueError("Invalid model output")
#         if proto is None:
#             raise ValueError("Segmentation model required")

#         pred = non_max_suppression(pred, conf_thres, iou_thres, nm=32)
#         detections = []

#         if pred[0] is not None:
#             pred[0][:, :4] = scale_boxes(im_t.shape[2:], pred[0][:, :4], frame.shape).round()
#             if isinstance(proto, list): proto = proto[0]
#             if not isinstance(proto, torch.Tensor):
#                 raise TypeError(f"Proto must be tensor, got {type(proto)}")
#             if proto.ndim == 4: proto = proto[0]
#             if proto.ndim != 3:
#                 raise ValueError(f"Unexpected proto shape: {proto.shape}")

#             masks = process_mask(
#                 proto, pred[0][:, 6:], pred[0][:, :4],
#                 frame.shape[:2], upsample=True
#             ).cpu().numpy()

#             for i, (*xyxy, conf, cls) in enumerate(pred[0][:, :6]):
#                 x1, y1, x2, y2 = map(int, xyxy)
#                 detections.append([x1, y1, x2, y2, conf.item(), int(cls), masks[i]])

#         tracks = tracker.update(
#             np.array([d[:5] for d in detections]) if detections else np.empty((0, 5))
#         )

#         for trk in tracks.astype(int):
#             x1, y1, x2, y2, tid = trk

#             det = next((d for d in detections if abs(x1 - d[0]) < 20), None)
#             if det is None:
#                 continue

#             cls_name  = names[det[5]]
#             mask      = det[6]
#             mask_bool = mask > 0.5

#             ys, xs = np.where(mask_bool)
#             if len(xs) == 0:
#                 continue

#             cx, cy = int(xs.mean()), int(ys.mean())
#             pos    = cx if axis == "x" else cy

#             # ---- line crossing ----
#             curr_side = get_side(pos, line_pos, buffer_px, axis)
#             prev_side = last_confirmed_side.get(tid)
#             direction = crossed_line(prev_side, curr_side, axis)

#             if direction == "out":
#                 count_out[cls_name] += 1
#                 print(f"⬅  OUT | {cls_name} | TID:{tid} | frame:{frame_no}")

#                 matched_tid = None

#                 for t in ocr_temp:
#                     if abs(t - tid) <= 1:   # tolerance (or use spatial distance)
#                         matched_tid = t
#                         break

#                 if matched_tid is not None and tid not in ocr_results:
#                     final = ocr_temp[matched_tid]
#                     #ocr_results[tid] = final
#                     final = ocr_temp[tid]
#                     ocr_results[tid] = final
#                     ts = datetime.now().isoformat()
#                     csv_writer.writerow([ts, tid, final, frame_no])
#                     csv_file.flush()
#                     print(f"🔥 FINAL CONFIRMED AFTER CROSSING TID:{tid} → {final}")

#             if curr_side != "buffer":
#                 last_confirmed_side[tid] = curr_side

#             # ---- OCR BLOCK ----
#             ocr_status = None

#             if OCR_CLASS_NAME and cls_name == OCR_CLASS_NAME and tid not in ocr_done:

#                 n_so_far   = len(ocr_candidates.get(tid, []))
#                 ocr_status = f"scanning {n_so_far}/{MIN_OCR_FRAMES}"

#                 ex1, ey1, ex2, ey2 = expanded_bbox(x1, y1, x2, y2, BBOX_PAD_RATIO, w, h)
#                 raw_crop = raw[ey1:ey2, ex1:ex2].copy()

#                 attempt = ocr_attempt_no.get(tid, 0) + 1
#                 ocr_attempt_no[tid] = attempt
#                 if DEBUG_CROPS and attempt <= 8:
#                     cv2.imwrite(
#                         str(debug_dir / f"tid{tid}_f{frame_no}_raw.jpg"),
#                         raw_crop
#                     )

#                 # ✅ Now uses PaddleOCR instead of EasyOCR
#                 number = extract_text_paddleocr(
#                     raw_crop,
#                     debug_dir=debug_dir if (DEBUG_CROPS and attempt <= 4) else None,
#                     tid=tid,
#                     frame_no=frame_no,
#                 )

#                 if number:
#                     ocr_candidates.setdefault(tid, []).append(number)
#                     n = len(ocr_candidates[tid])
#                     ocr_status = f"reading {n}/{MIN_OCR_FRAMES}"
#                     print(f"   ✔ OCR TID:{tid} [{n}/{MIN_OCR_FRAMES}] → {number}")

#                     if n >= MIN_OCR_FRAMES:
#                         most_common, count = Counter(ocr_candidates[tid]).most_common(1)[0]
#                         if count >= 2:
#                             final = most_common
#                             ocr_temp[tid] = final
#                             ocr_results[tid] = final
#                             ocr_done[tid] = True
                            
#                             print(f"🕒 OCR READY (waiting for crossing) TID:{tid} → {final}")
#                             ocr_status = None
#                             ts = datetime.now().isoformat()
#                             csv_writer.writerow([ts, tid, final, frame_no])
#                             csv_file.flush()
#                             print(f"🔥 CONFIRMED TID:{tid} → {final}  (frame {frame_no})")

#                             cv2.imwrite(
#                                 str(debug_dir / f"CONFIRMED_tid{tid}_{final}.jpg"),
#                                 raw_crop
#                             )
#                         else:
#                             ocr_status = f"no consensus ({n} reads)"
#                             print(f"   ⚠ No consensus yet TID:{tid} — {ocr_candidates[tid]}")

#             # ---- draw mask overlay ----
#             color = get_class_color(cls_name)
#             frame[mask_bool] = frame[mask_bool] * 0.5 + np.array(color) * 0.5

#             cv2.putText(frame, cls_name, (x1, max(y1 - 5, 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

#             draw_wagon_label_inside_mask(
#                 frame, mask_bool, tid,
#                 wagon_number=ocr_results.get(tid),
#                 ocr_status=ocr_status,
#             )

#         draw_counting_line(frame, axis, line_pos, buffer_px)
#         draw_count_hud(frame, count_out, class_order)
#         draw_ocr_panel(frame, ocr_results, ocr_candidates, ocr_done, names, OCR_CLASS_NAME)

#         vid_writer.write(frame)
#         cv2.imshow("FINAL SYSTEM", frame)
#         if cv2.waitKey(1) == 27:
#             break

#     if vid_writer:
#         vid_writer.release()
#     csv_file.close()
#     cv2.destroyAllWindows()
#     save_summary_json(save_dir, count_out, ocr_results)

#     print("\n" + "=" * 60)
#     print(f"  Run complete → {save_dir}")
#     print(f"  ├─ output.mp4              annotated video")
#     print(f"  ├─ ocr_results.csv         wagon numbers + timestamps")
#     print(f"  ├─ summary.json            counts + wagon-number map")
#     print(f"  └─ ocr_debug_crops/        raw crops + all 3 variants per attempt")
#     print("=" * 60)


# # ================= CLI =================
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights",    required=True)
#     parser.add_argument("--source",     required=True)
#     parser.add_argument("--axis",       choices=["x", "y"], default="x")
#     parser.add_argument("--imgsz",      type=int,   default=640)
#     parser.add_argument("--conf-thres", type=float, default=0.25)
#     parser.add_argument("--iou-thres",  type=float, default=0.45)
#     parser.add_argument("--device",     default="")
#     parser.add_argument("--project",    default="runs")
#     parser.add_argument("--name",       default="exp")
#     parser.add_argument("--ocr-class",  default=None,
#                         help="Class to OCR. Auto-detected if not set.")
#     return parser.parse_args()

# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))



















import argparse
import sys
import csv
import json
from datetime import datetime
from pathlib import Path
import cv2
import torch
import numpy as np
import pathlib
from tqdm import tqdm
import signal
import re
from collections import Counter
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
os.environ["FLAGS_use_mkldnn"] = "0"          # ← disables oneDNN (fixes the crash)
os.environ["FLAGS_mkldnn_disable"] = "1"       # ← belt-and-suspenders
# ================= PaddleOCR =================
from paddleocr import PaddleOCR

import sys
import os
import sys
sys.path.insert(0, r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Manfucturing_sector\TATA_steel\enginnering\final_code_v1\yolov5-master")  # YOLOv5 utils first
sys.path.insert(1, r"C:\Users\admin\.conda\envs\paddle_env\Lib\site-packages\paddleocr")  # PaddleOCR utils second
yolov5_path = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Manfucturing_sector\TATA_steel\enginnering\final_code_v1\yolov5-master"
sys.path.append(yolov5_path)

# ================= WINDOWS FIX =================
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# ================= ROOT =================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

# ================= YOLO =================
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.segment.general import process_mask
from utils.torch_utils import select_device, smart_inference_mode
from sort.sort import Sort

# ================= OCR INIT =================
# PaddleOCR v3+ API — deprecated params removed:
#   use_angle_cls  → replaced by use_textline_orientation=True
#   show_log       → removed (suppress logs via env var or logging module)
#   use_gpu        → removed (device selection now via CUDA_VISIBLE_DEVICES)
import logging
logging.getLogger("ppocr").setLevel(logging.ERROR)   # suppress PaddleOCR verbose logs
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"  # skip slow connectivity check

ocr_reader = PaddleOCR(
    lang="en",
    use_angle_cls=True,
    use_gpu=False,
    show_log=False
)
# ================= CONFIG =================
LINE_RATIO   = 0.4
BUFFER_RATIO = 0.06
ocr_temp     = {}
OCR_CLASS_NAME  = None
MIN_OCR_FRAMES  = 3
BBOX_PAD_RATIO  = 0.15
DEBUG_CROPS     = True
STOP_REQUESTED  = False

# ================= SIGNAL =================
print("Signal initiated")
def request_stop(sig=None, frame=None):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n⚠ Stopping safely...")
signal.signal(signal.SIGINT, request_stop)

# ================= OCR CLASS AUTO-DETECT =================
def detect_ocr_class(names, override=None):
    all_cls = list(names.values())
    print(f"ℹ  Model classes: {all_cls}")
    if override:
        if override in all_cls:
            print(f"✅ OCR class (override): '{override}'")
            return override
        print(f"⚠  --ocr-class '{override}' not in model. Available: {all_cls}")
    for kw in ["wagon_id", "wagon_number", "number", "id"]:
        match = [c for c in all_cls if kw in c.lower()]
        if match:
            print(f"✅ OCR class (auto '{kw}'): '{match[0]}'")
            return match[0]
    print(f"⚠  No OCR class found. OCR disabled.")
    return None

# ================= TEXT NORMALISATION =================
def normalise_ocr_text(text):
    text = text.upper()
    for src, dst in [("O","0"),("I","1"),("L","1"),("S","5"),
                     ("B","8"),("Z","2"),("G","6"),("T","1"),("Q","0")]:
        text = text.replace(src, dst)
    return re.sub(r'[^0-9]', '', text)

def extract_valid_number(raw_text):
    digits = normalise_ocr_text(raw_text)
    m = re.findall(r'\d{11}', digits)
    if m:
        return m[0]
    m = re.findall(r'\d{10,12}', digits)
    if m:
        num = m[0]
        if len(num) == 11: return num
        if len(num) == 12: return num[:11]
    return ""

# ================= CROP & PREPROCESS =================
def expanded_bbox(x1, y1, x2, y2, pad_ratio, frame_w, frame_h):
    pw = int((x2 - x1) * pad_ratio)
    ph = int((y2 - y1) * pad_ratio)
    return (
        max(x1 - pw, 0),
        max(y1 - ph, 0),
        min(x2 + pw, frame_w),
        min(y2 + ph, frame_h),
    )

def is_blurry_crop(raw_crop):
    gray = cv2.cvtColor(raw_crop, cv2.COLOR_BGR2GRAY)
    var  = cv2.Laplacian(gray, cv2.CV_64F).var()
    return var < 15, var

def make_ocr_variants(raw_crop):
    up = cv2.resize(raw_crop, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    variants = []

    # 1. CLAHE (best baseline)
    lab  = cv2.cvtColor(up, cv2.COLOR_BGR2LAB)
    cl   = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    variants.append(("CLAHE", clahe_bgr))

    # 2. SHARPEN
    blur  = cv2.GaussianBlur(up, (0, 0), 3)
    sharp = cv2.addWeighted(up, 1.8, blur, -0.8, 0)
    variants.append(("SHARP", sharp))

    # 3. THRESH + MORPH CLOSE
    gray     = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
    thresh   = cv2.adaptiveThreshold(
        denoised, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 8
    )
    kernel = np.ones((2, 2), np.uint8)
    morph  = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    morph3 = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
    variants.append(("THRESH_MORPH", morph3))

    return variants


def run_paddleocr_on_crop(img_bgr, label=""):
    results = ocr_reader.ocr(img_bgr, cls=True)

    if not results or not results[0]:
        return "", 0.0, []

    texts = []
    for line in results[0]:
        txt = line[1][0]
        conf = line[1][1]
        texts.append((txt, conf))

    kept = [(t.strip(), c) for t, c in texts if t.strip() and c > 0.3]

    if not kept:
        return "", 0.0, texts

    joined = "".join(t for t, _ in kept)
    best_conf = max(c for _, c in kept)

    return joined, best_conf, texts

def extract_text_paddleocr(raw_crop, debug_dir=None, tid=None, frame_no=None):
    if raw_crop is None or raw_crop.size == 0:
        return ""

    blurry, blur_var = is_blurry_crop(raw_crop)
    if blurry:
        print(f"      [OCR] skipped blurry raw crop (var={blur_var:.1f})")
        return ""

    print(f"      [OCR] trying  blur_var={blur_var:.1f}")

    # ✅ Save exactly what goes into PaddleOCR (for debugging)
    if debug_dir and tid is not None:
        cv2.imwrite(str(debug_dir / f"tid{tid}_f{frame_no}_TO_OCR.jpg"), raw_crop)

    raw_text, conf, results = run_paddleocr_on_crop(raw_crop, label="RAW")

    if results:
        for txt, c in results:
            print(f"      [OCR RAW] '{txt}'  conf={c:.2f}")

    number = extract_valid_number(raw_text)

    if number:
        print(f"      [OCR] BEST: {number}  conf={conf:.2f}")
    else:
        print(f"      [OCR] no valid 11-digit number found")

    return number

# ================= COUNTING LOGIC =================
def get_side(pos, line_pos, buffer_px, axis):
    lo, hi = line_pos - buffer_px, line_pos + buffer_px
    if axis == "x":
        if pos < lo: return "left"
        if pos > hi: return "right"
    else:
        if pos < lo: return "top"
        if pos > hi: return "bottom"
    return "buffer"

def crossed_line(prev_side, curr_side, axis):
    if prev_side is None or prev_side == "buffer" or curr_side == "buffer":
        return None
    if prev_side == curr_side:
        return None
    if axis == "x":
        if prev_side == "right" and curr_side == "left": return "out"
    else:
        if prev_side == "bottom" and curr_side == "top": return "out"
    return None


def draw_ocr_panel(frame, ocr_results, ocr_candidates, ocr_done, names, OCR_CLASS_NAME):
    h, w = frame.shape[:2]

    FONT    = cv2.FONT_HERSHEY_SIMPLEX
    PAD_X   = 12
    PAD_TOP = 10
    ROW_H   = 36
    HDR_H   = 26
    PANEL_W = 280

    entries = [(tid, num) for tid, num in ocr_results.items()]
    if not entries:
        return

    PANEL_H = min(PAD_TOP + HDR_H + 4 + ROW_H * len(entries) + PAD_TOP, h)
    px = w - PANEL_W

    ov = frame.copy()
    cv2.rectangle(ov, (px, 0), (w, PANEL_H), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.75, frame, 0.25, 0, frame)

    hy = PAD_TOP + HDR_H - 4
    cv2.putText(frame, "CONFIRMED WAGONS", (px + PAD_X, hy),
                FONT, 0.45, (0, 255, 100), 1, cv2.LINE_AA)

    div_y = PAD_TOP + HDR_H + 2
    cv2.line(frame, (px + PAD_X, div_y), (w - PAD_X, div_y), (55, 55, 55), 1)

    for idx, (tid, num) in enumerate(entries):
        ry = div_y + 6 + ROW_H * idx
        if ry + ROW_H > h:
            break

        if idx % 2 == 0:
            ro = frame.copy()
            cv2.rectangle(ro, (px, ry), (w, ry + ROW_H), (30, 30, 30), -1)
            cv2.addWeighted(ro, 0.35, frame, 0.40, 0, frame)

        text_y = ry + 26

        # Wagon number
        cv2.putText(frame, num, (px + PAD_X, text_y),
                    FONT, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

        # Detected label (closer now)
        cv2.putText(frame, "Detected", (px + PANEL_W - 90, text_y),
                    FONT, 0.45, (0, 200, 80), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (px, 0), (w - 1, PANEL_H), (70, 70, 70), 1)

# ================= UTIL =================
def get_class_color(cls):
    np.random.seed(abs(hash(cls)) % (2**32))
    return tuple(int(c) for c in np.random.randint(40, 255, 3))

def make_save_dir(project, name):
    base     = Path(project) / name
    save_dir = base
    i = 2
    while save_dir.exists():
        save_dir = Path(f"{base}{i}")
        i += 1
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir

# ================= DRAWING =================
def draw_counting_line(frame, axis, line_pos, buffer_px):
    h, w = frame.shape[:2]
    if axis == "x":
        lx  = int(np.clip(line_pos,             0, w - 1))
        blo = int(np.clip(line_pos - buffer_px, 0, w - 1))
        bhi = int(np.clip(line_pos + buffer_px, 0, w - 1))
        cv2.line(frame, (lx, 0), (lx, h - 1), (0, 30, 255), 3)
        for y in range(0, h, 20):
            y2 = min(y + 10, h - 1)
            cv2.line(frame, (blo, y), (blo, y2), (80, 80, 180), 1)
            cv2.line(frame, (bhi, y), (bhi, y2), (80, 80, 180), 1)
        cv2.putText(frame, "COUNT LINE", (min(lx + 6, w - 120), 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 30, 255), 1, cv2.LINE_AA)
    else:
        ly  = int(np.clip(line_pos,             0, h - 1))
        blo = int(np.clip(line_pos - buffer_px, 0, h - 1))
        bhi = int(np.clip(line_pos + buffer_px, 0, h - 1))
        cv2.line(frame, (0, ly), (w - 1, ly), (0, 30, 255), 3)
        for x in range(0, w, 20):
            x2 = min(x + 10, w - 1)
            cv2.line(frame, (x, blo), (x2, blo), (80, 80, 180), 1)
            cv2.line(frame, (x, bhi), (x2, bhi), (80, 80, 180), 1)
        cv2.putText(frame, "COUNT LINE", (10, max(ly - 8, 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 30, 255), 1, cv2.LINE_AA)


def draw_count_hud(frame, count_out, class_order):
    FONT     = cv2.FONT_HERSHEY_SIMPLEX
    PAD_X    = 12
    PAD_TOP  = 10
    ROW_H    = 30
    HDR_H    = 26
    COL_NAME = PAD_X
    COL_OUT  = PAD_X + 175
    PANEL_W  = COL_OUT + 60
    PANEL_H  = PAD_TOP + HDR_H + 4 + ROW_H * len(class_order) + PAD_TOP

    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (PANEL_W, PANEL_H), (15, 15, 15), -1)
    cv2.addWeighted(ov, 0.72, frame, 0.28, 0, frame)

    hy = PAD_TOP + HDR_H - 2
    cv2.putText(frame, "CLASS", (COL_NAME, hy), FONT, 0.46, (160,160,160), 1, cv2.LINE_AA)
    cv2.putText(frame, "Count", (COL_OUT,  hy), FONT, 0.46, (160,160,160), 1, cv2.LINE_AA)

    div_y = PAD_TOP + HDR_H + 2
    cv2.line(frame, (PAD_X, div_y), (PANEL_W - PAD_X, div_y), (55, 55, 55), 1)

    for idx, cls in enumerate(class_order):
        ry = div_y + 6 + ROW_H * idx + 18
        cls_color = get_class_color(cls)
        if idx % 2 == 0:
            ro = frame.copy()
            y0, y1 = div_y + 6 + ROW_H * idx, div_y + 6 + ROW_H * (idx + 1)
            cv2.rectangle(ro, (0, y0), (PANEL_W, y1), (30, 30, 30), -1)
            cv2.addWeighted(ro, 0.35, frame, 0.65, 0, frame)
        cv2.putText(frame, cls,
                    (COL_NAME, ry), FONT, 0.50, cls_color, 1, cv2.LINE_AA)
        cv2.putText(frame, str(count_out.get(cls, 0)),
                    (COL_OUT, ry), FONT, 0.52, (80, 150, 255), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 0), (PANEL_W, PANEL_H), (70, 70, 70), 1)


def draw_wagon_label_inside_mask(frame, mask_bool, tracker_id,
                                cls_name,
                                wagon_number=None, ocr_status=None):
    ys, xs = np.where(mask_bool)
    if len(xs) == 0:
        return

    cx, cy = int(xs.mean()), int(ys.mean())

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    SO   = 1

    # ================= WAGON LOGIC =================
    if cls_name == "wagon":
        # ✅ draw circle at center
        cv2.circle(frame, (cx, cy), 12, (0, 255, 255), 2)

        # ❌ disable ID text for wagon
        return

    # ================= OTHER CLASSES =================
    id_label = f"ID:{tracker_id}"
    (tw, th), _ = cv2.getTextSize(id_label, FONT, 0.6, 2)
    tx = cx - tw // 2
    ty = (cy - 22) if wagon_number else ((cy - 10) if ocr_status else (cy + th // 2))

    cv2.putText(frame, id_label, (tx+SO, ty+SO), FONT, 0.6, (0,0,0), 2, cv2.LINE_AA)
    cv2.putText(frame, id_label, (tx,    ty),    FONT, 0.6, (255,255,255), 2, cv2.LINE_AA)

    if ocr_status and not wagon_number:
        (sw, sh), _ = cv2.getTextSize(ocr_status, FONT, 0.45, 1)
        sx, sy = cx - sw // 2, ty + sh + 6
        cv2.putText(frame, ocr_status, (sx+SO, sy+SO), FONT, 0.45, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(frame, ocr_status, (sx,    sy),    FONT, 0.45, (0,220,255), 1, cv2.LINE_AA)

    if wagon_number:
        (nw, nh), _ = cv2.getTextSize(wagon_number, FONT, 0.65, 2)
        nx, ny = cx - nw // 2, ty + nh + 8
        cv2.putText(frame, wagon_number, (nx+SO, ny+SO), FONT, 0.65, (0,0,0), 2, cv2.LINE_AA)
        cv2.putText(frame, wagon_number, (nx,    ny),    FONT, 0.65, (0,255,255), 2, cv2.LINE_AA)

    cv2.putText(frame, id_label, (tx+SO, ty+SO), FONT, 0.6, (0,0,0),      2, cv2.LINE_AA)
    cv2.putText(frame, id_label, (tx,    ty),    FONT, 0.6, (255,255,255), 2, cv2.LINE_AA)

    if ocr_status and not wagon_number:
        (sw, sh), _ = cv2.getTextSize(ocr_status, FONT, 0.45, 1)
        sx, sy = cx - sw // 2, ty + sh + 6
        cv2.putText(frame, ocr_status, (sx+SO, sy+SO), FONT, 0.45, (0,0,0),    1, cv2.LINE_AA)
        cv2.putText(frame, ocr_status, (sx,    sy),    FONT, 0.45, (0,220,255), 1, cv2.LINE_AA)

    if wagon_number:
        (nw, nh), _ = cv2.getTextSize(wagon_number, FONT, 0.65, 2)
        nx, ny = cx - nw // 2, ty + nh + 8
        cv2.putText(frame, wagon_number, (nx+SO, ny+SO), FONT, 0.65, (0,0,0),    2, cv2.LINE_AA)
        cv2.putText(frame, wagon_number, (nx,    ny),    FONT, 0.65, (0,255,255), 2, cv2.LINE_AA)


# ================= SAVE HELPERS =================
def init_csv(save_dir):
    csv_path = save_dir / "ocr_results.csv"
    f = open(csv_path, "w", newline="", encoding="utf-8")
    w = csv.writer(f)
    w.writerow(["timestamp", "tracker_id", "wagon_number", "frame_no"])
    print(f"📄 OCR CSV  → {csv_path}")
    return f, w

def save_summary_json(save_dir, count_out, ocr_results):
    data = {
        "run_time":         datetime.now().isoformat(),
        "count_out":        count_out,
        "ocr_wagon_numbers": {str(tid): num for tid, num in ocr_results.items()}
    }
    out = save_dir / "summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"📊 Summary  → {out}")


# ==================================================
@smart_inference_mode()
def run(weights, source, imgsz=640, conf_thres=0.25, iou_thres=0.45,
        device="", project="runs", name="exp", axis="x", ocr_class=None):

    save_dir  = make_save_dir(project, name)
    debug_dir = save_dir / "ocr_debug_crops"
    debug_dir.mkdir(exist_ok=True)
    print(f"💾 Saving to: {save_dir}")

    device = select_device(device)
    model  = DetectMultiBackend(weights, device=device)
    stride, names = model.stride, model.names
    imgsz  = check_img_size(imgsz, s=stride)
    class_order = [v for v in names.values() if v != "engine"]

    global OCR_CLASS_NAME
    OCR_CLASS_NAME = detect_ocr_class(names, override=ocr_class)

    dataset = LoadStreams(source) if source.isnumeric() else LoadImages(source)

    vid_writer = None
    out_video  = save_dir / "output.mp4"

    tracker   = Sort()
    count_out = {v: 0 for v in names.values() if v != "engine"}

    last_confirmed_side = {}
    ocr_candidates      = {}
    ocr_done            = {}
    ocr_results         = {}
    ocr_attempt_no      = {}

    csv_file, csv_writer = init_csv(save_dir)

    frame_no  = 0
    line_pos  = None
    buffer_px = None

    for data in tqdm(dataset):

        if STOP_REQUESTED:
            break

        path, im, im0s, vid_cap, _ = data
        raw   = im0s[0] if isinstance(im0s, list) else im0s
        frame = raw.copy()
        frame_no += 1

        h, w = frame.shape[:2]

        if axis == "x":
            line_pos  = int(w * LINE_RATIO)
            buffer_px = int(w * BUFFER_RATIO)
        else:
            line_pos  = int(h * LINE_RATIO)
            buffer_px = int(h * BUFFER_RATIO)

        if vid_writer is None:
            fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
            fps = fps if fps > 0 else 25
            fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
            vid_writer = cv2.VideoWriter(str(out_video), fourcc, fps, (w, h))
            print(f"🎬 Video → {out_video}  ({w}×{h} @ {fps:.1f} fps)")

        im_t = torch.from_numpy(im).to(device).float() / 255.0
        if im_t.ndim == 3:
            im_t = im_t[None]

        out_m = model(im_t)
        if isinstance(out_m, (list, tuple)):
            pred, proto = out_m[0], out_m[1]
        else:
            raise ValueError("Invalid model output")
        if proto is None:
            raise ValueError("Segmentation model required")
        allowed_classes = [k for k, v in names.items() if v != "engine"]
        pred = non_max_suppression(
            pred,
            conf_thres,
            iou_thres,
            classes=allowed_classes,
            nm=32
        )
        detections = []

        if pred[0] is not None:
            pred[0][:, :4] = scale_boxes(im_t.shape[2:], pred[0][:, :4], frame.shape).round()
            if isinstance(proto, list): proto = proto[0]
            if not isinstance(proto, torch.Tensor):
                raise TypeError(f"Proto must be tensor, got {type(proto)}")
            if proto.ndim == 4: proto = proto[0]
            if proto.ndim != 3:
                raise ValueError(f"Unexpected proto shape: {proto.shape}")

            masks = process_mask(
                proto, pred[0][:, 6:], pred[0][:, :4],
                frame.shape[:2], upsample=True
            ).cpu().numpy()

            for i, (*xyxy, conf, cls) in enumerate(pred[0][:, :6]):
                cls_name = names[int(cls)]
                    # ❌ REMOVE engine class
                if cls_name == "engine":
                    continue
                x1, y1, x2, y2 = map(int, xyxy)
                detections.append([x1, y1, x2, y2, conf.item(), int(cls), masks[i]])

        tracks = tracker.update(
            np.array([d[:5] for d in detections]) if detections else np.empty((0, 5))
        )

        for trk in tracks.astype(int):
            x1, y1, x2, y2, tid = trk

            det = next((d for d in detections if abs(x1 - d[0]) < 20), None)
            if det is None:
                continue

            cls_name  = names[det[5]]
            mask      = det[6]
            mask_bool = mask > 0.5

            ys, xs = np.where(mask_bool)
            if len(xs) == 0:
                continue

            cx, cy = int(xs.mean()), int(ys.mean())
            pos    = cx if axis == "x" else cy

            # ---- line crossing ----
            curr_side = get_side(pos, line_pos, buffer_px, axis)
            prev_side = last_confirmed_side.get(tid)
            direction = crossed_line(prev_side, curr_side, axis)

            if direction == "out":
                count_out[cls_name] += 1
                print(f"⬅  OUT | {cls_name} | TID:{tid} | frame:{frame_no}")

                if tid in ocr_temp and tid not in ocr_results:
                    final = ocr_temp[tid]
                    ocr_results[tid] = final
                    ts = datetime.now().isoformat()
                    csv_writer.writerow([ts, tid, final, frame_no])
                    csv_file.flush()
                    print(f"🔥 FINAL CONFIRMED AFTER CROSSING TID:{tid} → {final}")

            if curr_side != "buffer":
                last_confirmed_side[tid] = curr_side

            # ---- OCR BLOCK ----
            ocr_status = None

            if OCR_CLASS_NAME and cls_name == OCR_CLASS_NAME and tid not in ocr_done:

                n_so_far   = len(ocr_candidates.get(tid, []))
                ocr_status = f"scanning {n_so_far}/{MIN_OCR_FRAMES}"

                ex1, ey1, ex2, ey2 = expanded_bbox(x1, y1, x2, y2, BBOX_PAD_RATIO, w, h)
                raw_crop = raw[ey1:ey2, ex1:ex2].copy()

                attempt = ocr_attempt_no.get(tid, 0) + 1
                ocr_attempt_no[tid] = attempt
                if DEBUG_CROPS and attempt <= 8:
                    cv2.imwrite(
                        str(debug_dir / f"tid{tid}_f{frame_no}_raw.jpg"),
                        raw_crop
                    )

                # ✅ Now uses PaddleOCR instead of EasyOCR
                number = extract_text_paddleocr(
                    raw_crop,
                    debug_dir=debug_dir if (DEBUG_CROPS and attempt <= 4) else None,
                    tid=tid,
                    frame_no=frame_no,
                )

                if number:
                    ocr_candidates.setdefault(tid, []).append(number)
                    n = len(ocr_candidates[tid])
                    ocr_status = f"reading {n}/{MIN_OCR_FRAMES}"
                    print(f"   ✔ OCR TID:{tid} [{n}/{MIN_OCR_FRAMES}] → {number}")

                    if n >= MIN_OCR_FRAMES:
                        most_common, count = Counter(ocr_candidates[tid]).most_common(1)[0]
                        if count >= 2:
                            final        = most_common
                            ocr_temp[tid] = final
                            ocr_done[tid] = True
                            print(f"🕒 OCR READY (waiting for crossing) TID:{tid} → {final}")
                            ocr_status = None
                            ts = datetime.now().isoformat()
                            csv_writer.writerow([ts, tid, final, frame_no])
                            csv_file.flush()
                            print(f"🔥 CONFIRMED TID:{tid} → {final}  (frame {frame_no})")

                            cv2.imwrite(
                                str(debug_dir / f"CONFIRMED_tid{tid}_{final}.jpg"),
                                raw_crop
                            )
                        else:
                            ocr_status = f"no consensus ({n} reads)"
                            print(f"   ⚠ No consensus yet TID:{tid} — {ocr_candidates[tid]}")

            # ---- draw mask overlay ----
            color = get_class_color(cls_name)

            # ❌ Skip mask for wagon
            if cls_name != "wagon":
                frame[mask_bool] = frame[mask_bool] * 0.5 + np.array(color) * 0.5

            cv2.putText(frame, cls_name, (x1, max(y1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            if cls_name != "wagon":
                draw_wagon_label_inside_mask(
                frame, mask_bool, tid,
                cls_name,
                wagon_number=ocr_results.get(tid),
                ocr_status=ocr_status,
            )
        draw_counting_line(frame, axis, line_pos, buffer_px)
        draw_count_hud(frame, count_out, class_order)
        draw_ocr_panel(frame, ocr_results, ocr_candidates, ocr_done, names, OCR_CLASS_NAME)

        vid_writer.write(frame)
        cv2.imshow("FINAL SYSTEM", frame)
        if cv2.waitKey(1) == 27:
            break

    if vid_writer:
        vid_writer.release()
    csv_file.close()
    cv2.destroyAllWindows()
    save_summary_json(save_dir, count_out, ocr_results)

    print("\n" + "=" * 60)
    print(f"  Run complete → {save_dir}")
    print(f"  ├─ output.mp4              annotated video")
    print(f"  ├─ ocr_results.csv         wagon numbers + timestamps")
    print(f"  ├─ summary.json            counts + wagon-number map")
    print(f"  └─ ocr_debug_crops/        raw crops + all 3 variants per attempt")
    print("=" * 60)


# ================= CLI =================
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    required=True)
    parser.add_argument("--source",     required=True)
    parser.add_argument("--axis",       choices=["x", "y"], default="x")
    parser.add_argument("--imgsz",      type=int,   default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres",  type=float, default=0.45)
    parser.add_argument("--device",     default="")
    parser.add_argument("--project",    default="runs")
    parser.add_argument("--name",       default="exp")
    parser.add_argument("--ocr-class",  default=None,
                        help="Class to OCR. Auto-detected if not set.")
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))