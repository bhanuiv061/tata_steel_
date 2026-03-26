



# import argparse
# import sys
# import time
# import traceback
# from pathlib import Path
# import cv2
# import torch
# import numpy as np
# import os
# import pathlib
# import signal

# # ================= WINDOWS FIX =================
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# # ================= ROOT =================
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# # ================= YOLO =================
# from models.common import DetectMultiBackend
# from utils.dataloaders import LoadImages, LoadStreams
# from utils.general import check_img_size, non_max_suppression, scale_boxes
# from utils.segment.general import process_mask
# from utils.torch_utils import select_device, smart_inference_mode
# from sort.sort import Sort

# # ================= CONFIG =================
# LINE_RATIO = 0.6
# STOP_REQUESTED = False

# # ================= SIGNAL =================
# def request_stop(sig=None, frame=None):
#     global STOP_REQUESTED
#     STOP_REQUESTED = True
#     print("\n⚠ Stopping safely...")

# signal.signal(signal.SIGINT, request_stop)
# signal.signal(signal.SIGTERM, request_stop)

# # ================= UTIL =================
# def get_class_color(cls):
#     np.random.seed(abs(hash(cls)) % (2**32))
#     return tuple(int(c) for c in np.random.randint(50, 255, 3))

# def get_next_path(save_dir, prefix):
#     save_dir.mkdir(parents=True, exist_ok=True)
#     files = list(save_dir.glob(f"{prefix}_*.mp4"))
#     if not files:
#         return save_dir / f"{prefix}_0001.mp4"
#     nums = [int(f.stem.split("_")[-1]) for f in files if f.stem.split("_")[-1].isdigit()]
#     return save_dir / f"{prefix}_{max(nums)+1:04d}.mp4"

# def draw_panel(frame, count_in):
#     h, w = frame.shape[:2]
#     x = w - 250
#     y = 10

#     overlay = frame.copy()
#     cv2.rectangle(overlay, (x-10,y), (w-5,y+200), (20,20,20), -1)
#     cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

#     cv2.putText(frame, "CLASS   IN", (x,y+25),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,215,255),2)

#     for i, cls in enumerate(count_in.keys()):
#         text = f"{cls[:10]:<10}{count_in[cls]:>4}"
#         cv2.putText(frame, text, (x,y+60+i*30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255),2)

# # ================= CROSS LOGIC =================
# def crossed_in(prev_cx, prev_cy, cx, cy, line_pos, direction):
#     if direction == "vertical":
#         return prev_cy < line_pos and cy >= line_pos
#     else:
#         return prev_cx < line_pos and cx >= line_pos

# # ==================================================
# @smart_inference_mode()
# def run(weights, source, imgsz=640, conf_thres=0.25,
#         iou_thres=0.45, device="", project="runs/seg",
#         name="exp", direction="vertical"):

#     raw_writer = None
#     ann_writer = None

#     try:
#         device = select_device(device)
#         model  = DetectMultiBackend(weights, device=device)
#         stride, names = model.stride, model.names
#         imgsz  = check_img_size(imgsz, s=stride)
#         model.warmup(imgsz=(1,3,imgsz,imgsz))

#         dataset = LoadStreams(source, img_size=imgsz, stride=stride) \
#             if source.isnumeric() else LoadImages(source, img_size=imgsz, stride=stride)

#         tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

#         count_in  = {cls:0 for cls in names.values()}
#         prev_centroid = {}
#         counted_ids = set()
#         track_class = {}

#         save_dir = Path(project)/name
#         raw_path = get_next_path(save_dir,"raw")
#         ann_path = get_next_path(save_dir,"annotated")

#         LINE_POS = None

#         for data in dataset:
#             if STOP_REQUESTED:
#                 break

#             path, im, im0s, vid_cap, _ = data
#             raw = im0s.copy() if not isinstance(im0s,list) else im0s[0].copy()
#             frame = raw.copy()
#             h,w = frame.shape[:2]

#             if LINE_POS is None:
#                 LINE_POS = int(h*LINE_RATIO) if direction=="vertical" else int(w*LINE_RATIO)

#             # preprocess
#             im = torch.from_numpy(im).to(device).float()/255.0
#             if im.ndim==3:
#                 im = im[None]

#             output = model(im)

#             if isinstance(output, (list, tuple)) and len(output) >= 2:
#                 pred, proto = output[0], output[1]
#             else:
#                 raise ValueError("❌ Not a segmentation model")

#             pred = non_max_suppression(pred, conf_thres, iou_thres, nm=32)

#             detections=[]

#             if len(pred[0]):
#                 pred[0][:,:4]=scale_boxes(im.shape[2:],pred[0][:,:4],frame.shape).round()
#                 masks = process_mask(proto[0], pred[0][:,6:], pred[0][:,:4], (h,w), upsample=True)

#                 for i,(*xyxy,conf,cls) in enumerate(pred[0][:,:6]):
#                     x1,y1,x2,y2 = map(int,xyxy)
#                     detections.append([x1,y1,x2,y2,conf.item(),int(cls),masks[i].cpu().numpy()])

#             tracks = tracker.update(np.array([d[:5] for d in detections]) if detections else np.empty((0,5)))

#             for trk in tracks.astype(int):
#                 x1,y1,x2,y2,tid = trk

#                 best_iou, det = 0, None
#                 for d in detections:
#                     xx1,yy1 = max(x1,d[0]), max(y1,d[1])
#                     xx2,yy2 = min(x2,d[2]), min(y2,d[3])
#                     inter = max(0,xx2-xx1)*max(0,yy2-yy1)
#                     area1=(x2-x1)*(y2-y1)
#                     area2=(d[2]-d[0])*(d[3]-d[1])
#                     iou=inter/(area1+area2-inter+1e-6)
#                     if iou>best_iou:
#                         best_iou, det = iou, d

#                 if det is None:
#                     continue

#                 if tid not in track_class:
#                     track_class[tid] = names[det[5]]

#                 cls_name = track_class[tid]
#                 mask = det[6]

#                 ys,xs = np.where(mask > 0.5)
#                 if len(ys)==0:
#                     continue

#                 cx = int(xs.mean())
#                 cy = int(ys.mean())

#                 # ================= COUNT =================
#                 prev_c = prev_centroid.get(tid, None)

#                 if prev_c is not None and tid not in counted_ids:
#                     prev_cx, prev_cy = prev_c

#                     if crossed_in(prev_cx, prev_cy, cx, cy, LINE_POS, direction):
#                         count_in[cls_name] += 1
#                         counted_ids.add(tid)

#                 prev_centroid[tid] = (cx, cy)

#                 # ================= DRAW =================
#                 color = get_class_color(cls_name)
#                 mask_bool = mask > 0.5

#                 frame[mask_bool] = (
#                     frame[mask_bool] * 0.5 + np.array(color) * 0.5
#                 ).astype(np.uint8)

#                 cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

#                 cv2.putText(frame,f"{cls_name}#{tid}",(x1,y1-5),
#                             cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

#             # draw line
#             if direction=="vertical":
#                 cv2.line(frame,(0,LINE_POS),(w,LINE_POS),(0,255,255),2)
#             else:
#                 cv2.line(frame,(LINE_POS,0),(LINE_POS,h),(0,255,255),2)

#             draw_panel(frame,count_in)

#             if raw_writer is None:
#                 fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
#                 raw_writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,(w,h))
#                 ann_writer = cv2.VideoWriter(str(ann_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,(w,h))

#             raw_writer.write(raw)
#             ann_writer.write(frame)

#             cv2.imshow("Output",frame)
#             if cv2.waitKey(1)&0xFF in [27,ord('q')]:
#                 break

#     except Exception as e:
#         traceback.print_exc()

#     finally:
#         if raw_writer: raw_writer.release()
#         if ann_writer: ann_writer.release()
#         cv2.destroyAllWindows()

#         print("\nFINAL COUNT:")
#         for k in count_in:
#             print(f"{k} -> IN:{count_in[k]}")

# # ================= CLI =================
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", required=True)
#     parser.add_argument("--source", required=True)
#     parser.add_argument("--imgsz", type=int, default=640)
#     parser.add_argument("--conf-thres", type=float, default=0.25)
#     parser.add_argument("--iou-thres", type=float, default=0.45)
#     parser.add_argument("--device", default="")
#     parser.add_argument("--project", default="runs/seg")
#     parser.add_argument("--name", default="exp")
#     parser.add_argument("--direction", default="vertical",
#                         choices=["vertical","horizontal"])
#     return parser.parse_args()

# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))






























































# import argparse
# import sys
# import time
# import traceback
# from pathlib import Path
# import cv2
# import torch
# import numpy as np
# import os
# import pathlib
# import signal

# # ================= WINDOWS FIX =================
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# # ================= ROOT =================
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# # ================= YOLO =================
# from models.common import DetectMultiBackend
# from utils.dataloaders import LoadImages, LoadStreams
# from utils.general import check_img_size, non_max_suppression, scale_boxes
# from utils.segment.general import process_mask
# from utils.torch_utils import select_device, smart_inference_mode
# from sort.sort import Sort

# # ================= CONFIG =================
# LINE_RATIO = 0.6
# STOP_REQUESTED = False

# # ================= SIGNAL =================
# def request_stop(sig=None, frame=None):
#     global STOP_REQUESTED
#     STOP_REQUESTED = True
#     print("\n⚠ Stopping safely...")

# signal.signal(signal.SIGINT, request_stop)
# signal.signal(signal.SIGTERM, request_stop)

# # ================= UTIL =================
# def get_class_color(cls):
#     np.random.seed(abs(hash(cls)) % (2**32))
#     return tuple(int(c) for c in np.random.randint(50, 255, 3))

# def get_next_path(save_dir, prefix):
#     save_dir.mkdir(parents=True, exist_ok=True)
#     files = list(save_dir.glob(f"{prefix}_*.mp4"))
#     if not files:
#         return save_dir / f"{prefix}_0001.mp4"
#     nums = [int(f.stem.split("_")[-1]) for f in files if f.stem.split("_")[-1].isdigit()]
#     return save_dir / f"{prefix}_{max(nums)+1:04d}.mp4"

# def draw_panel(frame, count_in):
#     h, w = frame.shape[:2]
#     x = 10       # left side instead of w - 250
#     y = 10

#     overlay = frame.copy()
#     cv2.rectangle(overlay, (x, y), (x + 240, y + 200), (20, 20, 20), -1)
#     cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

#     cv2.putText(frame, "CLASS   IN", (x + 5, y + 25),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

#     for i, cls in enumerate(count_in.keys()):
#         text = f"{cls[:10]:<10}{count_in[cls]:>4}"
#         cv2.putText(frame, text, (x + 5, y + 60 + i * 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# # ================= CROSS LOGIC =================
# def crossed_in(prev_cx, prev_cy, cx, cy, line_pos, direction):
#     if direction == "vertical":
#         return prev_cy < line_pos and cy >= line_pos
#     else:
#         return prev_cx < line_pos and cx >= line_pos

# # ==================================================
# @smart_inference_mode()
# def run(weights, source, imgsz=640, conf_thres=0.25,
#         iou_thres=0.45, device="", project="runs/seg",
#         name="exp", direction="vertical"):

#     raw_writer = None
#     ann_writer = None

#     try:
#         device = select_device(device)
#         model  = DetectMultiBackend(weights, device=device)
#         stride, names = model.stride, model.names
#         imgsz  = check_img_size(imgsz, s=stride)
#         model.warmup(imgsz=(1,3,imgsz,imgsz))

#         dataset = LoadStreams(source, img_size=imgsz, stride=stride) \
#             if source.isnumeric() else LoadImages(source, img_size=imgsz, stride=stride)

#         tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

#         count_in  = {cls:0 for cls in names.values()}
#         prev_centroid = {}
#         counted_ids = set()
#         track_class = {}

#         save_dir = Path(project)/name
#         raw_path = get_next_path(save_dir,"raw")
#         ann_path = get_next_path(save_dir,"top_view_annotated")

#         LINE_POS = None

#         for data in dataset:
#             if STOP_REQUESTED:
#                 break

#             path, im, im0s, vid_cap, _ = data
#             raw = im0s.copy() if not isinstance(im0s,list) else im0s[0].copy()
#             frame = raw.copy()
#             h,w = frame.shape[:2]

#             if LINE_POS is None:
#                 LINE_POS = int(h*LINE_RATIO) if direction=="vertical" else int(w*LINE_RATIO)

#             # preprocess
#             im = torch.from_numpy(im).to(device).float()/255.0
#             if im.ndim==3:
#                 im = im[None]

#             output = model(im)

#             if isinstance(output, (list, tuple)) and len(output) >= 2:
#                 pred, proto = output[0], output[1]
#             else:
#                 raise ValueError("❌ Not a segmentation model")

#             pred = non_max_suppression(pred, conf_thres, iou_thres, nm=32)

#             detections=[]

#             if len(pred[0]):
#                 pred[0][:,:4]=scale_boxes(im.shape[2:],pred[0][:,:4],frame.shape).round()
#                 masks = process_mask(proto[0], pred[0][:,6:], pred[0][:,:4], (h,w), upsample=True)

#                 for i,(*xyxy,conf,cls) in enumerate(pred[0][:,:6]):
#                     x1,y1,x2,y2 = map(int,xyxy)
#                     detections.append([x1,y1,x2,y2,conf.item(),int(cls),masks[i].cpu().numpy()])

#             tracks = tracker.update(np.array([d[:5] for d in detections]) if detections else np.empty((0,5)))

#             for trk in tracks.astype(int):
#                 x1,y1,x2,y2,tid = trk

#                 best_iou, det = 0, None
#                 for d in detections:
#                     xx1,yy1 = max(x1,d[0]), max(y1,d[1])
#                     xx2,yy2 = min(x2,d[2]), min(y2,d[3])
#                     inter = max(0,xx2-xx1)*max(0,yy2-yy1)
#                     area1=(x2-x1)*(y2-y1)
#                     area2=(d[2]-d[0])*(d[3]-d[1])
#                     iou=inter/(area1+area2-inter+1e-6)
#                     if iou>best_iou:
#                         best_iou, det = iou, d

#                 if det is None:
#                     continue

#                 if tid not in track_class:
#                     track_class[tid] = names[det[5]]

#                 cls_name = track_class[tid]
#                 mask = det[6]

#                 ys,xs = np.where(mask > 0.5)
#                 if len(ys)==0:
#                     continue

#                 cx = int(xs.mean())
#                 cy = int(ys.mean())

#                 # ================= COUNT =================
#                 prev_c = prev_centroid.get(tid, None)

#                 if prev_c is not None and tid not in counted_ids:
#                     prev_cx, prev_cy = prev_c

#                     if crossed_in(prev_cx, prev_cy, cx, cy, LINE_POS, direction):
#                         count_in[cls_name] += 1
#                         counted_ids.add(tid)

#                 prev_centroid[tid] = (cx, cy)

#                 # ================= DRAW =================
#                 color = get_class_color(cls_name)
#                 # mask_bool = mask > 0.5

#                 # frame[mask_bool] = (
#                 #     frame[mask_bool] * 0.5 + np.array(color) * 0.5
#                 # ).astype(np.uint8)

#                 cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

#                 cv2.putText(frame,f"{cls_name}#{tid}",(x1,y1-5),
#                             cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

#             # draw line
#             if direction=="vertical":
#                 cv2.line(frame,(0,LINE_POS),(w,LINE_POS),(0,255,255),2)
#             else:
#                 cv2.line(frame,(LINE_POS,0),(LINE_POS,h),(0,255,255),2)

#             draw_panel(frame,count_in)

#             if raw_writer is None:
#                 fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
#                 raw_writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,(w,h))
#                 ann_writer = cv2.VideoWriter(str(ann_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,(w,h))

#             raw_writer.write(raw)
#             ann_writer.write(frame)

#             cv2.imshow("Output",frame)
#             if cv2.waitKey(1)&0xFF in [27,ord('q')]:
#                 break

#     except Exception as e:
#         traceback.print_exc()

#     finally:
#         if raw_writer: raw_writer.release()
#         if ann_writer: ann_writer.release()
#         cv2.destroyAllWindows()

#         print("\nFINAL COUNT:")
#         for k in count_in:
#             print(f"{k} -> IN:{count_in[k]}")

# # ================= CLI =================
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", required=True)
#     parser.add_argument("--source", required=True)
#     parser.add_argument("--imgsz", type=int, default=640)
#     parser.add_argument("--conf-thres", type=float, default=0.25)
#     parser.add_argument("--iou-thres", type=float, default=0.45)
#     parser.add_argument("--device", default="")
#     parser.add_argument("--project", default="runs/seg")
#     parser.add_argument("--name", default="exp")
#     parser.add_argument("--direction", default="vertical",
#                         choices=["vertical","horizontal"])
#     return parser.parse_args()

# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))



































# import argparse
# import sys
# import time
# import traceback
# from pathlib import Path
# import cv2
# import torch
# import numpy as np
# import os
# import pathlib
# import signal

# # ================= WINDOWS FIX =================
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath

# # ================= ROOT =================
# FILE = Path(__file__).resolve()
# ROOT = FILE.parents[0]
# if str(ROOT) not in sys.path:
#     sys.path.append(str(ROOT))
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# # ================= YOLO =================
# from models.common import DetectMultiBackend
# from utils.dataloaders import LoadImages, LoadStreams
# from utils.general import check_img_size, non_max_suppression, scale_boxes
# from utils.segment.general import process_mask
# from utils.torch_utils import select_device, smart_inference_mode
# from sort.sort import Sort

# # ================= CONFIG =================
# LINE_RATIO = 0.6
# STOP_REQUESTED = False

# # ================= SIGNAL =================
# def request_stop(sig=None, frame=None):
#     global STOP_REQUESTED
#     STOP_REQUESTED = True
#     print("\n⚠ Stopping safely...")

# signal.signal(signal.SIGINT, request_stop)
# signal.signal(signal.SIGTERM, request_stop)

# # ================= UTIL =================
# def get_class_color(cls):
#     np.random.seed(abs(hash(cls)) % (2**32))
#     return tuple(int(c) for c in np.random.randint(50, 255, 3))

# def get_next_path(save_dir, prefix):
#     save_dir.mkdir(parents=True, exist_ok=True)
#     files = list(save_dir.glob(f"{prefix}_*.mp4"))
#     if not files:
#         return save_dir / f"{prefix}_0001.mp4"
#     nums = [int(f.stem.split("_")[-1]) for f in files if f.stem.split("_")[-1].isdigit()]
#     return save_dir / f"{prefix}_{max(nums)+1:04d}.mp4"

# def draw_panel(frame, count_in):
#     h, w = frame.shape[:2]
#     x = 10       # left side instead of w - 250
#     y = 10

#     overlay = frame.copy()
#     cv2.rectangle(overlay, (x, y), (x + 240, y + 200), (20, 20, 20), -1)
#     cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

#     cv2.putText(frame, "CLASS   IN", (x + 5, y + 25),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

#     for i, cls in enumerate(count_in.keys()):
#         text = f"{cls[:10]:<10}{count_in[cls]:>4}"
#         cv2.putText(frame, text, (x + 5, y + 60 + i * 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# # ================= CROSS LOGIC =================
# def crossed_in(prev_cx, prev_cy, cx, cy, line_pos, direction):
#     if direction == "vertical":
#         return prev_cy < line_pos and cy >= line_pos
#     else:
#         return prev_cx < line_pos and cx >= line_pos

# # ==================================================
# @smart_inference_mode()
# def run(weights, source, imgsz=640, conf_thres=0.25,
#         iou_thres=0.45, device="", project="runs/seg",
#         name="exp", direction="vertical"):

#     raw_writer = None
#     ann_writer = None

#     try:
#         device = select_device(device)
#         model  = DetectMultiBackend(weights, device=device)
#         stride, names = model.stride, model.names
#         imgsz  = check_img_size(imgsz, s=stride)
#         model.warmup(imgsz=(1,3,imgsz,imgsz))

#         dataset = LoadStreams(source, img_size=imgsz, stride=stride) \
#             if source.isnumeric() else LoadImages(source, img_size=imgsz, stride=stride)

#         tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

#         count_in  = {cls:0 for cls in names.values()}
#         prev_centroid = {}
#         counted_ids = set()
#         track_class = {}

#         save_dir = Path(project)/name
#         raw_path = get_next_path(save_dir,"raw")
#         ann_path = get_next_path(save_dir,"top_view_annotated")

#         LINE_POS = None

#         for data in dataset:
#             if STOP_REQUESTED:
#                 break

#             path, im, im0s, vid_cap, _ = data
#             raw = im0s.copy() if not isinstance(im0s,list) else im0s[0].copy()
#             frame = raw.copy()
#             h,w = frame.shape[:2]

#             if LINE_POS is None:
#                 LINE_POS = int(h*LINE_RATIO) if direction=="vertical" else int(w*LINE_RATIO)

#             # preprocess
#             im = torch.from_numpy(im).to(device).float()/255.0
#             if im.ndim==3:
#                 im = im[None]

#             output = model(im)

#             if isinstance(output, (list, tuple)) and len(output) >= 2:
#                 pred, proto = output[0], output[1]
#             else:
#                 raise ValueError("❌ Not a segmentation model")

#             pred = non_max_suppression(pred, conf_thres, iou_thres, nm=32)

#             detections=[]

#             if len(pred[0]):
#                 pred[0][:,:4]=scale_boxes(im.shape[2:],pred[0][:,:4],frame.shape).round()
#                 masks = process_mask(proto[0], pred[0][:,6:], pred[0][:,:4], (h,w), upsample=True)

#                 for i,(*xyxy,conf,cls) in enumerate(pred[0][:,:6]):
#                     x1,y1,x2,y2 = map(int,xyxy)
#                     detections.append([x1,y1,x2,y2,conf.item(),int(cls),masks[i].cpu().numpy()])

#             tracks = tracker.update(np.array([d[:5] for d in detections]) if detections else np.empty((0,5)))

#             for trk in tracks.astype(int):
#                 x1,y1,x2,y2,tid = trk

#                 best_iou, det = 0, None
#                 for d in detections:
#                     xx1,yy1 = max(x1,d[0]), max(y1,d[1])
#                     xx2,yy2 = min(x2,d[2]), min(y2,d[3])
#                     inter = max(0,xx2-xx1)*max(0,yy2-yy1)
#                     area1=(x2-x1)*(y2-y1)
#                     area2=(d[2]-d[0])*(d[3]-d[1])
#                     iou=inter/(area1+area2-inter+1e-6)
#                     if iou>best_iou:
#                         best_iou, det = iou, d

#                 if det is None:
#                     continue

#                 if tid not in track_class:
#                     track_class[tid] = names[det[5]]

#                 cls_name = track_class[tid]
#                 mask = det[6]

#                 ys,xs = np.where(mask > 0.5)
#                 if len(ys)==0:
#                     continue

#                 cx = int(xs.mean())
#                 cy = int(ys.mean())

#                 # ================= COUNT =================
#                 prev_c = prev_centroid.get(tid, None)

#                 if prev_c is not None and tid not in counted_ids:
#                     prev_cx, prev_cy = prev_c

#                     if crossed_in(prev_cx, prev_cy, cx, cy, LINE_POS, direction):
#                         count_in[cls_name] += 1
#                         counted_ids.add(tid)

#                 prev_centroid[tid] = (cx, cy)

#                 # ================= DRAW =================
#                 color = get_class_color(cls_name)
#                 # mask_bool = mask > 0.5

#                 # frame[mask_bool] = (
#                 #     frame[mask_bool] * 0.5 + np.array(color) * 0.5
#                 # ).astype(np.uint8)

#                 cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

#                 cv2.putText(frame,f"{cls_name}#{tid}",(x1,y1-5),
#                             cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

#             # draw line
#             if direction=="vertical":
#                 cv2.line(frame,(0,LINE_POS),(w,LINE_POS),(0,255,255),2)
#             else:
#                 cv2.line(frame,(LINE_POS,0),(LINE_POS,h),(0,255,255),2)

#             draw_panel(frame,count_in)

#             if raw_writer is None:
#                 fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
#                 raw_writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,(w,h))
#                 ann_writer = cv2.VideoWriter(str(ann_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,(w,h))

#             raw_writer.write(raw)
#             ann_writer.write(frame)

#             cv2.imshow("Output",frame)
#             if cv2.waitKey(1)&0xFF in [27,ord('q')]:
#                 break

#     except Exception as e:
#         traceback.print_exc()

#     finally:
#         if raw_writer: raw_writer.release()
#         if ann_writer: ann_writer.release()
#         cv2.destroyAllWindows()

#         print("\nFINAL COUNT:")
#         for k in count_in:
#             print(f"{k} -> IN:{count_in[k]}")

# # ================= CLI =================
# def parse_opt():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", required=True)
#     parser.add_argument("--source", required=True)
#     parser.add_argument("--imgsz", type=int, default=640)
#     parser.add_argument("--conf-thres", type=float, default=0.25)
#     parser.add_argument("--iou-thres", type=float, default=0.45)
#     parser.add_argument("--device", default="")
#     parser.add_argument("--project", default="runs/seg")
#     parser.add_argument("--name", default="exp")
#     parser.add_argument("--direction", default="vertical",
#                         choices=["vertical","horizontal"])
#     return parser.parse_args()

# if __name__ == "__main__":
#     opt = parse_opt()
#     run(**vars(opt))




















































































import argparse
import sys
import time
import traceback
from pathlib import Path
import cv2
import torch
import numpy as np
import os
import pathlib
import signal

# ================= WINDOWS FIX =================
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
yolov5_path = r"D:\bhanu\OneDrive - Imagevision.ai India Pvt Ltd\bhanu_iv061\Manfucturing_sector\TATA_steel\enginnering\final_code_v1\yolov5-master"
sys.path.append(yolov5_path)
# ================= ROOT =================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

# ================= YOLO =================
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadStreams
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.segment.general import process_mask
from utils.torch_utils import select_device, smart_inference_mode
from sort.sort import Sort

# ================= CONFIG =================
LINE_RATIO = 0.6
STOP_REQUESTED = False

# ================= SIGNAL =================
def request_stop(sig=None, frame=None):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n⚠ Stopping safely...")

signal.signal(signal.SIGINT, request_stop)
signal.signal(signal.SIGTERM, request_stop)

# ================= UTIL =================
def get_class_color(cls):
    np.random.seed(abs(hash(cls)) % (2**32))
    return tuple(int(c) for c in np.random.randint(50, 255, 3))

def get_next_path(save_dir, prefix):
    save_dir.mkdir(parents=True, exist_ok=True)
    files = list(save_dir.glob(f"{prefix}_*.mp4"))
    if not files:
        return save_dir / f"{prefix}_0001.mp4"
    nums = [int(f.stem.split("_")[-1]) for f in files if f.stem.split("_")[-1].isdigit()]
    return save_dir / f"{prefix}_{max(nums)+1:04d}.mp4"

def draw_panel(frame, count_in):
    h, w = frame.shape[:2]
    x = 10       # left side instead of w - 250
    y = 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + 240, y + 200), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, "CLASS   IN", (x + 5, y + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

    for i, cls in enumerate(count_in.keys()):
        text = f"{cls[:10]:<10}{count_in[cls]:>4}"
        cv2.putText(frame, text, (x + 5, y + 60 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# ================= CROSS LOGIC =================
def crossed_in(prev_cx, prev_cy, cx, cy, line_pos, direction):
    if direction == "vertical":
        return prev_cy < line_pos and cy >= line_pos
    else:
        return prev_cx < line_pos and cx >= line_pos

# ==================================================
@smart_inference_mode()
def run(weights, source, imgsz=640, conf_thres=0.25,
        iou_thres=0.45, device="", project="runs/seg",
        name="exp", direction="vertical"):

    raw_writer = None
    ann_writer = None

    try:
        device = select_device(device)
        model  = DetectMultiBackend(weights, device=device)
        stride, names = model.stride, model.names
        imgsz  = check_img_size(imgsz, s=stride)
        model.warmup(imgsz=(1,3,imgsz,imgsz))

        dataset = LoadStreams(source, img_size=imgsz, stride=stride) \
            if source.isnumeric() else LoadImages(source, img_size=imgsz, stride=stride)

        tracker = Sort(max_age=30, min_hits=2, iou_threshold=0.2)

        count_in  = {cls:0 for cls in names.values()}
        prev_centroid = {}
        counted_ids = set()
        track_class = {}

        save_dir = Path(project)/name
        raw_path = get_next_path(save_dir,"raw")
        ann_path = get_next_path(save_dir,"top_view_annotated")

        LINE_POS = None

        for data in dataset:
            if STOP_REQUESTED:
                break

            path, im, im0s, vid_cap, _ = data
            raw = im0s.copy() if not isinstance(im0s,list) else im0s[0].copy()
            frame = raw.copy()
            h,w = frame.shape[:2]

            if LINE_POS is None:
                LINE_POS = int(h*LINE_RATIO) if direction=="vertical" else int(w*LINE_RATIO)

            # preprocess
            im = torch.from_numpy(im).to(device).float()/255.0
            if im.ndim==3:
                im = im[None]

            output = model(im)

            if isinstance(output, (list, tuple)) and len(output) >= 2:
                pred, proto = output[0], output[1]
            else:
                raise ValueError("❌ Not a segmentation model")

            pred = non_max_suppression(pred, conf_thres, iou_thres, nm=32)

            detections=[]

            if len(pred[0]):
                pred[0][:,:4]=scale_boxes(im.shape[2:],pred[0][:,:4],frame.shape).round()
                masks = process_mask(proto[0], pred[0][:,6:], pred[0][:,:4], (h,w), upsample=True)

                for i,(*xyxy,conf,cls) in enumerate(pred[0][:,:6]):
                    x1,y1,x2,y2 = map(int,xyxy)
                    detections.append([x1,y1,x2,y2,conf.item(),int(cls),masks[i].cpu().numpy()])

            tracks = tracker.update(np.array([d[:5] for d in detections]) if detections else np.empty((0,5)))

            for trk in tracks.astype(int):
                x1,y1,x2,y2,tid = trk

                best_iou, det = 0, None
                for d in detections:
                    xx1,yy1 = max(x1,d[0]), max(y1,d[1])
                    xx2,yy2 = min(x2,d[2]), min(y2,d[3])
                    inter = max(0,xx2-xx1)*max(0,yy2-yy1)
                    area1=(x2-x1)*(y2-y1)
                    area2=(d[2]-d[0])*(d[3]-d[1])
                    iou=inter/(area1+area2-inter+1e-6)
                    if iou>best_iou:
                        best_iou, det = iou, d

                if det is None:
                    continue

                if tid not in track_class:
                    track_class[tid] = names[det[5]]

                cls_name = track_class[tid]
                mask = det[6]

                ys,xs = np.where(mask > 0.5)
                if len(ys)==0:
                    continue

                cx = int(xs.mean())
                cy = int(ys.mean())

                # ================= COUNT =================
                prev_c = prev_centroid.get(tid, None)

                if prev_c is not None and tid not in counted_ids:
                    prev_cx, prev_cy = prev_c

                    if crossed_in(prev_cx, prev_cy, cx, cy, LINE_POS, direction):
                        count_in[cls_name] += 1
                        counted_ids.add(tid)

                prev_centroid[tid] = (cx, cy)

                # ================= DRAW =================
                color = get_class_color(cls_name)
                # mask_bool = mask > 0.5

                # frame[mask_bool] = (
                #     frame[mask_bool] * 0.5 + np.array(color) * 0.5
                # ).astype(np.uint8)

                cv2.circle(frame, (cx, cy), 4, (0,0,255), -1)

                cv2.putText(frame,f"{cls_name}#{tid}",(x1,y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,color,2)

            # draw line
            if direction=="vertical":
                cv2.line(frame,(0,LINE_POS),(w,LINE_POS),(0,255,255),2)
            else:
                cv2.line(frame,(LINE_POS,0),(LINE_POS,h),(0,255,255),2)

            draw_panel(frame,count_in)

            if raw_writer is None:
                fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 25
                raw_writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,(w,h))
                ann_writer = cv2.VideoWriter(str(ann_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,(w,h))

            raw_writer.write(raw)
            ann_writer.write(frame)

            cv2.imshow("Output",frame)
            if cv2.waitKey(1)&0xFF in [27,ord('q')]:
                break

    except Exception as e:
        traceback.print_exc()

    finally:
        if raw_writer: raw_writer.release()
        if ann_writer: ann_writer.release()
        cv2.destroyAllWindows()

        print("\nFINAL COUNT:")
        for k in count_in:
            print(f"{k} -> IN:{count_in[k]}")

# ================= CLI =================
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.45)
    parser.add_argument("--device", default="")
    parser.add_argument("--project", default="runs/seg")
    parser.add_argument("--name", default="exp")
    parser.add_argument("--direction", default="vertical",
                        choices=["vertical","horizontal"])
    return parser.parse_args()

if __name__ == "__main__":
    opt = parse_opt()
    run(**vars(opt))
