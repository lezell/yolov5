import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=".*common.*")

import torch
from torch.serialization import add_safe_globals
from models.common import DetectMultiBackend

import cv2
import numpy as np
import torch
from utils.augmentations import letterbox

# --- Allow-list YOLOv5 model classes for PyTorch 2.6+ safe unpickling ---
try:
    from models import yolo as _y
    _allow = [getattr(_y, n) for n in ("DetectionModel", "SegmentationModel", "ClassificationModel") if hasattr(_y, n)]
    if _allow:
        add_safe_globals(_allow)
except Exception:
    pass
# -------------------------------------------------------------------------

def _auto_device():
    # Minimal, robust chooser (avoids utils.select_device banner + file_date)
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    # MPS (Apple) if you care:
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_stride(m):
    s = getattr(m, "stride", 32)
    # tensor -> max; list/tuple -> max; int -> itself
    try:
        import torch
        if hasattr(s, "numel"):              # torch.Tensor-like
            return int(s.max()) if s.numel() > 1 else int(s)
    except Exception:
        pass
    if isinstance(s, (list, tuple)):
        return int(max(s))
    return int(s)



def preprocess_bchw(img_bgr, imgsz, stride):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)     # HWC RGB
    img = letterbox(img_rgb, new_shape=imgsz, stride=stride, auto=True)[0]
    img = img.transpose(2, 0, 1)                            # HWC -> CHW
    img = np.ascontiguousarray(img)
    im = torch.from_numpy(img).float() / 255.0              # CHW float32 [0,1]
    im = im.unsqueeze(0)                                    # BCHW
    return im

# class MyYolo:
#     def __init__(self, weights="runs/train/exp8/weights/best.pt", device=None):
#         self.device = device if device is not None else _auto_device()
#         self.model = DetectMultiBackend(weights, device=self.device)
#     def process(self, frame):
#         results = self.model(frame)
#         # annotated_frame = results.render()[0]
#         # annotated_frame = annotated_frame.copy()
#         # detections = results.pandas().xyxy[0]
#         return results

class MyYolo:
    def __init__(self, weights="runs/train/exp8/weights/best.pt", device=None):
        self.device = device if device is not None else _auto_device()
        self.model = DetectMultiBackend(weights, device=self.device)

    def process(self, frame_bgr, imgsz=640, conf_thres=0.25, iou_thres=0.45):
        stride = get_stride(self.model)
        im = preprocess_bchw(frame_bgr, imgsz, stride).to(self.device)

        with torch.no_grad():
            preds = self.model(im)                           # raw predictions (not a Results object)

        # If you want boxes, run NMS + rescale:
        from utils.general import non_max_suppression, scale_boxes
        dets = non_max_suppression(preds, conf_thres, iou_thres)[0]  # for single image

        # Rescale to original frame coords
        if dets is not None and len(dets):
            dets[:, :4] = scale_boxes(im.shape[2:], dets[:, :4], frame_bgr.shape).round()

        return dets  # tensor [N,6]: x1,y1,x2,y2,conf,cls  (not a hub.Results)