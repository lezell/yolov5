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
    # imgsz can be int or tuple; normalize to (h, w)
    if isinstance(imgsz, int):
        imgsz = (imgsz, imgsz)
    else:
        assert len(imgsz) == 2, "imgsz must be int or (h, w)"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)     # HWC RGB
    # Force exact square output of (imgsz[0], imgsz[1]) that’s stride-aligned
    img, _, _ = letterbox(img_rgb, new_shape=imgsz, stride=stride, auto=False)
    img = img.transpose(2, 0, 1)                           # HWC -> CHW
    img = np.ascontiguousarray(img)
    im = torch.from_numpy(img).float() / 255.0             # CHW float32 [0,1]
    im = im.unsqueeze(0)                                   # BCHW
    return im


class MyYolo:
    def __init__(self, weights="runs/train/exp8/weights/best.pt", device=None):
        self.device = device if device is not None else _auto_device()
        self.model = DetectMultiBackend(weights, device=self.device)
        # Ensure detect head can rebuild grids dynamically
        try:
            m = getattr(self.model, "model", None)
            if m and hasattr(m, "model") and len(m.model):
                detect = m.model[-1]
                if hasattr(detect, "export"):
                    detect.export = False
        except Exception:
            pass
        # Warmup once at the final size you’ll use
        self.stride = get_stride(self.model)
        self.imgsz = (640, 640)
        try:
            self.model.warmup(imgsz=(1, 3, self.imgsz[0], self.imgsz[1]))
        except Exception:
            pass

    def process(self, frame_bgr, imgsz=640, conf_thres=0.25, iou_thres=0.45):
        # Always force square imgsz
        if isinstance(imgsz, int):
            imgsz = (imgsz, imgsz)
        stride = self.stride if hasattr(self, "stride") else get_stride(self.model)

        im = preprocess_bchw(frame_bgr, imgsz, stride).to(self.device)
        # Guardrail: ensure BCHW and square
        assert im.ndim == 4 and im.shape[2] == imgsz[0] and im.shape[3] == imgsz[1], f"Bad shape: {im.shape}"

        with torch.no_grad():
            preds = self.model(im)  # raw predictions

        from utils.general import non_max_suppression, scale_boxes
        dets = non_max_suppression(preds, conf_thres, iou_thres)[0]

        if dets is not None and len(dets):
            dets[:, :4] = scale_boxes(im.shape[2:], dets[:, :4], frame_bgr.shape).round()

        return dets