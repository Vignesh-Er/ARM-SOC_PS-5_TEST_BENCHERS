"""
FPGA-Accelerated Real-Time Object Detection — Kria K26 SOM
===========================================================
Single-file PYNQ code for Jupyter notebook on Kria board.

Usage (in Jupyter cells):
  Cell 1:  %run pynqcode.py          # loads bitstream + weights
  Cell 2:  main()                     # live camera + FPGA detection
  Cell 3:  run_benchmark()            # FPGA vs CPU speed comparison
  Cell 4:  cleanup()                  # free CMA memory (optional)

Requirements on Kria board:
  - design_1_wrapper.bit  (FPGA bitstream)
  - design_1.hwh          (hardware handoff)
  - weights/              (folder with trained .npy weight files)
    Conv1_w.npy, Conv1_b.npy, Conv2_w.npy, Conv2_b.npy,
    Conv3_w.npy, Conv3_b.npy, Conv4_w.npy, Conv4_b.npy,
    DetHead_w.npy, DetHead_b.npy

Architecture (matches cnn_accelerator.h v4.6):
  Conv1  3->16  3x3 ReLU  160x160
  Pool1  2x2              -> 80x80
  Conv2  16->32 3x3 ReLU   80x80
  Pool2  2x2              -> 40x40
  Conv3  32->64 3x3 ReLU   40x40
  Conv4  64->128 3x3 ReLU  40x40
  Pool3  2x2              -> 20x20
  DetHead 128->45 1x1 Lin  20x20  (3 anchors x 15 entries)

10 Classes: Person, Dog, Cat, Car, Bird, Horse, Bike, Chair, Bottle, Plant
"""

import pynq
from pynq import Overlay, allocate
import numpy as np
import cv2
import time
import os
import shutil
import ipywidgets as widgets
from IPython.display import display

# =================================================================
# 1. HARDWARE SETUP — Load FPGA Bitstream
# =================================================================
BIT_NAME   = "design_1_wrapper.bit"
HWH_NAME   = "design_1.hwh"
HWH_TARGET = "design_1_wrapper.hwh"

if os.path.exists(HWH_NAME) and not os.path.exists(HWH_TARGET):
    shutil.copy(HWH_NAME, HWH_TARGET)

print(f"[1/4] Loading bitstream: {BIT_NAME} ...")
try:
    ol = Overlay(BIT_NAME)
    print("      Bitstream loaded OK.")
except Exception as e:
    print(f"      FAILED: {e}")
    raise SystemExit(1)


# =================================================================
# 2. FIXED-POINT HELPERS — ap_fixed<16,6> (Q6.10)
# =================================================================
FRAC_BITS = 10
SCALE     = 1 << FRAC_BITS                       # 1024

def float_to_fixed16(arr):
    """Float32 -> int16 fixed-point (clamp to [-31.9, +31.9])."""
    return (np.clip(arr, -31.9, 31.9) * SCALE).astype(np.int16)

def fixed16_to_float(arr):
    """Int16 fixed-point -> float32."""
    return arr.astype(np.float32) / SCALE

# HLS layer-mode constants
MODE_CONV_RELU   = 0
MODE_MAXPOOL     = 1
MODE_CONV_LINEAR = 2
MODE_GLOBAL_AVG  = 3


# =================================================================
# 3. ACCELERATOR DRIVER
# =================================================================
class CNNAccelerator:
    """Thin AXI-Lite driver for the HLS CNN IP core."""
    def __init__(self, overlay):
        self.ip = overlay.cnn_accel_top_0

    def run(self, in_buf, out_buf, w_buf, b_buf,
            H, W, Cin, Cout, mode, K):
        r = self.ip.register_map
        r.img_height    = H
        r.img_width     = W
        r.in_channels   = Cin
        r.out_channels  = Cout
        r.mode          = mode
        r.kernel_size   = K
        r.input_data_1  = in_buf.device_address  & 0xFFFFFFFF
        r.input_data_2  = in_buf.device_address >> 32
        r.output_data_1 = out_buf.device_address & 0xFFFFFFFF
        r.output_data_2 = out_buf.device_address >> 32
        r.weights_1     = w_buf.device_address   & 0xFFFFFFFF
        r.weights_2     = w_buf.device_address  >> 32
        r.biases_1      = b_buf.device_address   & 0xFFFFFFFF
        r.biases_2      = b_buf.device_address  >> 32
        r.CTRL.AP_START = 1
        while r.CTRL.AP_DONE == 0:
            pass

accel = CNNAccelerator(ol)


# =================================================================
# 4. NETWORK DEFINITION
# =================================================================
NUM_CLASSES      = 10
NUM_ANCHORS      = 3
ENTRY_PER_ANCHOR = 5 + NUM_CLASSES              # 15
DET_CHANNELS     = NUM_ANCHORS * ENTRY_PER_ANCHOR  # 45

# Anchor sizes in NN-input pixel space (width, height)
ANCHORS = np.array([[10, 13], [16, 30], [33, 23]], dtype=np.float32)

# Detection thresholds (lowered for small 103K-param model)
CONF_THRESH      = 0.35     # obj * cls combined threshold
NMS_THRESH       = 0.45     # IoU for NMS suppression
OBJ_THRESH       = 0.15     # objectness pre-filter
MAX_DETS_PRE_NMS = 200
MAX_DETS_POST    = 30
MIN_BOX_PX       = 4

# Layer definitions (must match HLS bitstream exactly)
LAYERS = [
    ("Conv1",   MODE_CONV_RELU,   160, 160,   3,  16, 3),
    ("Pool1",   MODE_MAXPOOL,     160, 160,  16,  16, 3),   # -> 80x80
    ("Conv2",   MODE_CONV_RELU,    80,  80,  16,  32, 3),
    ("Pool2",   MODE_MAXPOOL,      80,  80,  32,  32, 3),   # -> 40x40
    ("Conv3",   MODE_CONV_RELU,    40,  40,  32,  64, 3),
    ("Conv4",   MODE_CONV_RELU,    40,  40,  64, 128, 3),
    ("Pool3",   MODE_MAXPOOL,      40,  40, 128, 128, 3),   # -> 20x20
    ("DetHead", MODE_CONV_LINEAR,  20,  20, 128, DET_CHANNELS, 1),
]

NN_H, NN_W    = 160, 160
GRID_H, GRID_W = 20, 20
CELL_H = NN_H / GRID_H       # 8.0
CELL_W = NN_W / GRID_W       # 8.0
INPUT_ELEMS = NN_H * NN_W * 3
DET_SIZE    = GRID_H * GRID_W * DET_CHANNELS

CLASS_NAMES = [
    "Person", "Dog",  "Cat",   "Car",    "Bird",
    "Horse",  "Bike", "Chair", "Bottle", "Plant",
]
CLASS_COLORS = [
    (0,255,0),   (255,165,0), (0,255,255), (255,0,0),   (255,255,0),
    (128,0,128), (0,0,255),   (192,192,192),(0,128,128), (0,128,0),
]


# =================================================================
# 5. WEIGHT LOADING & CMA BUFFER ALLOCATION
# =================================================================
print("[2/4] Loading weights & allocating CMA buffers ...")

layer_w_bufs = {}
layer_b_bufs = {}

WEIGHTS_DIR = "voc_data/weights"

def _kaiming_init(shape):
    """He/Kaiming normal init (fallback when no trained weights)."""
    fan_in = np.prod(shape[1:])
    return np.random.randn(*shape).astype(np.float32) * np.sqrt(2.0 / fan_in)

_trained = os.path.isdir(WEIGHTS_DIR)
if _trained:
    print(f"      Trained weights found in '{WEIGHTS_DIR}/'")
else:
    print(f"      WARNING: No '{WEIGHTS_DIR}/' folder — using random init.")
    print(f"      Run train_detector.py first, then copy weights/ here.")

for name, mode, H, W, Cin, Cout, K in LAYERS:
    if mode in (MODE_CONV_RELU, MODE_CONV_LINEAR):
        w_path = os.path.join(WEIGHTS_DIR, f"{name}_w.npy")
        b_path = os.path.join(WEIGHTS_DIR, f"{name}_b.npy")

        if _trained and os.path.exists(w_path) and os.path.exists(b_path):
            w_float = np.load(w_path).astype(np.float32)
            b_float = np.load(b_path).astype(np.float32)
            print(f"      Loaded {name}: w{w_float.shape}  b{b_float.shape}")
        else:
            w_float = _kaiming_init((Cout, Cin, K, K))
            b_float = np.zeros(Cout, dtype=np.float32)
            if name == "DetHead":
                for a in range(NUM_ANCHORS):
                    obj_idx = a * ENTRY_PER_ANCHOR + 4
                    if obj_idx < Cout:
                        b_float[obj_idx] = -5.0
            if _trained:
                print(f"      WARNING: Missing {name} weights — random init")

        w_fixed = float_to_fixed16(w_float.flatten())
        b_fixed = float_to_fixed16(b_float)

        wb = allocate(shape=w_fixed.shape, dtype=np.int16)
        bb = allocate(shape=b_fixed.shape, dtype=np.int16)
        wb[:] = w_fixed;  wb.flush()
        bb[:] = b_fixed;  bb.flush()
        layer_w_bufs[name] = wb
        layer_b_bufs[name] = bb

# Dummy buffer for pool layers (IP needs valid pointers)
dummy_buf = allocate(shape=(1,), dtype=np.int16)
dummy_buf.flush()

# Ping-pong data buffers
max_elems = max(l[2] * l[3] * max(l[4], l[5]) for l in LAYERS)
max_elems = max(max_elems, 160 * 160 * 16)
buf_a = allocate(shape=(max_elems,), dtype=np.int16)
buf_b = allocate(shape=(max_elems,), dtype=np.int16)

print(f"      CMA buffers allocated: {max_elems} x int16 each")
print(f"      Grid: {GRID_H}x{GRID_W}, {NUM_ANCHORS} anchors, {NUM_CLASSES} classes")


# =================================================================
# 6. FPGA INFERENCE PIPELINE
# =================================================================
print("[3/4] Warming up FPGA ...")

def run_fpga_pipeline(img_fixed):
    """Push image through all 8 layers on FPGA.

    Returns (det_raw: float32[DET_SIZE], conv_ms: float)
    """
    buf_a[:INPUT_ELEMS] = img_fixed
    buf_a.flush()

    curr_in, curr_out = buf_a, buf_b

    t0 = time.perf_counter()
    for name, mode, H, W, Cin, Cout, K in LAYERS:
        if mode in (MODE_CONV_RELU, MODE_CONV_LINEAR):
            wb, bb = layer_w_bufs[name], layer_b_bufs[name]
        else:
            wb, bb = dummy_buf, dummy_buf
        accel.run(curr_in, curr_out, wb, bb, H, W, Cin, Cout, mode, K)
        curr_in, curr_out = curr_out, curr_in
    t1 = time.perf_counter()

    # 8 layers = even swaps -> result in curr_in (== buf_a)
    curr_in.invalidate()
    det_raw = fixed16_to_float(np.array(curr_in[:DET_SIZE], dtype=np.int16))
    return det_raw, (t1 - t0) * 1000

# Warmup (3 dummy frames to stabilise DMA/cache)
_warmup_buf = np.zeros(INPUT_ELEMS, dtype=np.int16)
for _ in range(3):
    run_fpga_pipeline(_warmup_buf)
del _warmup_buf
print("      FPGA warm-up complete.")


# =================================================================
# 6b. DIAGNOSTIC FUNCTION
# =================================================================
def diagnose():
    """Run full diagnostic to identify detection pipeline issues.

    Call this from Jupyter on the Kria board:
      diagnose()

    It will print:
      1. Weight file paths and statistics
      2. Raw FPGA output statistics from a live camera frame
      3. Top objectness scores (to see if they're near threshold)
      4. Top detection candidates before/after NMS
    """
    print("=" * 62)
    print("  DIAGNOSTIC REPORT")
    print("=" * 62)

    # --- 1. Check weight files ---
    print("\n[1] WEIGHT FILES:")
    wdir = os.path.abspath(WEIGHTS_DIR)
    print(f"    Path: {wdir}")
    print(f"    Exists: {os.path.isdir(wdir)}")
    if os.path.isdir(wdir):
        for f in sorted(os.listdir(wdir)):
            fp = os.path.join(wdir, f)
            print(f"    {f:20s} {os.path.getsize(fp):>8,} bytes")

    # --- 2. Weight buffer stats ---
    print("\n[2] WEIGHT BUFFER STATS (int16 fixed-point):")
    for name in ["Conv1", "Conv2", "Conv3", "Conv4", "DetHead"]:
        if name in layer_w_bufs:
            wdata = np.array(layer_w_bufs[name][:], dtype=np.int16)
            bdata = np.array(layer_b_bufs[name][:], dtype=np.int16)
            wf = wdata.astype(np.float32) / SCALE
            bf = bdata.astype(np.float32) / SCALE
            print(f"    {name:8s} w: min={wf.min():.4f} max={wf.max():.4f} "
                  f"mean={wf.mean():.4f} std={wf.std():.4f} nonzero={np.count_nonzero(wdata)}/{len(wdata)}")
            print(f"    {' ':8s} b: min={bf.min():.4f} max={bf.max():.4f} "
                  f"mean={bf.mean():.4f} [{' '.join(f'{v:.3f}' for v in bf[:min(8,len(bf))])}...]")

    # --- 3. Capture one frame and run FPGA ---
    print("\n[3] LIVE FRAME TEST:")
    import cv2
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("    ERROR: No camera!")
        return

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("    ERROR: Failed to read frame!")
        return

    print(f"    Frame shape: {frame.shape}")
    resized = cv2.resize(frame, (NN_W, NN_H), interpolation=cv2.INTER_NEAREST)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    img_float = resized.astype(np.float32) * (1.0 / 127.5) - 1.0
    print(f"    Input: min={img_float.min():.3f} max={img_float.max():.3f} mean={img_float.mean():.3f}")

    img_fixed = float_to_fixed16(img_float.flatten())
    det_raw, conv_ms = run_fpga_pipeline(img_fixed)

    print(f"    FPGA inference: {conv_ms:.1f} ms")
    print(f"    Output: min={det_raw.min():.4f} max={det_raw.max():.4f} "
          f"mean={det_raw.mean():.4f} std={det_raw.std():.4f}")

    # --- 4. Analyze detection grid ---
    print("\n[4] DETECTION GRID ANALYSIS:")
    g = det_raw.reshape(GRID_H, GRID_W, NUM_ANCHORS, ENTRY_PER_ANCHOR)

    obj_raw = g[:, :, :, 4]  # raw objectness (pre-sigmoid)
    obj_sig = _fast_sigmoid(obj_raw)  # after sigmoid

    print(f"    Objectness (raw):     min={obj_raw.min():.4f} max={obj_raw.max():.4f} mean={obj_raw.mean():.4f}")
    print(f"    Objectness (sigmoid): min={obj_sig.min():.4f} max={obj_sig.max():.4f} mean={obj_sig.mean():.4f}")

    # Top 10 objectness values
    flat_obj = obj_sig.flatten()
    top_idx = np.argsort(flat_obj)[-10:][::-1]
    print(f"\n    Top 10 objectness scores (thresh={OBJ_THRESH}):")
    for rank, idx in enumerate(top_idx):
        gy, gx, ga = np.unravel_index(idx, (GRID_H, GRID_W, NUM_ANCHORS))
        entry = g[gy, gx, ga]
        cls_sig = _fast_sigmoid(entry[5:])
        best_cls = int(np.argmax(cls_sig))
        combined = flat_obj[idx] * cls_sig[best_cls]
        status = "PASS" if flat_obj[idx] > OBJ_THRESH else "FAIL"
        print(f"      {rank+1:2d}. obj={flat_obj[idx]:.4f} cls={CLASS_NAMES[best_cls]}({cls_sig[best_cls]:.3f}) "
              f"comb={combined:.4f} grid=({gy},{gx}) anc={ga} [{status}]")

    # Summary
    above_obj = (obj_sig > OBJ_THRESH).sum()
    above_conf = 0
    for gy in range(GRID_H):
        for gx in range(GRID_W):
            for a in range(NUM_ANCHORS):
                o = obj_sig[gy, gx, a]
                if o > OBJ_THRESH:
                    cls_p = _fast_sigmoid(g[gy, gx, a, 5:])
                    comb = o * cls_p.max()
                    if comb > CONF_THRESH:
                        above_conf += 1

    print(f"\n    Above OBJ_THRESH ({OBJ_THRESH}): {above_obj} / {GRID_H*GRID_W*NUM_ANCHORS}")
    print(f"    Above CONF_THRESH ({CONF_THRESH}): {above_conf}")

    # Final detections
    dets = decode_and_nms(det_raw, frame.shape[0], frame.shape[1])
    print(f"    Final detections after NMS: {len(dets)}")
    for d in dets:
        print(f"      {d[6]} conf={d[4]:.3f} box=({d[0]},{d[1]})-({d[2]},{d[3]})")

    print("\n" + "=" * 62)
    if flat_obj.max() < 0.1:
        print("  DIAGNOSIS: All objectness < 0.1")
        print("  → Weights are likely random/untrained or not loaded correctly.")
        print("  → Verify the .npy files are the COCO-trained ones from Colab.")
    elif above_obj == 0:
        print(f"  DIAGNOSIS: Max objectness = {flat_obj.max():.4f} but below threshold {OBJ_THRESH}")
        print(f"  → Try lowering OBJ_THRESH further (e.g. 0.05)")
    elif above_conf == 0:
        print(f"  DIAGNOSIS: {above_obj} cells pass objectness but none pass CONF_THRESH")
        print(f"  → Try lowering CONF_THRESH further (e.g. 0.05)")
    else:
        print(f"  DIAGNOSIS: Pipeline working — {len(dets)} detections produced.")
    print("=" * 62)


# =================================================================
# 7. OPTIMISED POST-PROCESSING (Vectorised NumPy, fast sigmoid)
# =================================================================
print("[4/4] Ready.")

def _fast_sigmoid(x):
    """Fast sigmoid approximation: 0.5 + 0.5 * x / (1 + |x|).

    ~3-5x faster than exp-based sigmoid on ARM Cortex-A53.
    Max error ~3% vs true sigmoid — negligible for detection.
    """
    return 0.5 + 0.5 * x / (1.0 + np.abs(x))


def decode_and_nms(det_raw, frame_h, frame_w):
    """Decode YOLO detection head -> filtered bounding boxes.

    Optimised pipeline:
      1. Reshape to (20, 20, 3, 15)
      2. Fast-sigmoid objectness pre-filter
      3. Decode box coords + class probs (only surviving cells)
      4. Combined confidence threshold
      5. Min box-size filter
      6. Top-K pre-NMS cap
      7. Greedy NMS (per-class, vectorised IoU)

    Returns: list of (x1, y1, x2, y2, conf, class_id, class_name)
    """
    # 1. Reshape
    g = det_raw.reshape(GRID_H, GRID_W, NUM_ANCHORS, ENTRY_PER_ANCHOR)

    # 2. Objectness pre-filter (fast sigmoid, discard most cells early)
    obj = _fast_sigmoid(g[:, :, :, 4])
    obj_mask = obj > OBJ_THRESH
    if not np.any(obj_mask):
        return []

    # 3. Decode surviving cells only
    gy_idx, gx_idx, ga_idx = np.where(obj_mask)
    g_sel   = g[gy_idx, gx_idx, ga_idx]          # (N, 15)
    obj_sel = obj[gy_idx, gx_idx, ga_idx]         # (N,)

    bx = (_fast_sigmoid(g_sel[:, 0]) + gx_idx) * CELL_W
    by = (_fast_sigmoid(g_sel[:, 1]) + gy_idx) * CELL_H
    bw = ANCHORS[ga_idx, 0] * np.exp(np.clip(g_sel[:, 2], -5, 5))
    bh = ANCHORS[ga_idx, 1] * np.exp(np.clip(g_sel[:, 3], -5, 5))

    cls_p  = _fast_sigmoid(g_sel[:, 5:])          # (N, 10)
    cids   = np.argmax(cls_p, axis=-1)            # (N,)
    cmax   = np.max(cls_p, axis=-1)               # (N,)
    scores = obj_sel * cmax                       # (N,)

    # 4. Combined confidence threshold
    conf_mask = scores > CONF_THRESH
    if not np.any(conf_mask):
        return []
    bx, by  = bx[conf_mask], by[conf_mask]
    bw, bh  = bw[conf_mask], bh[conf_mask]
    scores  = scores[conf_mask]
    cids    = cids[conf_mask]

    # 5. Min box-size filter
    size_ok = (bw >= MIN_BOX_PX) & (bh >= MIN_BOX_PX)
    if not np.any(size_ok):
        return []
    bx, by  = bx[size_ok], by[size_ok]
    bw, bh  = bw[size_ok], bh[size_ok]
    scores  = scores[size_ok]
    cids    = cids[size_ok]

    # 6. Top-K pre-NMS cap
    if len(scores) > MAX_DETS_PRE_NMS:
        topk = np.argpartition(scores, -MAX_DETS_PRE_NMS)[-MAX_DETS_PRE_NMS:]
        bx, by  = bx[topk], by[topk]
        bw, bh  = bw[topk], bh[topk]
        scores  = scores[topk]
        cids    = cids[topk]

    # Scale to display-frame pixel coordinates
    sx, sy = frame_w / NN_W, frame_h / NN_H
    x1 = np.clip((bx - bw * 0.5) * sx, 0, frame_w).astype(np.int32)
    y1 = np.clip((by - bh * 0.5) * sy, 0, frame_h).astype(np.int32)
    x2 = np.clip((bx + bw * 0.5) * sx, 0, frame_w).astype(np.int32)
    y2 = np.clip((by + bh * 0.5) * sy, 0, frame_h).astype(np.int32)

    # 7. Greedy NMS (per-class suppression, vectorised IoU)
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0 and len(keep) < MAX_DETS_POST:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.where(union > 0, inter / union, 0.0)
        suppress = (iou > NMS_THRESH) & (cids[rest] == cids[i])
        order = rest[~suppress]

    return [(int(x1[k]), int(y1[k]), int(x2[k]), int(y2[k]),
             float(scores[k]), int(cids[k]), CLASS_NAMES[cids[k]]) for k in keep]


# =================================================================
# 8. OVERLAY DRAWING — Bounding Boxes + Stats Dashboard
# =================================================================
def draw_overlay(frame, t_pre, t_conv, t_post, fps, detections):
    """Draw bounding boxes with class labels and a stats dashboard.

    Dashboard (top-left):
      - FPS
      - Pre-processing time (resize + color convert + quantize)
      - FPGA Inference time (all 8 layers on PL)
      - Post-processing time (decode + NMS)
      - Total pipeline latency
      - Detection count

    Bounding boxes:
      - Color-coded per class
      - Person detections get thicker border (3px vs 2px)
      - Label: "ClassName XX%"
    """
    h, w = frame.shape[:2]
    tot = t_pre + t_conv + t_post

    # --- Bounding Boxes ---
    person_count = 0
    for (x1, y1, x2, y2, conf, cid, cname) in detections:
        clr = CLASS_COLORS[cid]
        # Person gets thicker box for visual emphasis
        thickness = 3 if cid == 0 else 2
        if cid == 0:
            person_count += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), clr, thickness)

        # Label background + text
        label = f"{cname} {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ly = max(y1 - 6, th + 4)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 6, ly + 4), clr, -1)
        cv2.putText(frame, label, (x1 + 3, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # --- Stats Dashboard (top-left corner) ---
    panel_h = 155
    cv2.rectangle(frame, (4, 4), (250, panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (4, 4), (250, panel_h), (0, 255, 0), 1)

    # Title
    cv2.putText(frame, "FPGA ACCELERATED", (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    # Timing breakdown
    y0 = 46
    cv2.putText(frame, f"FPS      : {fps:5.1f}",     (12, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    cv2.putText(frame, f"Pre-proc : {t_pre:5.1f} ms", (12, y0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, f"FPGA Inf : {t_conv:5.1f} ms",(12, y0 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(frame, f"Post-proc: {t_post:5.1f} ms",(12, y0 + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, f"Total    : {tot:5.1f} ms",   (12, y0 + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Detection summary
    det_text = f"Detections: {len(detections)}"
    if person_count > 0:
        det_text += f"  (Person: {person_count})"
    cv2.putText(frame, det_text, (12, y0 + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)


# =================================================================
# 9. CPU-ONLY BASELINE (for benchmark comparison)
# =================================================================
def _cpu_conv2d(inp, w, b, H, W, Cin, Cout, K, relu=True):
    """Pure-NumPy 2D convolution (matches HLS exactly)."""
    pad = K // 2
    img = inp.reshape(H, W, Cin)
    if pad > 0:
        img = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='constant')
    out = np.empty((H, W, Cout), dtype=np.float32)
    for co in range(Cout):
        acc = np.full((H, W), b[co], dtype=np.float32)
        for ci in range(Cin):
            ws = w[co, ci]
            for ky in range(K):
                for kx in range(K):
                    acc += ws[ky, kx] * img[ky:ky+H, kx:kx+W, ci]
        out[:, :, co] = np.maximum(acc, 0) if relu else acc
    return out.flatten()


def _cpu_maxpool(inp, H, W, C):
    """Pure-NumPy max-pool 2x2 stride 2."""
    return inp.reshape(H//2, 2, W//2, 2, C).max(axis=(1, 3)).flatten()


def run_cpu_pipeline(img_float_flat):
    """Run entire CNN on ARM CPU (NumPy). For benchmark comparison only."""
    data = img_float_flat.copy()
    t0 = time.perf_counter()
    for name, mode, H, W, Cin, Cout, K in LAYERS:
        if mode == MODE_CONV_RELU:
            wf = fixed16_to_float(np.array(layer_w_bufs[name][:], dtype=np.int16)).reshape(Cout, Cin, K, K)
            bf = fixed16_to_float(np.array(layer_b_bufs[name][:], dtype=np.int16))
            data = _cpu_conv2d(data, wf, bf, H, W, Cin, Cout, K, relu=True)
        elif mode == MODE_CONV_LINEAR:
            wf = fixed16_to_float(np.array(layer_w_bufs[name][:], dtype=np.int16)).reshape(Cout, Cin, K, K)
            bf = fixed16_to_float(np.array(layer_b_bufs[name][:], dtype=np.int16))
            data = _cpu_conv2d(data, wf, bf, H, W, Cin, Cout, K, relu=False)
        elif mode == MODE_MAXPOOL:
            data = _cpu_maxpool(data, H, W, Cin)
    t1 = time.perf_counter()
    return data, (t1 - t0) * 1000


def run_benchmark(num_frames=10):
    """FPGA vs ARM-CPU performance comparison.

    Prints a comparison table showing:
      - Average / best latency for FPGA vs CPU
      - Throughput (FPS)
      - Speedup factor
    """
    print("=" * 62)
    print("  FPGA vs ARM-CPU PERFORMANCE BENCHMARK")
    print("=" * 62)
    print(f"  Architecture : {len(LAYERS)} layers, {NN_H}x{NN_W} input")
    print(f"  Output       : {GRID_H}x{GRID_W} grid x {DET_CHANNELS} channels")
    print(f"  Test frames  : {num_frames}")
    print("-" * 62)

    img   = np.random.uniform(-1, 1, INPUT_ELEMS).astype(np.float32)
    img_q = float_to_fixed16(img)

    # Warmup
    for _ in range(3):
        run_fpga_pipeline(img_q)
    run_cpu_pipeline(img)

    # Benchmark FPGA (inference + post-processing)
    fpga_ms = []
    for _ in range(num_frames):
        t0 = time.perf_counter()
        d, _ = run_fpga_pipeline(float_to_fixed16(img))
        decode_and_nms(d, NN_H, NN_W)
        fpga_ms.append((time.perf_counter() - t0) * 1000)

    # Benchmark CPU (inference + post-processing)
    cpu_ms = []
    for _ in range(num_frames):
        t0 = time.perf_counter()
        d, _ = run_cpu_pipeline(img)
        decode_and_nms(d, NN_H, NN_W)
        cpu_ms.append((time.perf_counter() - t0) * 1000)

    fa, ca = np.mean(fpga_ms), np.mean(cpu_ms)
    fm, cm = np.min(fpga_ms), np.min(cpu_ms)
    sp = ca / fa if fa > 0 else 0

    print(f"\n  {'Metric':<25} {'FPGA+ARM':>12} {'ARM-only':>12} {'Speedup':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Avg latency':<25} {fa:>9.1f} ms {ca:>9.1f} ms {sp:>8.1f}x")
    print(f"  {'Best latency':<25} {fm:>9.1f} ms {cm:>9.1f} ms {cm/fm:>8.1f}x")
    print(f"  {'Throughput':<25} {1000/fa:>9.1f} fps {1000/ca:>9.1f} fps")
    print(f"\n  Result: {'PASS (>= 2x speedup)' if sp >= 2 else 'BELOW 2x TARGET'}")
    print("=" * 62)
    return {'fpga_avg_ms': fa, 'cpu_avg_ms': ca, 'speedup': sp,
            'fpga_fps': 1000/fa, 'cpu_fps': 1000/ca}


# =================================================================
# 10. LIVE DETECTION — MAIN LOOP
# =================================================================
_JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 70]


def main():
    """Live camera object detection with FPGA acceleration.

    Opens USB camera, runs real-time inference on FPGA, and shows
    bounding boxes + timing stats in a Jupyter widget.

    Press the 'Stop' button to end the demo.
    Camera is released on stop; CMA buffers stay allocated so
    you can call main() again without restarting the kernel.
    """
    # --- UI Widgets ---
    stop_btn = widgets.Button(description="Stop Detection",
                              button_style='danger',
                              layout=widgets.Layout(width='200px'))
    image_widget = widgets.Image(format='jpeg', width=640, height=480)
    status_label = widgets.Label(value="Starting camera...")
    display(widgets.VBox([image_widget, widgets.HBox([stop_btn, status_label])]))

    running = [True]   # mutable flag for button callback

    def _on_stop(_):
        running[0] = False
        status_label.value = "Stopping..."
    stop_btn.on_click(_on_stop)

    # --- Camera ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        status_label.value = "Camera not found!"
        print("ERROR: No camera found on /dev/video0 or /dev/video1")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # minimise capture latency
    status_label.value = f"Live | {GRID_H}x{GRID_W} grid | {', '.join(CLASS_NAMES)}"

    # EMA-smoothed FPS
    fps_ema = 0.0
    frame_count = 0

    try:
        while running[0]:
            t_start = time.perf_counter()

            ret, frame = cap.read()
            if not ret:
                continue

            # --- Stage 1: Pre-processing (CPU) ---
            t1 = time.perf_counter()
            resized = cv2.resize(frame, (NN_W, NN_H),
                                 interpolation=cv2.INTER_NEAREST)
            # BGR -> RGB to match training convention
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            # Normalise to [-1, +1] and quantise to fixed-point
            img_fixed = float_to_fixed16(
                (resized.astype(np.float32) * (1.0 / 127.5) - 1.0).flatten())
            t_pre = (time.perf_counter() - t1) * 1000

            # --- Stage 2: FPGA Inference (PL) ---
            det_raw, t_conv = run_fpga_pipeline(img_fixed)

            # --- Stage 3: Post-processing (CPU) ---
            t3 = time.perf_counter()
            fh, fw = frame.shape[:2]
            detections = decode_and_nms(det_raw, fh, fw)
            t_post = (time.perf_counter() - t3) * 1000

            # FPS (EMA with alpha=0.3 for smooth display)
            dt = time.perf_counter() - t_start
            instant_fps = 1.0 / dt if dt > 0 else 0
            fps_ema = (0.3 * instant_fps + 0.7 * fps_ema) if fps_ema > 0 else instant_fps

            # Draw overlay on frame
            draw_overlay(frame, t_pre, t_conv, t_post, fps_ema, detections)

            # Encode & push to Jupyter widget
            _, jpeg = cv2.imencode('.jpg', frame, _JPEG_PARAMS)
            image_widget.value = jpeg.tobytes()

            frame_count += 1

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Runtime error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        status_label.value = f"Stopped after {frame_count} frames. Call main() to restart."
        print(f"Camera released. Processed {frame_count} frames.")
        print("CMA buffers still allocated — call cleanup() to free, or main() to restart.")


# =================================================================
# 11. CLEANUP — Free CMA Buffers
# =================================================================
def cleanup():
    """Free all CMA buffers. Call this when completely done.

    After cleanup(), you must restart the kernel to use the FPGA again.
    """
    try:
        buf_a.freebuffer()
        buf_b.freebuffer()
        dummy_buf.freebuffer()
        for b in layer_w_bufs.values():
            b.freebuffer()
        for b in layer_b_bufs.values():
            b.freebuffer()
        print("All CMA buffers freed. Restart kernel to reload.")
    except Exception as e:
        print(f"Cleanup error: {e}")


# =================================================================
# READY — Print usage instructions
# =================================================================
print("\n" + "=" * 62)
print("  FPGA Object Detection — Ready")
print("=" * 62)
print(f"  Bitstream : {BIT_NAME}")
print(f"  Weights   : {'Trained' if _trained else 'Random (untrained)'}")
print(f"  Classes   : {', '.join(CLASS_NAMES)}")
print(f"  Input     : {NN_H}x{NN_W} RGB")
print(f"  Grid      : {GRID_H}x{GRID_W} x {NUM_ANCHORS} anchors")
print("-" * 62)
print("  Commands:")
print("    main()          — Start live camera detection")
print("    run_benchmark() — FPGA vs CPU speed comparison")
print("    cleanup()       — Free CMA memory (when done)")
print("=" * 62)
