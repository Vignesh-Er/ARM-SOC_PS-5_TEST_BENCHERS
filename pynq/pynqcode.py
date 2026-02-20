"""
FPGA-Accelerated Real-Time Object Detection — Kria KV260 (FINAL SUBMISSION)
===========================================================
Uses high-accuracy pre-trained 10-class weights for the FPGA to ensure detection, 
but strictly filters the software output to ONLY display the "Person" class 
to meet the project's single-class demo requirements.
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
# 2. FIXED-POINT HELPERS — ap_fixed<16,10> (Q6.10)
# =================================================================
FRAC_BITS = 10
SCALE     = 1 << FRAC_BITS                       

def float_to_fixed16(arr):
    return (np.clip(arr, -31.9, 31.9) * SCALE).astype(np.int16)

def fixed16_to_float(arr):
    return arr.astype(np.float32) / SCALE

MODE_CONV_RELU   = 0
MODE_MAXPOOL     = 1
MODE_CONV_LINEAR = 2


# =================================================================
# 3. ACCELERATOR DRIVER
# =================================================================
class CNNAccelerator:
    def __init__(self, overlay):
        self.ip = overlay.cnn_accel_top_0

    def run(self, in_buf, out_buf, w_buf, b_buf, H, W, Cin, Cout, mode, K):
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
# 4. NETWORK DEFINITION (10-Class structure for Smart Weights)
# =================================================================
NUM_CLASSES      = 10
NUM_ANCHORS      = 3
ENTRY_PER_ANCHOR = 5 + NUM_CLASSES              
DET_CHANNELS     = NUM_ANCHORS * ENTRY_PER_ANCHOR  # 45

ANCHORS = np.array([[10, 13], [16, 30], [33, 23]], dtype=np.float32)

# Healthy thresholds because these weights are fully trained
CONF_THRESH      = 0.40     
OBJ_THRESH       = 0.15     
NMS_THRESH       = 0.45     
MAX_DETS_PRE_NMS = 200
MAX_DETS_POST    = 30       
MIN_BOX_PX       = 4

LAYERS = [
    ("Conv1",   MODE_CONV_RELU,   160, 160,   3,  16, 3),
    ("Pool1",   MODE_MAXPOOL,     160, 160,  16,  16, 3),   
    ("Conv2",   MODE_CONV_RELU,    80,  80,  16,  32, 3),
    ("Pool2",   MODE_MAXPOOL,      80,  80,  32,  32, 3),   
    ("Conv3",   MODE_CONV_RELU,    40,  40,  32,  64, 3),
    ("Conv4",   MODE_CONV_RELU,    40,  40,  64, 128, 3),
    ("Pool3",   MODE_MAXPOOL,      40,  40, 128, 128, 3),   
    ("DetHead", MODE_CONV_LINEAR,  20,  20, 128, DET_CHANNELS, 1),
]

NN_H, NN_W    = 160, 160
GRID_H, GRID_W = 20, 20
CELL_H = NN_H / GRID_H       
CELL_W = NN_W / GRID_W       
INPUT_ELEMS = NN_H * NN_W * 3
DET_SIZE    = GRID_H * GRID_W * DET_CHANNELS

CLASS_NAMES = ["Person", "Dog", "Cat", "Car", "Bird", "Horse", "Bike", "Chair", "Bottle", "Plant"]


# =================================================================
# 5. LOAD SMART WEIGHTS (From your original voc_data/weights)
# =================================================================
print("[2/4] Loading SMART weights & allocating CMA buffers ...")

layer_w_bufs = {}
layer_b_bufs = {}

# Reverted to the folder containing your highly-trained weights
WEIGHTS_DIR = "voc_data/weights"

def _kaiming_init(shape):
    fan_in = np.prod(shape[1:])
    return np.random.randn(*shape).astype(np.float32) * np.sqrt(2.0 / fan_in)

_trained = os.path.isdir(WEIGHTS_DIR)

for name, mode, H, W, Cin, Cout, K in LAYERS:
    if mode in (MODE_CONV_RELU, MODE_CONV_LINEAR):
        w_path = os.path.join(WEIGHTS_DIR, f"{name}_w.npy")
        b_path = os.path.join(WEIGHTS_DIR, f"{name}_b.npy")

        if _trained and os.path.exists(w_path) and os.path.exists(b_path):
            w_float = np.load(w_path).astype(np.float32)
            b_float = np.load(b_path).astype(np.float32)
        else:
            w_float = _kaiming_init((Cout, Cin, K, K))
            b_float = np.zeros(Cout, dtype=np.float32)
            if name == "DetHead":
                for a in range(NUM_ANCHORS):
                    obj_idx = a * ENTRY_PER_ANCHOR + 4
                    if obj_idx < Cout: b_float[obj_idx] = -5.0

        w_fixed = float_to_fixed16(w_float.flatten())
        b_fixed = float_to_fixed16(b_float)

        wb = allocate(shape=w_fixed.shape, dtype=np.int16)
        bb = allocate(shape=b_fixed.shape, dtype=np.int16)
        wb[:] = w_fixed;  wb.flush()
        bb[:] = b_fixed;  bb.flush()
        layer_w_bufs[name] = wb
        layer_b_bufs[name] = bb

dummy_buf = allocate(shape=(1,), dtype=np.int16)
dummy_buf.flush()

max_elems = max(l[2] * l[3] * max(l[4], l[5]) for l in LAYERS)
max_elems = max(max_elems, 160 * 160 * 16)
buf_a = allocate(shape=(max_elems,), dtype=np.int16)
buf_b = allocate(shape=(max_elems,), dtype=np.int16)


# =================================================================
# 6. FPGA INFERENCE PIPELINE
# =================================================================
print("[3/4] Warming up FPGA ...")

def run_fpga_pipeline(img_fixed):
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

    curr_in.invalidate()
    det_raw = fixed16_to_float(np.array(curr_in[:DET_SIZE], dtype=np.int16))
    return det_raw, (t1 - t0) * 1000

_warmup_buf = np.zeros(INPUT_ELEMS, dtype=np.int16)
for _ in range(3): run_fpga_pipeline(_warmup_buf)
del _warmup_buf


# =================================================================
# 7. OPTIMISED POST-PROCESSING
# =================================================================
print("[4/4] Ready.")

def _fast_sigmoid(x):
    return 0.5 + 0.5 * x / (1.0 + np.abs(x))

def decode_and_nms(det_raw, frame_h, frame_w):
    g = det_raw.reshape(GRID_H, GRID_W, NUM_ANCHORS, ENTRY_PER_ANCHOR)
    obj = _fast_sigmoid(g[:, :, :, 4])
    obj_mask = obj > OBJ_THRESH
    if not np.any(obj_mask): return []

    gy_idx, gx_idx, ga_idx = np.where(obj_mask)
    g_sel   = g[gy_idx, gx_idx, ga_idx]
    obj_sel = obj[gy_idx, gx_idx, ga_idx]

    bx = (_fast_sigmoid(g_sel[:, 0]) + gx_idx) * CELL_W
    by = (_fast_sigmoid(g_sel[:, 1]) + gy_idx) * CELL_H
    bw = ANCHORS[ga_idx, 0] * np.exp(np.clip(g_sel[:, 2], -5, 5))
    bh = ANCHORS[ga_idx, 1] * np.exp(np.clip(g_sel[:, 3], -5, 5))

    cls_p  = _fast_sigmoid(g_sel[:, 5:])
    cids   = np.argmax(cls_p, axis=-1)
    cmax   = np.max(cls_p, axis=-1)
    scores = obj_sel * cmax

    conf_mask = scores > CONF_THRESH
    if not np.any(conf_mask): return []
    bx, by  = bx[conf_mask], by[conf_mask]
    bw, bh  = bw[conf_mask], bh[conf_mask]
    scores  = scores[conf_mask]
    cids    = cids[conf_mask]

    size_ok = (bw >= MIN_BOX_PX) & (bh >= MIN_BOX_PX) 
    
    if not np.any(size_ok): return []
    bx, by  = bx[size_ok], by[size_ok]
    bw, bh  = bw[size_ok], bh[size_ok]
    scores  = scores[size_ok]
    cids    = cids[size_ok]

    if len(scores) > MAX_DETS_PRE_NMS:
        topk = np.argpartition(scores, -MAX_DETS_PRE_NMS)[-MAX_DETS_PRE_NMS:]
        bx, by  = bx[topk], by[topk]
        bw, bh  = bw[topk], bh[topk]
        scores  = scores[topk]
        cids    = cids[topk]

    sx, sy = frame_w / NN_W, frame_h / NN_H
    x1 = np.clip((bx - bw * 0.5) * sx, 0, frame_w).astype(np.int32)
    y1 = np.clip((by - bh * 0.5) * sy, 0, frame_h).astype(np.int32)
    x2 = np.clip((bx + bw * 0.5) * sx, 0, frame_w).astype(np.int32)
    y2 = np.clip((by + bh * 0.5) * sy, 0, frame_h).astype(np.int32)

    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0 and len(keep) < MAX_DETS_POST:
        i = order[0]
        keep.append(i)
        if order.size == 1: break
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
# 8. OVERLAY DRAWING (FILTERED STRICTLY FOR "PERSON")
# =================================================================
def draw_overlay(frame, t_pre, t_conv, t_post, fps, detections):
    tot = t_pre + t_conv + t_post
    person_count = 0
    
    for (x1, y1, x2, y2, conf, cid, cname) in detections:
        # THE MAGIC FILTER: Ignore anything that isn't a Person (Class ID 0)
        if cid != 0:
            continue
            
        person_count += 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"Person {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        ly = max(y1 - 6, th + 4)
        cv2.rectangle(frame, (x1, ly - th - 4), (x1 + tw + 6, ly + 4), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1 + 3, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # Metrics Dashboard
    panel_h = 155
    cv2.rectangle(frame, (4, 4), (250, panel_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (4, 4), (250, panel_h), (0, 255, 0), 1)

    cv2.putText(frame, "FPGA ACCELERATED", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

    y0 = 46
    cv2.putText(frame, f"FPS      : {fps:5.1f}",      (12, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    cv2.putText(frame, f"Pre-proc : {t_pre:5.1f} ms", (12, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, f"FPGA Inf : {t_conv:5.1f} ms",(12, y0 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(frame, f"Post-proc: {t_post:5.1f} ms",(12, y0 + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.putText(frame, f"Total    : {tot:5.1f} ms",   (12, y0 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv2.putText(frame, f"Persons Detected: {person_count}", (12, y0 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)


# =================================================================
# 9. LIVE DETECTION — MAIN LOOP
# =================================================================
_JPEG_PARAMS = [int(cv2.IMWRITE_JPEG_QUALITY), 70]

def main():
    stop_btn = widgets.Button(description="Stop Detection", button_style='danger', layout=widgets.Layout(width='200px'))
    image_widget = widgets.Image(format='jpeg', width=640, height=480)
    status_label = widgets.Label(value="Starting camera...")
    display(widgets.VBox([image_widget, widgets.HBox([stop_btn, status_label])]))

    running = [True]
    def _on_stop(_):
        running[0] = False
        status_label.value = "Stopping..."
    stop_btn.on_click(_on_stop)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        status_label.value = "Camera not found!"
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    fps_ema = 0.0
    frame_count = 0

    try:
        while running[0]:
            t_start = time.perf_counter()
            ret, frame = cap.read()
            if not ret: continue

            # Stage 1: Pre-processing (Reverted to standard [-1, 1] normalization for original weights)
            t1 = time.perf_counter()
            resized = cv2.resize(frame, (NN_W, NN_H), interpolation=cv2.INTER_NEAREST)
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            img_float = resized.astype(np.float32) * (1.0 / 127.5) - 1.0
            img_fixed = float_to_fixed16(img_float.flatten())
            t_pre = (time.perf_counter() - t1) * 1000

            # Stage 2: FPGA Inference
            det_raw, t_conv = run_fpga_pipeline(img_fixed)

            # Stage 3: Post-processing
            t3 = time.perf_counter()
            fh, fw = frame.shape[:2]
            detections = decode_and_nms(det_raw, fh, fw)
            t_post = (time.perf_counter() - t3) * 1000

            dt = time.perf_counter() - t_start
            instant_fps = 1.0 / dt if dt > 0 else 0
            fps_ema = (0.3 * instant_fps + 0.7 * fps_ema) if fps_ema > 0 else instant_fps

            draw_overlay(frame, t_pre, t_conv, t_post, fps_ema, detections)

            _, jpeg = cv2.imencode('.jpg', frame, _JPEG_PARAMS)
            image_widget.value = jpeg.tobytes()
            
            # Count only persons for the status label
            persons_only = [d for d in detections if d[5] == 0]
            status_label.value = f"Live | Demo Mode: Person Only | Drawn: {len(persons_only)}"
            frame_count += 1

    except KeyboardInterrupt: pass
    finally:
        cap.release()
        status_label.value = f"Stopped. Processed {frame_count} frames."

# =================================================================
# 10. CLEANUP
# =================================================================
def cleanup():
    try:
        buf_a.freebuffer()
        buf_b.freebuffer()
        dummy_buf.freebuffer()
        for b in layer_w_bufs.values(): b.freebuffer()
        for b in layer_b_bufs.values(): b.freebuffer()
        print("All CMA buffers freed. Restart kernel to reload.")
    except Exception as e:
        print(f"Cleanup error: {e}")