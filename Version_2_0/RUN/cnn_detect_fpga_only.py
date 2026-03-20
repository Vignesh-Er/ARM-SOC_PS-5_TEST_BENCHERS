""" FPGA-ONLY PERSON DETECTOR (NO HOG FALLBACK)
============================================================
Pure PL computation for KV260 with detailed diagnostics.
Architecture: Conv1(3->16,3x3)+ReLU -> Pool -> Conv2(16->32,3x3)+ReLU -> Pool ->
              Conv3(32->64,3x3)+ReLU -> Pool -> Conv4(64->128,3x3)+ReLU -> Pool ->
              DetHead(128->7,1x1,linear) [SOFTWARE]
Output: 10x10x7 tensor [tx, ty, tw, th, obj, bg, person]
"""

import pynq
import numpy as np
import cv2
import time
import threading
import ipywidgets as widgets
from IPython.display import display
import os

# ============================================================================
# 1. HARDWARE INITIALIZATION
# ============================================================================
current_dir = os.getcwd()
BIT_NAME = os.path.join(current_dir, 'design_1_wrapper.bit')

if not os.path.exists(BIT_NAME):
    raise FileNotFoundError(f'Bitstream not found: {BIT_NAME}')

print('Loading KV260 Overlay...')
overlay = pynq.Overlay(BIT_NAME)
cnn_ip = overlay.cnn_accel_top_0

REG_CTRL = 0x00
REG_INPUT_L, REG_INPUT_H = 0x10, 0x14
REG_OUTPUT_L, REG_OUTPUT_H = 0x1C, 0x20
REG_WEIGHTS_L, REG_WEIGHTS_H = 0x28, 0x2C
REG_BIASES_L, REG_BIASES_H = 0x34, 0x38
REG_IMG_H, REG_IMG_W = 0x40, 0x48
REG_IN_C, REG_OUT_C = 0x50, 0x58
REG_MODE, REG_KERNEL = 0x60, 0x68

MODE_CONV_RELU, MODE_MAXPOOL, MODE_CONV_LINEAR = 0, 1, 2

# ============================================================================
# 2. PARAMETERS
# ============================================================================
FP_SCALE = 1024
INPUT_ELEMS = 160 * 160 * 3
MAX_BUF_ELEMS = 160 * 160 * 128
DISPLAY_EVERY_N = 3
JPEG_QUALITY = 80
CONF_THRESH = 0.05  # Keep low (0.05-0.10) for QAT weights
NMS_IOU_THRESH = 0.30

# ============================================================================
# 3. MEMORY ALLOCATION
# ============================================================================
print('Allocating flat CMA buffers...')
buf_A = pynq.allocate(shape=(MAX_BUF_ELEMS,), dtype=np.int16)
buf_B = pynq.allocate(shape=(MAX_BUF_ELEMS,), dtype=np.int16)

# ============================================================================
# 4. WEIGHT LOADING
# ============================================================================
WEIGHT_DIRS = [
    os.path.join(current_dir, 'kv260_hls_weights_qat'),
    os.path.join(current_dir, 'kv260_hls_weights_fixed'),
    os.path.join(current_dir, 'weights'),
]
WEIGHTS_DIR = next((d for d in WEIGHT_DIRS if os.path.isdir(d)), None)
if not WEIGHTS_DIR:
    raise FileNotFoundError('Weights directory not found.')

def load_weights_file(name, shape):
    path = os.path.join(WEIGHTS_DIR, name)
    raw = np.load(path)
    fixed = np.clip(np.round(raw * FP_SCALE), -32768, 32767).astype(np.int16)
    buf = pynq.allocate(shape=shape, dtype=np.int16)
    np.copyto(buf, fixed.reshape(shape))
    buf.flush()
    return buf

print(f'Loading weights from {WEIGHTS_DIR}...')
weights = {
    'l1_w': load_weights_file('layer0_weights.npy', (16, 3, 3, 3)),
    'l1_b': load_weights_file('layer0_biases.npy', (16,)),
    'l2_w': load_weights_file('layer1_weights.npy', (32, 16, 3, 3)),
    'l2_b': load_weights_file('layer1_biases.npy', (32,)),
    'l3_w': load_weights_file('layer2_weights.npy', (64, 32, 3, 3)),
    'l3_b': load_weights_file('layer2_biases.npy', (64,)),
    'l4_w': load_weights_file('layer3_weights.npy', (128, 64, 3, 3)),
    'l4_b': load_weights_file('layer3_biases.npy', (128,)),
}
W_ADDR = {k: int(v.physical_address) for k, v in weights.items()}

# Pre-load software det_head weights (Bypassing FPGA bug)
det_w_np = np.load(os.path.join(WEIGHTS_DIR, 'det_head_weights.npy')).reshape(7, 128).astype(np.float32)
det_b_np = np.load(os.path.join(WEIGHTS_DIR, 'det_head_biases.npy')).astype(np.float32)

# ============================================================================
# 5. INFERENCE PIPELINE
# ============================================================================
def run_layer(in_addr, out_addr, wt_addr, bi_addr, h, w, in_c, out_c, mode, k, timeout=5.0):
    cnn_ip.write(REG_INPUT_L, in_addr & 0xFFFFFFFF)
    cnn_ip.write(REG_INPUT_H, (in_addr >> 32) & 0xFFFFFFFF)
    cnn_ip.write(REG_OUTPUT_L, out_addr & 0xFFFFFFFF)
    cnn_ip.write(REG_OUTPUT_H, (out_addr >> 32) & 0xFFFFFFFF)
    cnn_ip.write(REG_WEIGHTS_L, wt_addr & 0xFFFFFFFF)
    cnn_ip.write(REG_WEIGHTS_H, (wt_addr >> 32) & 0xFFFFFFFF)
    cnn_ip.write(REG_BIASES_L, bi_addr & 0xFFFFFFFF)
    cnn_ip.write(REG_BIASES_H, (bi_addr >> 32) & 0xFFFFFFFF)
    cnn_ip.write(REG_IMG_H, h)
    cnn_ip.write(REG_IMG_W, w)
    cnn_ip.write(REG_IN_C, in_c)
    cnn_ip.write(REG_OUT_C, out_c)
    cnn_ip.write(REG_MODE, mode)
    cnn_ip.write(REG_KERNEL, k)
    cnn_ip.write(REG_CTRL, 0x01)
    t0 = time.perf_counter()
    while not (cnn_ip.read(REG_CTRL) & 0x2):
        if (time.perf_counter() - t0) > timeout:
            raise TimeoutError(f'Layer timeout (mode={mode})')

def hardware_inference():
    A, B = buf_A.physical_address, buf_B.physical_address
    # FPGA Backbone
    run_layer(A, B, W_ADDR['l1_w'], W_ADDR['l1_b'], 160, 160, 3, 16, MODE_CONV_RELU, 3)
    run_layer(B, A, 0, 0, 160, 160, 16, 16, MODE_MAXPOOL, 1)
    run_layer(A, B, W_ADDR['l2_w'], W_ADDR['l2_b'], 80, 80, 16, 32, MODE_CONV_RELU, 3)
    run_layer(B, A, 0, 0, 80, 80, 32, 32, MODE_MAXPOOL, 1)
    run_layer(A, B, W_ADDR['l3_w'], W_ADDR['l3_b'], 40, 40, 32, 64, MODE_CONV_RELU, 3)
    run_layer(B, A, 0, 0, 40, 40, 64, 64, MODE_MAXPOOL, 1)
    run_layer(A, B, W_ADDR['l4_w'], W_ADDR['l4_b'], 20, 20, 64, 128, MODE_CONV_RELU, 3)
    run_layer(B, A, 0, 0, 20, 20, 128, 128, MODE_MAXPOOL, 1)
    # Software Det Head to bypass HLS alignment bug
    buf_A.invalidate()
    feat_flat = np.asarray(buf_A[:10 * 10 * 128], dtype=np.int16)
    feat = (feat_flat.astype(np.float32) / FP_SCALE).reshape(10, 10, 128)
    return feat @ det_w_np.T + det_b_np

# ============================================================================
# 6. POST-PROCESSING
# ============================================================================
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0

def decode_detections(output, conf_thresh, frame_w, frame_h):
    stride_x, stride_y = frame_w / 10, frame_h / 10
    # Sigmoids for classification only
    obj_map = sigmoid(output[:, :, 4])
    bg_map = sigmoid(output[:, :, 5])
    person_map = sigmoid(output[:, :, 6])
    score_map = obj_map * person_map * (1.0 - bg_map)
    mask_y, mask_x = np.where(score_map > conf_thresh)
    candidates = []
    for gy, gx in zip(mask_y, mask_x):
        pred = output[gy, gx]
        conf = float(score_map[gy, gx])
        # FIX: tx, ty are strictly linear during training, do not use sigmoid
        cx = (gx + pred[0]) * stride_x
        cy = (gy + pred[1]) * stride_y
        bw = np.exp(np.clip(pred[2], -4, 4)) * stride_x * 0.8
        bh = np.exp(np.clip(pred[3], -4, 4)) * stride_y * 0.8
        x1, y1 = int(cx - bw/2), int(cy - bh/2)
        x2, y2 = int(cx + bw/2), int(cy + bh/2)
        candidates.append([max(0,x1), max(0,y1), min(frame_w,x2), min(frame_h,y2), conf])
    # Non-Maximum Suppression
    candidates.sort(key=lambda x: x[4], reverse=True)
    final_boxes = []
    for cand in candidates:
        if not any(iou(cand, fb) > NMS_IOU_THRESH for fb in final_boxes):
            final_boxes.append(cand)
    return final_boxes

class CameraCapture:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.lock = threading.Lock()
        self.latest = None
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.latest = frame
            else:
                time.sleep(0.01)

    def read(self):
        with self.lock:
            return (True, self.latest.copy()) if self.latest is not None else (False, None)

    def release(self):
        self.running = False
        self.cap.release()

# ============================================================================
# 7. MAIN LOOP
# ============================================================================
video_widget = widgets.Image(format='jpeg', width=640, height=480)
display(video_widget)

try:
    try:
        cap = CameraCapture(0)
    except RuntimeError:
        cap = CameraCapture(1)

    rgb_small = np.empty((160, 160, 3), dtype=np.uint8)

    while True:
        loop_start = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        img_resized = cv2.resize(frame, (160, 160), interpolation=cv2.INTER_NEAREST)
        rgb_small[:] = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_fp = np.clip(np.round((rgb_small.astype(np.float32)/255.0)*FP_SCALE), -32768, 32767).astype(np.int16)
        buf_A[:INPUT_ELEMS] = img_fp.ravel()
        buf_A.flush()

        det_output = hardware_inference()
        boxes = decode_detections(det_output, CONF_THRESH, frame.shape[1], frame.shape[0])
        loop_elapsed = max(time.perf_counter() - loop_start, 1e-6)

        for b in boxes:
            cv2.rectangle(frame, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)
            cv2.putText(frame, f'person {b[4]:.2f}', (b[0], max(0, b[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(frame, f'FPS: {1.0/loop_elapsed:.1f}', (24, 38),
                        cv2.FONT_HERSHEY_DUPLEX, 0.72, (0, 255, 255), 2)

        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        video_widget.value = jpeg.tobytes()

except KeyboardInterrupt:
    print('Stopped.')

finally:
    if 'cap' in locals():
        cap.release()
    buf_A.freebuffer()
    buf_B.freebuffer()
    for w_buf in weights.values():
        w_buf.freebuffer()
