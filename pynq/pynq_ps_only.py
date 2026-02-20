import cv2
import numpy as np
import ipywidgets as widgets
from IPython.display import display
import time
import os

# ==============================================================================
# 1. MODEL SETUP (YOLOv3-Tiny)
# ==============================================================================
# Files required: yolov3-tiny.weights, yolov3-tiny.cfg, coco.names
if not os.path.exists("yolov3-tiny.weights") or not os.path.exists("yolov3-tiny.cfg"):
    print("❌ Error: YOLO weights/cfg not found. Please upload them to the board.")
    raise SystemExit(1)

# Load Network
try:
    net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
    # Force CPU Backend
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    print("✅ Model Loaded on CPU.")
except Exception as e:
    print(f"❌ Model Load Error: {e}")
    raise SystemExit(1)

# ==============================================================================
# 2. CAMERA SETUP
# ==============================================================================
def open_camera():
    # Try USB camera indices 0 and 1
    for i in [0, 1]:
        cap = cv2.VideoCapture(i)
        if cap.isOpened(): return cap
    return None

cap = open_camera()

# ==============================================================================
# 3. UI SETUP
# ==============================================================================
# Create display widget
image_widget = widgets.Image(format='jpeg', width=640, height=480)
stop_btn = widgets.Button(description="Stop CPU Demo", button_style='danger')
display(widgets.VBox([image_widget, stop_btn]))

stop_flag = False
def on_stop(b):
    global stop_flag
    stop_flag = True
stop_btn.on_click(on_stop)

# ==============================================================================
# 4. MAIN LOOP
# ==============================================================================
if cap is None or not cap.isOpened():
    print("❌ Camera not found.")
else:
    print("🎥 Starting PS (CPU) Only Inference...")
    
    try:
        while not stop_flag:
            ret, frame = cap.read()
            if not ret: continue
            
            # --- TIMER START ---
            start_total = time.time()
            
            # ---------------------------------------------------------
            # STAGE 1: PREPROCESSING (Resize -> Blob)
            # ---------------------------------------------------------
            t1 = time.time()
            # Resize to 416x416 (Standard YOLO input), SwapRB=True, Crop=False
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            t_pre = (time.time() - t1) * 1000  # ms

            # ---------------------------------------------------------
            # STAGE 2: CONVOLUTION / INFERENCE (CPU)
            # ---------------------------------------------------------
            t2 = time.time()
            outs = net.forward(output_layers)
            t_conv = (time.time() - t2) * 1000 # ms

            # ---------------------------------------------------------
            # STAGE 3: POSTPROCESSING (NMS -> Draw)
            # ---------------------------------------------------------
            t3 = time.time()
            
            h, w, _ = frame.shape
            class_ids = []
            confidences = []
            boxes = []
            
            # Parse outputs
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3: # Confidence Threshold
                        cx = int(detection[0] * w)
                        cy = int(detection[1] * h)
                        bw = int(detection[2] * w)
                        bh = int(detection[3] * h)
                        x = int(cx - bw / 2)
                        y = int(cy - bh / 2)
                        boxes.append([x, y, bw, bh])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Non-Maximum Suppression (Remove overlapping boxes)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            # Draw Boxes
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, bw, bh = boxes[i]
                    label = str(classes[class_ids[i]])
                    conf = confidences[i]
                    
                    # Box Color (Red for CPU)
                    color = (0, 0, 255) 
                    
                    cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                    
                    # Label background
                    label_text = f"{label} {int(conf*100)}%"
                    (wt, ht), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(frame, (x, y - ht - 5), (x + wt, y), color, -1)
                    
                    cv2.putText(frame, label_text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            t_post = (time.time() - t3) * 1000 # ms
            
            # --- TIMER END ---
            end_total = time.time()
            fps = 1.0 / (end_total - start_total)
            
            # ---------------------------------------------------------
            # METRICS DASHBOARD
            # ---------------------------------------------------------
            # Background
            cv2.rectangle(frame, (5, 5), (220, 110), (0, 0, 0), -1)
            cv2.rectangle(frame, (5, 5), (220, 110), (255, 255, 255), 1)
            
            cv2.putText(frame, "PS (CPU) ONLY", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Detailed Breakdown
            cv2.putText(frame, f"Pre-proc: {t_pre:.1f} ms", (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Infer   : {t_conv:.1f} ms", (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            cv2.putText(frame, f"Post    : {t_post:.1f} ms", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Push to widget
            _, jpeg = cv2.imencode('.jpg', frame)
            image_widget.value = jpeg.tobytes()
            
    except Exception as e:
        print(f"Runtime Error: {e}")
        
    finally:
        cap.release()
        print("🛑 Camera Released.")