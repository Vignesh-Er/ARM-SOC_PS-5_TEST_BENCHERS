# 🚀 ARM-SOC_PS-5_TEST_BENCHERS  
## ARM_KRIA_KV260_HARDWARE_ACCELERATOR_FOR_MACHINE_LEARNING  

---

## 📌 Project Overview

This repository contains a complete implementation of a **hardware-accelerated Convolutional Neural Network (CNN)** on the **AMD Xilinx Kria KV260 (K26 SOM)** platform.

The system demonstrates ARM–FPGA hardware/software co-design for real-time object detection and achieves measurable performance improvement over CPU-only execution.

---

## 🎯 Key Highlights

- ✅ Custom CNN accelerator using **Vitis HLS**
- ✅ AXI DMA-based high-speed PS–PL communication
- ✅ Quantization-aware training (ap_fixed<16,6>)
- ✅ Real-time person detection demo
- ✅ >2× speedup over CPU-only implementation
- ✅ Fully deployed on physical hardware (not simulation)

---

# 🧠 Problem Statement Reference

**Problem Statement 5**  
Real-Time Object Detection Using Hardware-Accelerated CNN  
(Bharat AI-SoC Student Challenge)

Objective:  
Design and implement a hardware-accelerated CNN inference system on a Zynq SoC and demonstrate measurable performance improvement over CPU-only execution.

---

# 🏗 System Architecture

## Hardware Platform

- Board: Kria KV260 Vision AI Starter Kit  
- SoC: Zynq UltraScale+ MPSoC  
- PS: Quad-core ARM Cortex-A53  
- PL: FPGA Fabric  
- Interface: AXI4-Stream + AXI DMA  
- Runtime: PYNQ Linux  

---

## 🔷 Processing Partition

| Component        | Location | Function |
|------------------|----------|----------|
| Preprocessing    | PS       | Resize, Normalize |
| Convolution      | PL       | Accelerated CNN |
| Activation       | PL       | ReLU |
| Pooling          | PL       | Downsampling |
| Postprocessing   | PS       | NMS + Bounding Boxes |

---

## 🔷 Data Flow

```
Image / Camera
      ↓
Preprocessing (ARM - PS)
      ↓
AXI DMA (MM2S)
      ↓
CNN Accelerator (FPGA - PL)
      ↓
AXI DMA (S2MM)
      ↓
Postprocessing (ARM - PS)
      ↓
Detection Output
```

---

# ⚙️ Development Workflow

## 1️⃣ Platform Setup

- Flashed PYNQ image to SD card
- Booted KV260
- Connected via Ethernet
- Accessed Jupyter Notebook

---

## 2️⃣ Vitis HLS Accelerator

Implemented:

- 2D Convolution Engine  
- ReLU Activation  
- Max Pooling  
- AXI4-Stream Interface  
- AXI4-Lite Control Registers  

### Key Optimizations

```cpp
#pragma HLS PIPELINE II=1
#pragma HLS DATAFLOW
#pragma HLS ARRAY_PARTITION
```

Achieved:
- Initiation Interval (II) = 1  
- Efficient BRAM usage  
- High-throughput streaming architecture  

---

## 3️⃣ Vivado Block Design

Integrated:

- Zynq MPSoC  
- AXI DMA (MM2S & S2MM)  
- Custom CNN HLS IP  
- AXI Interconnect  
- Clocking & Reset Modules  

Generated:

- design_1_wrapper.bit  
- design_1_wrapper.hwh  
- .xsa file  

---

## 4️⃣ PYNQ Runtime Execution

Example Python Execution:

```python
overlay = Overlay("design_1_wrapper.bit")
dma.sendchannel.transfer(inp_buffer)
dma.recvchannel.transfer(out_buffer)
cnn_ip.write(0x00, 0x01)
dma.sendchannel.wait()
dma.recvchannel.wait()
```

---

# 🧠 Model Training & Quantization

### Challenges Faced

- Colab RAM crashes  
- Weight export issues  
- FP32 vs fixed-point mismatch  

### Solutions Implemented

- Disabled RAM caching  
- Monolithic training script  
- Fake quantization for ap_fixed<16,6>  

Final Model:

- Single-class (Person) detector  
- Quantization-aware trained  
- FPGA-compatible weights (.npy)  

---

# 📊 Performance Comparison

| Implementation | Latency |
|---------------|----------|
| CPU-only (PS) | ~630 ms |
| PS + PL       | ~215 ms |

### 🚀 Speedup Achieved

```
630 / 215 ≈ 2.9×
```

✔ Exceeds required 2× performance improvement.

---

# 🛠 Major Issues & Fixes

| Issue | Cause | Fix |
|-------|--------|------|
| DMA Hang | Missing TLAST | Added TLAST logic |
| No Detection | Weight mismatch | Retrained model |
| II > 1 | Memory dependency | Partitioned arrays |
| Timing violation | Over-unrolling | Balanced DSP usage |
| DDR stale data | Cache issue | Cache flush/invalidate |

---

# 📈 Resource Optimization

- Line-buffer architecture (3-row buffer)  
- ~70% BRAM reduction  
- Controlled DSP utilization  
- Stable AXI streaming  
- Deterministic execution  

---

# 🏆 Achievements

- ✔ Custom CNN accelerator  
- ✔ Fully functional PS–PL system  
- ✔ Stable AXI DMA communication  
- ✔ Real hardware deployment  
- ✔ >2× measurable speedup  
- ✔ Accurate person detection  

---

# 📂 Repository Structure

```
ARM-SOC_PS-5_TEST_BENCHERS/
│
├── hls/
│   ├── cnn_accelerator.cpp
│   └── tb_cnn.cpp
│
├── vivado/
│   ├── design_1_wrapper.bit
│   ├── design_1_wrapper.hwh
│
├── pynq/
│   └── inference.py
│
├── training/
│   ├── train.py
│   └── export_weights.py
│
├── docs/
│   └── Final_Report.pdf
│
└── README.md
```

---

# 🎓 Learning Outcomes

- Vitis HLS optimization techniques  
- AXI protocol debugging  
- DMA system integration  
- Hardware-software co-design  
- Quantization-aware deployment  
- Real FPGA debugging  

---

# 🔮 Future Improvements

- Move NMS to FPGA  
- INT8 full pipeline  
- Multi-class expansion  
- Parallel convolution engines  
- Compare with Vitis AI DPU  

---

# 🏁 Conclusion

This project demonstrates a complete hardware-accelerated CNN inference system on the Kria KV260 platform using:

- Custom Vitis HLS IP  
- AXI DMA streaming  
- PS–PL co-design  
- Quantization-aware training  
- Measurable real-world performance improvement  

This is a full-stack FPGA deployment project — not simulation-only.

---

## 📜 License

MIT License

---

## 👨‍💻 Author

Developed as part of Bharat AI-SoC Student Challenge – Problem Statement 5.
