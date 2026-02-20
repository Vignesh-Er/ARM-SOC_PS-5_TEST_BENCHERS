# ARM-SOC_PS-5_TEST_BENCHERS  
## Real-Time Object Detection Using Hardware-Accelerated CNN on AMD Xilinx Kria KV260

Bharat AI-SoC Student Challenge  
Problem Statement 5 – Hardware/Software Co-Design for Edge AI

---

## 1. Project Overview

This project presents the design, implementation, optimization, and validation of a hardware-accelerated Convolutional Neural Network (CNN) inference system deployed on the AMD Xilinx Kria KV260 Vision AI Starter Kit.

The underlying platform is the Zynq UltraScale+ MPSoC (K26 SOM), which integrates:

- Quad-core ARM Cortex-A53 Processing System (PS)
- FPGA Programmable Logic (PL)
- AXI interconnect fabric
- Shared DDR4 memory

The objective of this project is to offload compute-intensive CNN operations to FPGA fabric and demonstrate measurable performance improvement over a CPU-only implementation.

The final system achieves a 2.93× speedup compared to the ARM-only baseline, exceeding the minimum 2× acceleration requirement defined in the competition problem statement.

---

## 2. System Architecture

The system follows a hardware/software co-design methodology.

### Processing System (PS) Responsibilities

- Image acquisition and preprocessing
- DMA buffer allocation
- AXI DMA transaction control
- Post-processing and output decoding
- Performance measurement
- Runtime control via PYNQ

### Programmable Logic (PL) Responsibilities

- Convolution computation
- ReLU activation
- Max pooling
- Fixed-point arithmetic
- Streaming output generation

### Data Movement

- AXI4-Stream interface between DMA and accelerator
- AXI4-Lite interface for control signals
- AXI DMA (MM2S and S2MM) for DDR–PL data transfers
- Explicit cache flush/invalidate for coherency

---

## 3. CNN Model Architecture

A custom lightweight 3-layer CNN was designed specifically for FPGA deployment.

Input Size: 128 × 128 × 3  
Total Parameters: 3,729  

| Layer        | Operation                | Output Shape |
|--------------|--------------------------|--------------|
| Conv1        | 3×3 + ReLU               | 64×64×8      |
| MaxPool1     | 2×2                      | 32×32×8      |
| Conv2        | 3×3 + ReLU               | 32×32×16     |
| MaxPool2     | 2×2                      | 16×16×16     |
| Conv3        | 3×3 + ReLU               | 16×16×16     |
| Global Avg   | GAP                      | 1×1×16       |
| Dense        | Fully Connected + Sigmoid| 1            |

The model performs single-class person detection.

---

## 4. Quantization Strategy

To enable efficient FPGA arithmetic, the model was quantized using fixed-point representation.

Fixed-point format used:

```
ap_fixed<16,6>
```

- 16 total bits
- 6 integer bits (including sign)
- 10 fractional bits
- Resolution: 2^-10
- Range: [-32, 31.999]

Quantization-aware training was applied using fake quantization during training to preserve accuracy.

Validation accuracy after quantization remained within 0.5% of the FP32 baseline.

---

## 5. Hardware Accelerator Design (Vitis HLS)

The convolution accelerator was implemented using Vitis HLS.

Key optimizations applied:

- Loop pipelining with Initiation Interval (II) = 1
- 9-parallel MAC execution (3×3 kernel)
- Array partitioning of weights
- DATAFLOW pragma for task-level pipelining
- Three-row line buffer architecture
- AXI4-Stream input/output ports
- AXI4-Lite control interface

The line-buffer architecture reduced BRAM usage by approximately 70% compared to naive full feature map storage.

Clock Frequency: 150 MHz  
Timing Closure: Achieved with positive slack  

---

## 6. Vivado Integration

The accelerator IP was exported from Vitis HLS and integrated using Vivado IP Integrator.

Block Design Components:

- Zynq UltraScale+ MPSoC
- AXI DMA IP
- CNN Accelerator IP
- AXI Interconnect
- Clock Wizard (150 MHz PL clock)

Final outputs:

- design_1_wrapper.bit
- design_1_wrapper.hwh

Deployment was performed using PYNQ runtime.

---

## 7. Performance Results

Benchmarking was conducted on real hardware under controlled conditions.

### CPU-Only Baseline (ARM Cortex-A53)

- Median Inference Latency: 630 ms
- Throughput: 1.59 FPS
- Std Dev: ±12 ms

### FPGA-Accelerated (PS + PL)

- Median Inference Latency: 215 ms
- Throughput: 4.65 FPS
- Std Dev: ±4 ms

### Speedup Calculation

```
Speedup = 630 / 215 = 2.93×
```

The design exceeds the 2× minimum requirement by a significant margin.

---

## 8. Resource Utilization (Post-Implementation)

| Resource | Used | Available | Utilization |
|----------|------|-----------|------------|
| LUT      | 18,432 | 117,120 | 15.7% |
| FF       | 24,576 | 234,240 | 10.5% |
| BRAM     | 12 | 144 | 8.3% |
| DSP48E2  | 36 | 1,248 | 2.9% |

The design demonstrates high efficiency and leaves substantial headroom for future expansion.

---

## 9. Power and Energy Efficiency

Estimated On-Chip Power During Inference:

- PS: ~2.1 W
- PL Dynamic: ~0.8 W
- DDR: ~0.5 W
- Total: ~3.7 W

Energy per inference:

FPGA path:
```
3.7 W × 0.215 s = 0.796 J
```

CPU-only path:
```
2.1 W × 0.630 s = 1.323 J
```

Energy reduction ≈ 40%.

---

## 10. Major Engineering Challenges Solved

- TLAST misalignment causing DMA hang
- AXI stream deadlock
- Cache coherency issues
- Weight tensor transpose mismatch
- Initiation Interval > 1
- Timing violations at 200 MHz
- Colab RAM crash during dataset caching

All issues were diagnosed and resolved through systematic hardware debugging and simulation validation.

---

## 11. Repository Structure

```
/hls        -> Vitis HLS source and reports  
/vivado     -> Block design, bitstream, hardware files  
/pynq       -> Runtime inference scripts  
/training   -> Model training and weight export  
/docs       -> Final project report and screenshots  
README.md   -> Project documentation  
```

---

## 12. Compliance with Problem Statement

This project satisfies all competition requirements:

- Custom CNN deployed on Zynq UltraScale+ MPSoC
- FPGA fabric actively performs convolution computation
- Hardware bitstream generated and tested on real board
- Measured >2× speedup over CPU-only implementation
- Documented performance, resource usage, and power analysis
- Complete hardware/software co-design implementation

---

## 13. Conclusion

This project demonstrates that a carefully optimized custom CNN accelerator implemented using Vitis HLS and integrated via AXI DMA can significantly improve inference latency, throughput, and energy efficiency on embedded FPGA SoC platforms.

The achieved 2.93× speedup validates the effectiveness of hardware acceleration for edge AI workloads and highlights the strengths of heterogeneous ARM + FPGA architectures for real-time embedded inference systems.

---

Bharat AI-SoC Student Challenge  
Problem Statement 5  
Team: TESTBENCHERS  
Platform: AMD Xilinx Kria KV260  
Toolchain: Vitis HLS 2024.2 | Vivado 2024.2 | PYNQ 3.0
