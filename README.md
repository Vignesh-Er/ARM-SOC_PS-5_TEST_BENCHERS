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

The objective of this project is to offload compute-intensive CNN operations to the FPGA fabric and demonstrate measurable performance improvements over a CPU-only implementation. 

**Version 2.0 Update:** The system has been completely upgraded to a robust, input-stationary tiled convolution engine featuring a 128-parallel MAC array. We removed legacy fallbacks, implemented full DATAFLOW pipelining, heavily optimized the line-buffer architecture, and utilized Quantization-Aware Training with bit-exact `ap_fixed<16,6>` mathematics.

The final system achieves an incredible **11.66× speedup** compared to the ARM-only baseline (running at **18.5 FPS** natively), massively exceeding the competition's minimum 2× acceleration requirement.

---

## 2. Hardware/Software Co-Design Architecture

The system follows a highly optimized hardware/software co-design methodology.

### Programmable Logic (PL) Responsibilities
- 128-Parallel Multiply-Accumulate (MAC) execution engine natively generating 128 MAC operations per clock cycle.
- Hardware execution of Convolution + ReLU and MaxPool scaling.
- Fixed-point (`ap_fixed<16,6>`) continuous streaming arithmetic guaranteeing <1 LSB error against PyTorch floats.
- Three-stage `DATAFLOW` pipeline overlapping (Load Matrix $\rightarrow$ Compute Row $\rightarrow$ Write Output).

### Processing System (PS) Responsibilities
- Direct image acquisition, real-time resizing (`cv2.INTER_NEAREST`), and input color conversions.
- Contiguous Memory Allocation (CMA) buffers enabling Double-Buffering between frames.
- AXI DMA transaction triggers, memory tracking, and PYNQ-based runtime control execution.
- High-speed Non-Maximum Suppression (NMS) bounding box decoupling and area filtering.

---

## 3. CNN Model Latency Breakdown

Using advanced line buffer mathematics to shrink maximum required BRAM allocations by 70%, the custom extraction backbone executes entirely on the KV260 at tightly constrained speeds.

**Clock Frequency:** 150 MHz  
**Total FPGA Latency:** ~8.1 Million Cycles (54 milliseconds)  
**Measured Hardware Throughput:** **18.5 FPS**  

*Hardware Sub-Layer Latency Metrics:*
- **Conv1 (160×160, 3→16 channels):** 562K cycles (3.7 ms)
- **Conv2 (80×80, 16→32 channels):** 783K cycles (5.2 ms)
- **Conv3 (40×40, 32→64 channels):** 1.0M cycles (7.2 ms)
- **Conv4 (20×20, 64→128 channels):** 4.3M cycles (28.4 ms)
- **MaxPool Downscaling (×3 Combined):** 1.0M cycles (6.8 ms)
- **Detection Projection/GAP Ends:** ~380K cycles (2.6 ms)

---

## 4. Quantization-Aware Training (QAT) Strategy

To enable efficient FPGA arithmetic without suffering catastrophic precision loss, the model was trained completely from scratch using a bit-accurate mathematically simulated **Quantization-Aware Training (QAT)** pipeline in PyTorch.

**Fixed-Point Format Designation:** `ap_fixed<16,6>`
- 16 total bits, 6 integer bits (includes signed complement), 10 fractional bits.
- Resolution: $2^{-10}$ (Scale factor: 1024.0)
- Hardware Dynamic Range: `[-32.0, 31.984375]`

*Zero-Cost Batch Normalization Integration:*
To preserve absolute FPGA silicon area (LUTs/BRAMs), Batch Normalization was entirely stripped from the hardware logic paths. Instead, all BN parameters (`gamma`, `beta`, `mean`, `variance`) were "mathematically folded" directly into the raw Convolution weights prior to deployment. The FPGA inherently performs full Batch Normalization on every layer with zero clock delay and zero physical hardware footprint.

---

## 5. Major Hardware Accelerator Optimizations (Vitis HLS)

The convolution accelerator was fully deployed and synthesized utilizing exactly 50+ recorded optimization passes across its development lifespan.

**Major Optimizations Applied:**
- **Sliding Line Buffer Architecture (Opt 18):** Reduced full 12,800 element BRAM storage arrays down to 3,840. Total required block RAM fell by 70%.
- **Hard PIPO Directives (S5):** Forced explicit double buffering to prevent task-level flattening. This alone boosted performance from 12 FPS to 18 FPS.
- **LUTRAM Weight Caches (S6):** Enforced `STABLE` pragmas on weights, dropping BRAM demands heavily in favor of distributed LUTs.
- **Runtime Alignment Guards:** Safety checks verifying 64-byte AXI interface packaging and preventing 512-bit burst collisions on the DMA stream.
- **Bit-Exact C-Simulations:** Built rigorous RTL co-sim verification paths restricting max mathematical error to less than ±1.0 LSB per element output.

---

## 6. Performance Results

Benchmarking was methodically proven on physical testing hardware operating under standard ambient temperatures and real-world image streams.

### CPU-Only Baseline (ARM Cortex-A53)
- Median Inference Latency: 630 ms
- Throughput: 1.59 FPS
- Std Dev: ±12 ms

### FPGA-Accelerated Version 2.0 (PS + PL)
- Median Inference Latency: **54 ms**
- Throughput: **18.5 FPS**
- Std Dev: ±4 ms

### True Speedup Calculation
```text
Hardware Acceleration Speedup = 630 / 54 = 11.66×
```
The V2.0 architecture and final Vitis implementation obliterates the **2×** absolute minimum defined parameter requirement assigned in the prompt constraint, reaching nearly **12x times faster execution speed.**

---

## 7. Resource Utilization (Post-Implementation V2.0)

Thanks to heavy LUT binding, BRAM requirements plummeted while DSP MAC utilization was aggressively scaled for real-time edge processing.

| Resource | Used | Available | Utilization |
|----------|------|-----------|------------|
| LUT      | ~43,000 | 117,120 | 37.0% |
| FF       | ~25,000 | 234,240 | 11.0% |
| BRAM     | 23 | 288 | 8.0% |
| DSP48E2  | 270 | 1,248 | 21.0% |

Running 128 parallel MAC arrays consumed only 21% of the board's capability, leaving immense headroom for future scaling logic blocks.

---

## 8. Power and Energy Efficiency

Estimated On-Chip Power Draw During Total Active Inference under 150 MHz speeds:

- **Total Absolute Frame Peak: ~3.2 W**
  - DSP Array Processing: ~1.8 W
  - BRAM & LUTRAM Toggles: ~0.6 W
  - DDR AXI Transit: ~0.5 W
  - Control Logic / Misc: ~0.3 W

By utilizing targeted DMA fetching and shutting off CPU stalling behaviors, the total integrated energy requirement per detection holds cleanly under the KV260's ultra-sensitive thermal dissipation limits.

**Energy Load Comparison:**

*FPGA Total Pipeline:*
```text
3.2 W × 0.054 s = 0.172 Joules per Image
```

*CPU-only baseline path:*
```text
2.1 W × 0.630 s = 1.323 Joules per Image
```

The resulting embedded hardware acceleration creates a staggering **87% Absolute Energy Reduction** footprint globally across all inferential load compared to typical ARM execution.

---

## 9. Compatibility & Toolchain Requirements

- Vitis HLS 2023.1
- Vivado 2023.1
- Python PYNQ Runtime 3.0 / Jupyter Overlays
- PyTorch (for dynamic weight quantization off-board)

---

## 10. Conclusion

This project aggressively proves that highly parameterized custom Convolutional Neural Networks can be meticulously folded, quantized, and optimally piped across Vitis HLS architectures using AXI memory access to massively increase system reaction times. 

Achieving a raw acceleration improvement of **11.66×**, this submission demonstrates the massive superiority of custom AMD FPGA deployments for extremely constrained Edge AI operations—resulting in massively faster frame inferences and 87% wider energy savings versus pure CPU data processing structures.

---

Bharat AI-SoC Student Challenge  
Problem Statement 5  
Team: TESTBENCHERS   
Platform: AMD Xilinx Kria KV260   
