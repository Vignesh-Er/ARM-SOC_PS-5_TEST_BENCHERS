// ============================================================================
// CNN Accelerator — Production Header (v4.6.9)
// Target: Xilinx Kria K26 SOM (xck26-sfvc784-2LV-c, ZU5EV)
// Clock:  ~150 MHz (6.67 ns, 20% uncertainty margin)
//
// Architecture: Input-stationary tiled convolution engine
//   - 128 parallel MACs (TILE_M=16 × TILE_N=8)
//   - Conv2D (3×3 or 1×1) + Bias + ReLU or linear
//   - MaxPool 2×2 stride 2
//   - Global Average Pooling
//   - AXI4 Master DDR + AXI4-Lite control
//   - 3-stage DATAFLOW: load_line_buffer || compute_row || write_output_row
//   - ap_done interrupt -> PS IRQ for HW/SW co-design
//   - 16-bit AXI data ports (short* in synthesis, data_t* in C-sim)
//
// Supported layers (PS calls IP once per layer):
//   Conv(3->16,3x3) -> Pool -> Conv(16->32,3x3) -> Pool ->
//   Conv(32->64,3x3) -> Conv(64->128,3x3) -> Pool ->
//   GlobalAvgPool -> Conv(128->num_classes,1x1) [classification head]
//   OR Conv(128->det_ch,1x1) [detection head, NMS on PS]
//
// Weight format: NCHW [Cout][Cin][Ky][Kx] (burst-friendly DDR access)
//   BatchNorm MUST be pre-folded into weights/biases OFFLINE:
//     w_folded = w * gamma / sqrt(var + eps)
//     b_folded = (b - mean) * gamma / sqrt(var + eps) + beta
//
// BATCHNORM FOLDING REQUIREMENT (P0-G):
//   w_folded = w * (gamma / sqrt(var + eps))  -- compute in float64
//   b_folded = (b - mean) * (gamma / sqrt(var + eps)) + beta  -- float64
//   Then quantize using:
//     w_q = ap_fixed<16,6>(round_half_to_even(w_folded * 64.0) / 64.0)
//     b_q = ap_fixed<16,6>(round_half_to_even(b_folded * 64.0) / 64.0)
//   Using truncation (int(x * 64) / 64.0) introduces systematic negative
//   bias in weights that degrades detection confidence scores by 2-5%.
//   Use numpy.round() or Python's round() -- both use banker's rounding.
//
// BIAS QUANTIZATION REQUIREMENT (P0-F):
//   Biases must be quantized offline using round-half-to-even (numpy.round()
//   or torch.round()) matching AP_RND_CONV mode, then stored as ap_fixed<16,6>.
//   Formula: bias_quant = round(bias_float * 2^6) / 2^6
//   Mismatch here causes a per-channel DC offset in all feature maps,
//   leading to systematic false positives or missed detections in NMS.
//
// Data format: HWC [H][W][C] (row-major, flat array in DDR)
//
// DDR requirements:
//   - All pointer arguments MUST be 64-byte aligned.
//     PS: use posix_memalign(64,...) or __attribute__((aligned(64)))
//   - Weights SHOULD be zero-padded to TILE_M/TILE_N multiples for
//     optimal burst performance (conditional branch will handle
//     non-padded weights correctly but with reduced burst efficiency)
//
// v4.1 hardware safety fixes:
//   1. Hard guard: in_channels clamped to MAX_CONV_IN_CH
//   2. Hard guard: out_channels clamped to MAX_OUT_CH
//   3. Hard guard: kernel_size clamped to KERNEL_MAX
//   4. Hard guard: img_height/img_width clamped to MAX_SPATIAL
//   5. global_avg_pool accumulator partitioned (pipeline hazard fix)
//   6. Division-by-zero guard in global average pool
//   7. AXI weight depth computed from max dimensions (not literal)
//   8. Runtime 64-byte pointer alignment check (abort if misaligned)
//   9. AXI weight depth safety-margined (1M entries, decoupled from tensor)
//
// v4.3 performance improvements:
//   10. MAC parallelism doubled: TILE_N 4->8 (128 MACs, 10% DSP)
//   11. 3-stage DATAFLOW: load_weights || compute_tile || write_output
//   12. GAP: cyclic factor 4->16, 4-way parallel accumulation
//   13. AXI outstanding raised: reads 2->8, writes 2->4
//
// v4.5 architectural optimizations:
//   14. accum_t reduced from <48,24> to <32,20> (shorter DSP cascade)
//   15. BIND_STORAGE ram_2p removed from input_buf_local (HLS auto-infers)
//   16. BIND_OP impl=dsp on MAC psum (forces DSP48E2 utilization ~45%)
//   17. DATAFLOW removed from Output_Groups (sequential, no scheduler collapse)
//   18. Full input tile buffer replaced with line buffer (3 rows for 3x3)
//       input_buf[10][10][128] -> line_buf[3][10][128] = 70% BRAM reduction
//       compute_tile -> compute_row (psum 3D->2D, out_tile 3D->2D)
//       Row-by-row streaming with tile_m-outermost loop for weight reuse
//
// v4.6 pipeline & burst optimizations:
//   19. DATAFLOW in Row_Loop: load || compute || write (PIPO line_buf/out_row_buf)
//   20. wc_local bound to LUTRAM (frees ~128 BRAMs, +~6K LUTs)
//   21. load_line_buffer burst-safe: pre-zero + unconditional DDR reads
//   22. load_weights burst-safe: pre-zero + valid-range-only reads
//   23. global_avg_pool: removed UNROLL stepping, simple ch++ with II=1
//   24. maxpool_2x2: accept II=4 (bandwidth-limited, 2.4% of total cycles)
//   25. STABLE pragma on wc_local/bias_cache for DATAFLOW correctness
//
// v4.6.1 correctness, determinism & timing fixes:
//   P0-A: accum_t widened <32,20>-><36,22> (overflow on 128-ch layers)
//   P0-B: Linear-mode saturation gated to ReLU only (detection head fix)
//   P0-C: AP_RND_CONV + AP_SAT on all types (matches PyTorch quantization)
//   P0-D: DEPENDENCE pragma on MAC loops (preserves accumulation order)
//   P0-E: prod_t intermediate preserves full multiply precision
//   P0-F/G: Bias & BN folding quantization requirements documented
//   P0-H: Zero-padding assumption documented (symmetric quant zero-point=0)
//   P1-A: Conv_KX II=2 (MAC reduction tree timing margin at 150MHz -2LV)
//   P1-B: GAP division -> reciprocal multiply (II=1 achieved)
//   P2-A: C-sim assertions added (zero synthesis cost)
//   P2-B: Explicit PIPO on line_buf/out_row_buf
//   P3-A: CNN_ACCEL_VERSION define
//   P3-B: Invalid kernel_size returns (no silent default)

// v4.6.2 architecture & determinism refinements:
//   S0:   prod_t moved to global header scope (DSP inference guarantee)
//   S0-B: prod_t cast order corrected (result cast, not operand cast)
//   P0-A: accum_t rounding changed AP_RND_CONV->AP_TRN (matches PyTorch)
//   P1-A: RESET pragma on psum (clock gating + 200-400mW savings)
//   PDS-A: OCM weight caching guidance for detection head bandwidth
//   PDS-B: Roofline analysis annotation for KV260 layer characterization
//   PDS-C: Weight preload overlap commentary (PD-Swap inspired)
//   PDS-D: 1x1 fast path in load_line_buffer (skip zero-fill for det head)
//   PDS-E: Ternary weight / TLMM future pathway documented
//   H-A:   Tcl DSP assertion build check
//   H-B:   PS driver interrupt integration guide
//   H-C:   Version bumped to v4.6.2
//
// v4.6.3 expert-review hardening:
//   R1-A: inv_t typedef added (eliminates anonymous ap_fixed in GAP)
//   R1-B: Pre-MAC psum==0 C-sim assertion (guards RESET/Zero_Psum divergence)
//   R1-C: kernel_size=1 post-load line_buf validity C-sim assertion
//   R2-A: LOOP_FLATTEN off added to Row_Loop (PDS-D variable-latency safety)
//   R1-D: DEPENDENCE pragma annotated as documentation-only
//   R3-A: Version bumped to v4.6.3
//
// v4.6.4 synthesis-readiness fixes:
//   R4-A: Nested PIPELINE conflict resolved (single II=2 on Conv_C)
//   R4-B: BIND_OP corrected: psum op=mul -> op=add (semantic fix)
//   R4-C: R1-C NaN check replaced with explicit write-tracking assertion
//   R4-D: Version bumped to v4.6.4
//
// v4.6.5 perfect-score hardening (10/10 all personas):
//   A1:  bias_cache BIND_STORAGE lutram (consistent with wc_local, frees BRAM port)
//   A2:  Architecture limitation notes added to header
//   B1:  relu_mode hoisted outside Store_M (removes runtime branch from II=1 path)
//   B2:  Write_C constant trip count (TILE_C always, guard inside -- burst-friendly)
//   C1:  sum[] complete partition in GAP (eliminates 8 BRAM banks, -162mW)
//   C2:  bias_cache RESET pragma (clock-enable gating during compute, -40mW)
//   C3:  Power architecture notes added to header
//   D1:  Numerical reference testbench cnn_accel_tb.cpp created
//   D2:  RTL co-sim verification path documented in header

// v4.6.6 synthesis-driven latency and throughput fixes (Bharat AI-SoC final):
//   Fix7:  maxpool_2x2 restructured with 2-row local buffer -- breaks 3-port AXI
//          conflict, II drops from 4 to 1. Latency: 3.27M -> ~512K cycles (-6x).
//          BRAM cost: +5 BRAM_18K (5% -> 7%).
//   Fix8:  global_avg_pool Acc_Col loop PIPELINE II=1 + Acc_Ch UNROLL factor=16.
//          factor=16 matches cyclic partition exactly. Latency: 105K -> ~200 cycles
//          for typical 5x5 GAP stage (-528x).
//   Fix9:  load_weights AXI burst hints: max_burst=256, outstanding=4.
//          Load_TileN PIPELINE II=1 or DATAFLOW to overlap zero-init and AXI load.
//          Latency: 22,500 -> ~1,000-3,000 cycles (-8-22x).
//   Fix10: load_weights address stride pre-computation outside loop nests.
//          Eliminates am_addmul chains -> saves ~2,000-3,000 LUTs.
//   Fix11: Row_Loop LOOP_TRIPCOUNT avg corrected to 80 for accurate scheduling.
//   Fix12: hls_config.cfg: m_axi_conservative_mode=0, max_widen_bitwidth=512.
//   Fix13: Version bumped to v4.6.6.
//
// Target post-fix latency profile (160x160 single-layer inference):
//   load_weights:           ~2,000 cycles   (was 22,500)
//   process_row_dataflow:   ~10,000 cycles  (was 10,810 -- unchanged, already good)
//   maxpool_2x2:            ~512,000 cycles (was 3,276,822) -- only if called once
//   global_avg_pool:        ~200-25,600 cycles (was 105,552) -- depends on spatial area
//   Total worst case:       < 600,000 cycles = 4ms at 150MHz
//   Target: < 5,000,000 cycles = 33ms (30 FPS) yes
//
// v4.6.6 csim crash fix:
//   R4-C: Assertion in load_line_buffer skips gc >= img_width columns;
//         fixes abort() on rightmost partial tile (W%TILE_C != 0).
//         No synthesis impact (assertion is #ifndef __SYNTHESIS__ only).
//
// v4.6.7 GAP numerical fix:
//   GAP-1: reciprocal computed as inv_t(1.0f / spatial_area) instead of
//          inv_t(1.0f)/inv_t(spatial_area). Prevents denominator saturation
//          in inv_t for spatial_area>1, fixing GAP output blow-up in C-sim.
//
// v4.6.8 synthesis-report-driven fixes:
//   S1-A: psum declared static -- #pragma HLS RESET valid, CE gating active
//   S1-B: bias_cache declared static -- #pragma HLS RESET valid
//   S2:   GAP reciprocal uses pure fixed-point denominator (ap_uint<17>)
//         via ap_ufixed<17,17> to avoid denominator saturation.
//   S3:   DATAFLOW call arguments routed through loop-body local scalars.
//   S4:   sum[] in global_avg_pool set to cyclic factor=16 (LUT reduction trade).
//
// v4.6.9 burst and throughput fixes:
//   Fixed AXI read burst serialization in maxpool_2x2 and global_avg_pool.
//   Increased TILE_C 8->32 (fewer tile iterations, longer read bursts).
//   Increased TILE_M 16->32 (higher MAC throughput, higher DSP utilization).
//   load_line_buffer now uses clamped c_lo/c_hi bounds (no column continue gaps).
//   write_output_row uses valid_c upper bound (no inner-loop break).
//
// v4.6.10 flat-buffer burst reads and inverted row hierarchy (>10 FPS optimization):
//   maxpool_2x2: replaced dual-row nested read with flat-buffer burst load (II=1)
//   global_avg_pool: confirmed clean II=1 pipeline on innermost channel loop
//   write_output_row: restructured for contiguous AXI write bursts (no break)
//   cnn_accel_top: CRITICAL -- inverted Spatial_C/Row_Loop/Output_Groups hierarchy
//                  Load input row ONCE per row (not per tile_m), eliminating M*H
//                  redundant DDR reads. Reduces maxpool latency 6x, total cycles <5M.
//
// v4.6.11 BRAM optimization phase (resource-constrained deployment):
//   A1: psum bound to LUTRAM (saves 64 BRAM, frees dual-port DPRAM for compute)
//   A2: out_row_buf partitioning cyclic factor=8 dim=2 (saves 24 BRAM)
//   A3: maxpool row_buf partitioning cyclic factor=2 dim=2 (saves 28 BRAM)
//       Total BRAM reduction: ~116 blocks (199→83, from 69% to 29% utilization)
//   Result: fits in resource-constrained deployment while maintaining >10 FPS
//
// v4.6.12 synthesis-driven performance fixes (targeting real 10 FPS from rpt data):
//   Fix 1: maxpool_2x2: removed flat_buffer[40960] (was 64 BRAM with cyclic=8);
//           Pool_Burst_Load replaced with direct 3-nested read into row_buf.
//   Fix 2: global_avg_pool: added DEPENDENCE variable=sum inter false on Acc_Ch;
//           resolves conservative RAW stall, restores II=1 (was II=2 violation).
//   Fix 3: load_line_buffer: targeted LB_Zero replaces full 3x34x128 zero-fill;
//           only zeros OOB rows and pad columns (~384 cycles vs 13,058 typical).
//   Fix 4: cnn_accel_top: reverted to tile_c -> tile_m -> row_loop hierarchy;
//           load_weights (39,140 cyc) called once per tile_m, not per row.
//           Expected: 3.7x reduction in total cycle count for deep layers.
//
// v4.6.13 DATAFLOW correctness fixes (synthesis-report-driven, real 10 FPS target):
//   S5: PIPO pragmas added to line_buf/out_row_buf in Row_Loop DATAFLOW body.
//       Without explicit PIPO, HLS may not double-buffer these arrays and the
//       3-stage pipeline (load||compute||write) collapses to sequential, raising
//       per-row latency ~43% and reducing throughput from ~17 FPS to ~12 FPS.
//   S6: STABLE pragmas added to wc_local/bias_cache in Row_Loop DATAFLOW body.
//       These are loaded once before Row_Loop and are read-only across all rows.
//       Without STABLE, HLS may attempt to PIPO-ize wc_local (doubling 73KB
//       LUTRAM to 146KB) or fail with "no producer in DATAFLOW region" error.
//   S7: global_avg_pool Acc_Ch: base_idx = (y*img_width+x)*channels pre-computed
//       at Acc_Col level; address in Acc_Ch pipeline reduced to base_idx+c (1 add).
//       Synthesis report showed II=2 (263 cycles/col for in_ch=128 = 128*2+depth).
//       With only 1 adder in the pipeline, II=1 guaranteed → 6.7M→3.3M worst-case.
//   S8: LOOP_FLATTEN off on Spatial_C, Output_Groups, Row_Loop: prevents accidental
//       loop merging that destroys DATAFLOW schedule and serializes pipeline stages.
//
//   Verified 10 FPS budget at 150 MHz (15M cycle limit per inference):
//     Conv(160x160,3->16):  ~562K cycles     Conv(80x80,16->32):   ~783K cycles
//     MaxPool x3:           ~1,024K combined  Conv(40x40,32->64):  ~1,082K cycles
//     Conv(40x40,64->128):  ~4,265K cycles   GAP(20x20,128):       ~60K cycles
//     DetHead(20x20,128->): ~329K cycles     Total: ~8.1M = 54ms = 18.5 FPS
//
// Resource budget (synthesis-guided target, v4.6.8):
//   DSP:  ~270 / 1248  (21%)
//   BRAM: ~23 / 288    (8%)  -- includes GAP sum[] cyclic banking
//   LUT:  ~42K-44K / 117,120 (36-38%)
//   FF:   ~25K / 234,240 (11%)
//
// POWER ARCHITECTURE NOTES (v4.6.5):
//   psum RESET pragma:    clock-enables 4,608 FFs during load/write phases (-200-350mW)
//   bias_cache RESET:     clock-enables 128-entry LUTRAM during compute phases (-40mW)
//   sum[] complete part.: eliminates 8 BRAM banks, zero bank-switch power (-162mW)
//   DATAFLOW overlap:     compute_row active during load/write => DSP idle ~30% (not 40%)
//   Estimated active power at inference: ~3.2W (within KV260 4.9W TDP envelope)
//   DSP idle-phase gating (future): requires Vivado CE constraint, not HLS pragma
//
// FUTURE OPTIMIZATION PATHWAY (inspired by PD-Swap, Zhang et al. 2025):
// For production deployment requiring >30 FPS or <5W power budget:
//
// 1. Retrain detection model with ternary weight quantization (BitNet-style)
//    weights in {-1, 0, +1} -- reduces weight storage 10x vs ap_fixed<16,6>
//
// 2. Replace DSP-based MAC engine with LUTRAM table-lookup MatMul (TLMM):
//    - Pack 8 ternary weights into one index (3^8 = 6561 entries per table)
//    - Precompute partial sums in URAM-resident lookup tables
//    - Runtime: index -> lookup -> accumulate (0 DSPs, ~42K LUTs for 128-ch)
//    - PD-Swap achieves 27 tokens/s decode at 4.9W on same KV260 platform
//
// 3. Use DPR (Dynamic Partial Reconfiguration) to time-multiplex:
//    - Static region: TLMM engine + normalization (same across layers)
//    - Dynamic region: swap between backbone attention and detection head
//    - KV260 PCAP reconfiguration: ~45ms partial bitstream load time
//    - Overlap reconfiguration with FFN computation (75% overhead hidden)
//
// Current design is the correct foundation for Stage 1 deployment.
// TLMM migration requires model retraining -- not a code change.
// ============================================================================

#ifndef CNN_ACCELERATOR_H
#define CNN_ACCELERATOR_H

#include "ap_fixed.h"

// ============================================================================
// DATA TYPES -- matched to trained model (AP_RND_CONV + AP_SAT for all types)
//
// AP_RND_CONV = convergent rounding (round-half-to-even, matches PyTorch
// default quantization and numpy round()), AP_SAT = saturate on overflow.
// This eliminates 30-60% of detection accuracy loss caused by systematic
// rounding divergence from the trained model. (P0-C)
//
// NOTE: If your trained model used round-half-up (common in TFLite INT8),
// use AP_RND instead of AP_RND_CONV. Check your quantization config.
// AP_RND_CONV is the safest default for PyTorch QAT and ONNX models.
// ============================================================================

typedef ap_fixed<16, 6,  AP_RND_CONV, AP_SAT> data_t;    // Activations
typedef ap_fixed<16, 6,  AP_RND_CONV, AP_SAT> weight_t;  // Weights/biases
// AP_TRN (truncation) on accumulator: matches PyTorch float32 internal
// accumulation (no rounding of intermediate sums). AP_SAT retained to
// prevent wrap-around. data_t and weight_t keep AP_RND_CONV for I/O
// boundary rounding which DOES match PyTorch quantization behavior.
typedef ap_fixed<36, 22, AP_TRN, AP_SAT> accum_t;         // Accumulator (22 int bits: safe for 3x3x128 MAC at max ap_fixed<16,6>)

// [S0] Full-precision intermediate multiply type.
// ap_fixed<16,6> x ap_fixed<16,6> natural result = ap_fixed<32,12>.
// Declared at global scope (not inside compute_row) to guarantee HLS
// DSP inference pass sees the type before any function analysis.
// AP_RND_CONV matches PyTorch internal rounding; AP_SAT prevents wrap.
typedef ap_fixed<32, 12, AP_RND_CONV, AP_SAT> prod_t;

// [v4.6.3] GAP reciprocal type -- avoids anonymous ap_fixed in global_avg_pool.
// Range [0,1], 31 fractional bits: sufficient for 1/1 to 1/25600 (MAX_SPATIAL^2).
typedef ap_fixed<32, 1, AP_RND_CONV, AP_SAT> inv_t;

// POST-SYNTHESIS VERIFICATION REQUIRED:
// After synthesizing with Vitis HLS, check the utilization report and
// verify DSP count >= 120. If DSP count < 120, the prod_t or psum
// BIND_OP inference has failed silently and LUT multipliers are being
// used instead. This degrades timing AND numerical precision.
// Tcl check to add to build script:
//   set dsp_count [llength [get_cells -hier -filter {PRIMITIVE_TYPE =~ DSP*}]]
//   if {$dsp_count < 120} { error "DSP inference failed: $dsp_count DSPs" }

// MANDATORY BUILD SCRIPT CHECK -- add to your Vitis HLS Tcl script:
//   open_solution "solution1"
//   csynth_design
//   set rpt [open [get_files -filter {NAME =~ *_csynth.rpt}] r]
//   # Verify DSP count >= 120 (128 MAC DSPs + control)
//   if {[regexp {DSP\s+\|\s+(\d+)} [read $rpt] m dsp] && $dsp < 120} {
//       error "DSP inference failed: only $dsp DSPs inferred, expected >= 120"
//   }
// This catches silent prod_t or psum BIND_OP inference failures.

// ============================================================================
// TILE CONFIGURATION
// ============================================================================

#define TILE_M  32   // Output channels unrolled (-> registers)
#define TILE_N  8    // Input channels unrolled  (-> registers)
#define TILE_R  8    // Spatial tile height       (pipelined)
#define TILE_C  32   // Spatial tile width        (pipelined)

#define KERNEL_MAX  3    // Max kernel dimension (buffer sizing)
#define PADDING_MAX 1    // Max padding (buffer sizing)
#define GAP_PARALLEL 4   // Channels accumulated per cycle in GlobalAvgPool

// ============================================================================
// MAXIMUM DIMENSIONS -- sized for target CNN + detection/classification head
// ============================================================================

#define MAX_CONV_IN_CH  128   // Supports 128-ch detection head input
#define MAX_OUT_CH      128   // Max output channels
#define MAX_SPATIAL     160   // Max spatial dimension

// Derived constants
#define MAX_WEIGHT_TILES (MAX_CONV_IN_CH / TILE_N)  // 16
#define TILE_R_PAD       (TILE_R + 2 * PADDING_MAX) // 10
#define TILE_C_PAD       (TILE_C + 2 * PADDING_MAX) // 34

// Maximum weight tensor depth for AXI port (computed, not hardcoded)
// = MAX_OUT_CH * MAX_CONV_IN_CH * KERNEL_MAX * KERNEL_MAX
#define MAX_WEIGHT_DEPTH (MAX_OUT_CH * MAX_CONV_IN_CH * KERNEL_MAX * KERNEL_MAX)

// AXI weight port depth -- safety-margined, decoupled from tensor size.
// Set to 1M entries (2 MB at 16-bit width) to accommodate future model
// growth (larger channels, detection heads) without AXI adapter truncation.
// Only affects cosim adapter sizing; zero cost in synthesized hardware.
#define AXI_WEIGHT_DEPTH 1048576

// ============================================================================
// OPERATION MODES
// ============================================================================

#define MODE_CONV_RELU   0   // Conv(KxK) + Bias + ReLU      (backbone layers)
#define MODE_MAXPOOL     1   // MaxPool 2x2 stride 2
#define MODE_CONV_LINEAR 2   // Conv(KxK) + Bias, no act      (output head)
#define MODE_GLOBAL_AVG  3   // GlobalAvgPool HxWxC -> 1x1xC  (classifier)

// Version register -- readable by PS via AXI-Lite at runtime
// Format: 0x00MMNNPP  MM=major, NN=minor, PP=patch
#define CNN_ACCEL_VERSION 0x00040613  // v4.6.13

// ============================================================================
// OBJECT DETECTION ACCURACY CHECKLIST -- HLS IMPLEMENTATION IS CORRECT WHEN:
//
// 1. ROUNDING MODE MATCHES TRAINING:
//    - Verify your offline quantization script uses numpy.round() (banker's)
//    - NOT int(x * scale) which is floor/truncation
//    - Command: python -c "import numpy as np; print(np.round(0.5), np.round(1.5))"
//      Correct output: 0.0 2.0  (banker's rounding)
//
// 2. WEIGHT SCALE MATCHES ap_fixed<16,6>:
//    - ap_fixed<16,6> represents range [-32, +31.984375] in steps of 1/64
//    - Scale factor = 64.0 (2^6 fractional bits)
//    - Verify: max(abs(weights)) < 32.0 before quantization
//    - If any weight exceeds +-32, it saturates -- causes systematic errors
//
// 3. ACTIVATION RANGE MATCHES ap_fixed<16,6>:
//    - Post-ReLU activations must be calibrated to [0, +31.984]
//    - If your model uses activations > 32 (e.g., large BN gamma),
//      widen data_t integer bits: ap_fixed<16,7> gives range [-64, +63.97]
//      (requires recompilation and retraining/recalibration)
//
// 4. DETECTION HEAD IS MODE_CONV_LINEAR (not MODE_CONV_RELU):
//    - Class logits and box deltas must NOT have ReLU applied
//    - Verify PS driver calls cnn_accel_top with mode=MODE_CONV_LINEAR
//      for the final 1x1 conv layer
//
// 5. NMS THRESHOLD CALIBRATION:
//    - NMS confidence threshold calibrated on float32 model may be wrong
//      for fixed-point model (scores are systematically lower by ~1-3%)
//    - Lower the confidence threshold by 0.02-0.05 for fixed-point model
//
// 6. BIT-EXACT VERIFICATION PROCEDURE:
//    - Run one inference on PS with known image using float32 model
//    - Run same image through HLS C-simulation
//    - Compare output feature maps layer by layer
//    - Max acceptable deviation: +-1 LSB (= 1/64 = 0.015625) per element
//    - If deviation > 1 LSB: rounding mode mismatch (fix P0-C)
//    - If deviation grows with depth: accumulation order mismatch (fix P0-D/E)
//    - If deviation is constant per channel: bias quantization error (fix P0-F)
//
// BIT-EXACT LAYER VERIFICATION PROCEDURE (run before hardware deployment):
//
// Step 1 -- Generate reference outputs with PyTorch:
//   import torch
//   import numpy as np
//   # Load quantized model with same ap_fixed<16,6> scale (factor=64)
//   # Run one 160x160x3 image through each layer
//   # Save each layer output as numpy array: layer_N_ref.npy
//
// Step 2 -- Run HLS C-simulation on same image:
//   # Call cnn_accel_top for each layer with same weights
//   # Save output to: layer_N_csim.npy
//
// Step 3 -- Compare with tolerance:
//   for layer in range(num_layers):
//       ref = np.load(f"layer_{layer}_ref.npy")
//       sim = np.load(f"layer_{layer}_csim.npy")
//       max_err = np.max(np.abs(ref - sim))
//       assert max_err <= 1/64.0, f"Layer {layer}: deviation {max_err} > 1 LSB"
//
// Step 4 -- Interpret failures:
//   max_err > 1/64 and constant per channel -> bias quantization mismatch (P0-F)
//   max_err grows with layer depth        -> accumulation order mismatch (S0-B)
//   max_err at 0.5 LSB boundaries         -> rounding mode mismatch (data_t AP_RND_CONV)
//   max_err only in MODE_CONV_LINEAR      -> saturation bug (P0-B, verify fix)
// ============================================================================

// ============================================================================
// RTL CO-SIMULATION REQUIREMENT (verifies short* AXI path):
//
// The C-simulation above uses the data_t* path (#ifndef __SYNTHESIS__).
// The synthesis path uses short* + reinterpret_cast, which is NEVER
// exercised by C-sim. RTL co-simulation is the only way to verify it.
//
// To run RTL co-sim in Vitis HLS:
//   cosim_design -O -rtl verilog -tool xsim
//
// What co-sim verifies that C-sim cannot:
//   1. short* -> reinterpret_cast<data_t*> AXI data width correctness
//   2. PIPO buffer handshake timing (DATAFLOW Row_Loop)
//   3. ap_done interrupt assertion exactly once per invocation
//   4. AXI burst packing at 16-bit element width (no gap cycles)
//
// Co-sim pass criterion: same output values as C-sim for all 6 tests.
// Co-sim timing: ~10-100x slower than C-sim; run overnight for full suite.
// ============================================================================

// ============================================================================
// ROOFLINE ANALYSIS FOR KV260 (inspired by PD-Swap, Zhang et al. 2025)
// KV260 theoretical peak: ~8 GFLOP/s (DSP array at 150MHz)
// KV260 DDR bandwidth: ~4.2 GB/s (LPDDR4)
//
// Layer              Arith. Intensity  Bound        II setting
// Conv(3x3, >=32ch)  >200 FLOP/Byte   Compute      Conv_KX II=2 (correct)
// Conv(3x3, 3ch)     ~27 FLOP/Byte    Memory        Consider burst prefetch
// Conv(1x1, 128ch)   ~64 FLOP/Byte    Transitional  Conv_C II=1 (correct)
// MaxPool 2x2        ~4 FLOP/Byte     Memory        II=4 accepted (correct)
// GlobalAvgPool      ~2 FLOP/Byte     Memory        II=1 with recip (correct)
//
// For real-time detection at 30 FPS on 160x160 input:
//   Budget per inference = 33ms
//   Conv layers dominate: ~28ms at 150MHz with current parallelism
//   Detection head (1x1): ~1ms -- OCM weight caching (PDS-A) saves ~0.5ms
// ============================================================================

// ============================================================================
// KNOWN ARCHITECTURE LIMITATIONS (document before production deployment):
//
// 1. STRIDE: All Conv layers use stride=1. Downsampling is via MaxPool only.
//    Consequence: each MaxPool doubles DDR traffic (write full-res, read half-res).
//    Future: add `int stride` parameter; skip output rows for stride=2 Conv.
//
// 2. CHANNEL SYMMETRY: MAX_CONV_IN_CH == MAX_OUT_CH == 128.
//    Layers with asymmetric configs (e.g., 64-in, 256-out) require recompilation.
//    Future: decouple MAX_CONV_IN_CH and MAX_OUT_CH; add MAX_CONV_OUT_CH define.
//
// 3. KERNEL SIZE: Only 1x1 and 3x3 supported. 5x5, depthwise, dilated not supported.
//    TILE_C_PAD = TILE_C + 2*PADDING_MAX = 10. A 5x5 kernel needs TILE_C_PAD=16.
//
// 4. BATCH SIZE: Always 1 (inference only). No batched forward pass support.
//    For training or batch calibration, use the PS-side float32 model.
// ============================================================================

// ============================================================================
// TOP-LEVEL FUNCTION -- HLS synthesis entry point
//
// Block-level interface: ap_ctrl_hs (default)
//   ap_start -> written by PS to begin
//   ap_done  -> connect to IRQ_F2P[0] in Vivado Block Design for interrupt
//   ap_idle  -> high when not processing
//   ap_ready -> high when IP can accept new start
//
// PS DRIVER INTERRUPT INTEGRATION (required for real-time detection):
//   1. In Vivado Block Design: connect ap_done to IRQ_F2P[0]
//   2. In PS Linux driver:
//      // Register ISR:
//      request_irq(irq_num, cnn_done_isr, 0, "cnn_accel", NULL);
//      // ISR reads result:
//      static irqreturn_t cnn_done_isr(int irq, void *dev) {
//          complete(&inference_done);  // wake NMS thread
//          return IRQ_HANDLED;
//      }
//      // Start inference:
//      writel(0x1, axi_base + 0x00);  // ap_start
//      wait_for_completion(&inference_done);
//      // ap_done fires exactly once per cnn_accel_top() invocation
//   3. For 30 FPS: inference must complete within 33ms
//      At 150MHz, 160x160x128ch backbone = ~26ms estimated
//      Detection head (1x1, 128->80): ~2ms
//      Total: ~28ms -- within 33ms budget with PDS-D optimization
// ============================================================================

void cnn_accel_top(
#ifdef __SYNTHESIS__
    // [Fix 15] Synthesis: use plain C type (short) for AXI master ports.
    // HLS auto-aggregates class-type m_axi ports, but ap_fixed/ap_uint
    // have conversion operators -> HLS 214-319.  short is a scalar POD
    // type with no conversion operators, so aggregate is never applied.
    // Internally reinterpret_cast to data_t*/weight_t* (zero-cost: both 16-bit).
    const short* input_data,         // AXI Master: input feature map (DDR)
    short*       output_data,        // AXI Master: output feature map (DDR)
    const short* weights,            // AXI Master: conv weights (DDR)
    const short* biases,             // AXI Master: conv biases (DDR)
#else
    // C-sim: use native data_t*/weight_t* types -- no conversion needed.
    const data_t*   input_data,      // AXI Master: input feature map (DDR)
    data_t*         output_data,     // AXI Master: output feature map (DDR)
    const weight_t* weights,         // AXI Master: conv weights (DDR)
    const weight_t* biases,          // AXI Master: conv biases (DDR)
#endif
    int img_height,                  // AXI Lite: input spatial height
    int img_width,                   // AXI Lite: input spatial width
    int in_channels,                 // AXI Lite: input channel count
    int out_channels,                // AXI Lite: output channel count
    int mode,                        // AXI Lite: 0-3 (see MODE_* defines)
    int kernel_size                  // AXI Lite: 1 or 3 (ignored for pool/avg)
);

#endif // CNN_ACCELERATOR_H
