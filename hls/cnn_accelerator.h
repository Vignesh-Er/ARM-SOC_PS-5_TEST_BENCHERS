// ============================================================================
// CNN Accelerator — Production Header (v4.6)
// Target: Xilinx Kria K26 SOM (xck26-sfvc784-2LV-c, ZU5EV)
// Clock:  ~150 MHz (6.67 ns, 20% uncertainty margin)
//
// Architecture: Input-stationary tiled convolution engine
//   - 128 parallel MACs (TILE_M=16 × TILE_N=8)
//   - Conv2D (3×3 or 1×1) + Bias + ReLU or linear
//   - MaxPool 2×2 stride 2
//   - Global Average Pooling
//   - AXI4 Master DDR + AXI4-Lite control
//   - 3-stage DATAFLOW: load_line_buffer ‖ compute_row ‖ write_output_row
//   - ap_done interrupt → PS IRQ for HW/SW co-design
//   - 16-bit AXI data ports (short* in synthesis, data_t* in C-sim)
//
// Supported layers (PS calls IP once per layer):
//   Conv(3→16,3×3) → Pool → Conv(16→32,3×3) → Pool →
//   Conv(32→64,3×3) → Conv(64→128,3×3) → Pool →
//   GlobalAvgPool → Conv(128→num_classes,1×1) [classification head]
//   OR Conv(128→det_ch,1×1) [detection head, NMS on PS]
//
// Weight format: NCHW [Cout][Cin][Ky][Kx] (burst-friendly DDR access)
//   BatchNorm MUST be pre-folded into weights/biases OFFLINE:
//     w_folded = w * gamma / sqrt(var + eps)
//     b_folded = (b - mean) * gamma / sqrt(var + eps) + beta
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
//   10. MAC parallelism doubled: TILE_N 4→8 (128 MACs, 10% DSP)
//   11. 3-stage DATAFLOW: load_weights ‖ compute_tile ‖ write_output
//   12. GAP: cyclic factor 4→16, 4-way parallel accumulation
//   13. AXI outstanding raised: reads 2→8, writes 2→4
//
// v4.5 architectural optimizations:
//   14. accum_t reduced from <48,24> to <32,20> (shorter DSP cascade)
//   15. BIND_STORAGE ram_2p removed from input_buf_local (HLS auto-infers)
//   16. BIND_OP impl=dsp on MAC psum (forces DSP48E2 utilization ~45%)
//   17. DATAFLOW removed from Output_Groups (sequential, no scheduler collapse)
//   18. Full input tile buffer replaced with line buffer (3 rows for 3×3)
//       input_buf[10][10][128] → line_buf[3][10][128] = 70% BRAM reduction
//       compute_tile → compute_row (psum 3D→2D, out_tile 3D→2D)
//       Row-by-row streaming with tile_m-outermost loop for weight reuse
//
// v4.6 pipeline & burst optimizations:
//   19. DATAFLOW in Row_Loop: load ‖ compute ‖ write (PIPO line_buf/out_row_buf)
//   20. wc_local bound to LUTRAM (frees ~128 BRAMs, +~6K LUTs)
//   21. load_line_buffer burst-safe: pre-zero + unconditional DDR reads
//   22. load_weights burst-safe: pre-zero + valid-range-only reads
//   23. global_avg_pool: removed UNROLL stepping, simple ch++ with II=1
//   24. maxpool_2x2: accept II=4 (bandwidth-limited, 2.4% of total cycles)
//   25. STABLE pragma on wc_local/bias_cache for DATAFLOW correctness
//
// Resource budget (estimated, v4.6):
//   DSP:  ~140 / 1248  (11%) — 128 MAC DSPs + control DSPs
//   BRAM: ~24 / 288    (8%)  — LUTRAM weights, PIPO line buf, AXI adapters
//   LUT:  ~35K / 117K  (30%) — includes ~6K LUTRAM for weight cache
//   FF:   ~17K / 234K  (7%)
// ============================================================================

#ifndef CNN_ACCELERATOR_H
#define CNN_ACCELERATOR_H

#include "ap_fixed.h"

// ============================================================================
// DATA TYPES — matched to trained Keras model (ap_fixed<16,6>)
// ============================================================================

typedef ap_fixed<16, 6>  data_t;    // Activations
typedef ap_fixed<16, 6>  weight_t;  // Weights/biases
typedef ap_fixed<32, 20> accum_t;   // Accumulator (20 int bits: safe for 3×3×128 MAC)

// ============================================================================
// TILE CONFIGURATION
// ============================================================================

#define TILE_M  16   // Output channels unrolled (→ registers)
#define TILE_N  8    // Input channels unrolled  (→ registers)
#define TILE_R  8    // Spatial tile height       (pipelined)
#define TILE_C  8    // Spatial tile width        (pipelined)

#define KERNEL_MAX  3    // Max kernel dimension (buffer sizing)
#define PADDING_MAX 1    // Max padding (buffer sizing)
#define GAP_PARALLEL 4   // Channels accumulated per cycle in GlobalAvgPool

// ============================================================================
// MAXIMUM DIMENSIONS — sized for target CNN + detection/classification head
// ============================================================================

#define MAX_CONV_IN_CH  128   // Supports 128-ch detection head input
#define MAX_OUT_CH      128   // Max output channels
#define MAX_SPATIAL     160   // Max spatial dimension

// Derived constants
#define MAX_WEIGHT_TILES (MAX_CONV_IN_CH / TILE_N)  // 16
#define TILE_R_PAD       (TILE_R + 2 * PADDING_MAX) // 10
#define TILE_C_PAD       (TILE_C + 2 * PADDING_MAX) // 10

// Maximum weight tensor depth for AXI port (computed, not hardcoded)
// = MAX_OUT_CH * MAX_CONV_IN_CH * KERNEL_MAX * KERNEL_MAX
#define MAX_WEIGHT_DEPTH (MAX_OUT_CH * MAX_CONV_IN_CH * KERNEL_MAX * KERNEL_MAX)

// AXI weight port depth — safety-margined, decoupled from tensor size.
// Set to 1M entries (2 MB at 16-bit width) to accommodate future model
// growth (larger channels, detection heads) without AXI adapter truncation.
// Only affects cosim adapter sizing; zero cost in synthesized hardware.
#define AXI_WEIGHT_DEPTH 1048576

// ============================================================================
// OPERATION MODES
// ============================================================================

#define MODE_CONV_RELU   0   // Conv(KxK) + Bias + ReLU      (backbone layers)
#define MODE_MAXPOOL     1   // MaxPool 2×2 stride 2
#define MODE_CONV_LINEAR 2   // Conv(KxK) + Bias, no act      (output head)
#define MODE_GLOBAL_AVG  3   // GlobalAvgPool H×W×C → 1×1×C   (classifier)

// ============================================================================
// TOP-LEVEL FUNCTION — HLS synthesis entry point
//
// Block-level interface: ap_ctrl_hs (default)
//   ap_start → written by PS to begin
//   ap_done  → connect to IRQ_F2P[0] in Vivado Block Design for interrupt
//   ap_idle  → high when not processing
//   ap_ready → high when IP can accept new start
// ============================================================================

void cnn_accel_top(
#ifdef __SYNTHESIS__
    // [Fix 15] Synthesis: use plain C type (short) for AXI master ports.
    // HLS auto-aggregates class-type m_axi ports, but ap_fixed/ap_uint
    // have conversion operators → HLS 214-319.  short is a scalar POD
    // type with no conversion operators, so aggregate is never applied.
    // Internally reinterpret_cast to data_t*/weight_t* (zero-cost: both 16-bit).
    const short* input_data,         // AXI Master: input feature map (DDR)
    short*       output_data,        // AXI Master: output feature map (DDR)
    const short* weights,            // AXI Master: conv weights (DDR)
    const short* biases,             // AXI Master: conv biases (DDR)
#else
    // C-sim: use native data_t*/weight_t* types — no conversion needed.
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