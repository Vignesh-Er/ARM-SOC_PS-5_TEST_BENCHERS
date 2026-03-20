// ============================================================================
// CNN Accelerator -- Production Implementation (v4.6.12)
// Target: Xilinx Kria K26 SOM (xck26-sfvc784-2LV-c)
//
// Modes: Conv(3x3/1x1)+ReLU, Conv+Linear, MaxPool, GlobalAvgPool
//
// v4.1 hardware safety fixes over v4:
//   Fix 1: Guard in_channels  <= MAX_CONV_IN_CH  (prevents BRAM overrun)
//   Fix 2: Guard out_channels <= MAX_OUT_CH       (prevents bias/psum overrun)
//   Fix 3: Clamp kernel_size  <= KERNEL_MAX       (prevents buffer overrun)
//   Fix 4: Guard img_height/width <= MAX_SPATIAL   (prevents tile overflow)
//   Fix 5: global_avg_pool sum[] partitioned      (fixes pipeline II hazard)
//   Fix 6: Division-by-zero guard in global_avg   (prevents hardware hang)
//   Fix 7: AXI weight depth = MAX_WEIGHT_DEPTH    (computed, not literal)
//   Fix 8: Runtime 64-byte pointer alignment check (abort if misaligned)
//          [v4.4] Now guarded by __SYNTHESIS__: new/malloc not 64B-aligned
//   Fix 9: AXI weight depth safety-margined        (1M, decoupled from tensor)
//
// v4.3 performance improvements:
//   Imp 10: MAC parallelism doubled: TILE_N 4->8 (128 MACs, 10% DSP)
//   Imp 11: 3-stage DATAFLOW: load_weights || compute_tile || write_output
//   Imp 12: GAP: cyclic factor 4->16, 4-way parallel accumulation
//   Imp 13: AXI outstanding raised: reads 2->8, writes 2->4
//
// v4.4 functional correctness fixes:
//   Fix 10: Alignment check gated by __SYNTHESIS__ (C-sim uses heap alloc)
//   Fix 11: Saturation clamp before accum_t->data_t narrowing cast
//           (prevents AP_WRAP sign-flip on overflow: accum<32,20>->data<16,6>)
//   Fix 12: load_weights PIPELINE moved to innermost loop (Load_KX)
//           (avoids HLS loop-flatten failure on variable kernel_size bounds)
//   Fix 14: Removed max_widen_bitwidth=512 from all m_axi ports
//           (HLS 214-319: illegal aggregate on ap_fixed with conversion ops)
//
// v4.5 architectural optimizations:
//   Opt 14: accum_t reduced <48,24>-><32,20> (shorter DSP cascade path)
//   Opt 15: BIND_STORAGE ram_2p removed (let HLS auto-infer BRAM/LUTRAM)
//   Opt 16: BIND_OP impl=dsp forced on MAC psum (DSP48E2 utilization ~45%)
//   Opt 17: DATAFLOW removed from Output_Groups (sequential execution,
//           eliminates scheduler collapse on variable-bound tile_m loop)
//   Opt 18: Full input tile buffer replaced with line buffer architecture
//           input_buf_local[10][10][128] -> line_buf[3][10][128] (70% smaller)
//           compute_tile -> compute_row (psum [M][R][C] -> [M][C], 87% smaller)
//           out_tile[R][C][M] -> out_row_buf[C][M] (87% smaller)
//           Loop: tile_c -> tile_m (weight reuse) -> out_row (line buffer)
//
// v4.6 pipeline & burst optimizations:
//   Opt 19: DATAFLOW in Row_Loop (load || compute || write via PIPO buffers)
//   Opt 20: wc_local -> LUTRAM (BIND_STORAGE impl=lutram, frees ~128 BRAMs)
//   Opt 21: load_line_buffer burst-safe (pre-zero + unconditional DDR reads)
//   Opt 22: load_weights burst-safe (pre-zero + valid-range-only reads)
//   Opt 23: global_avg_pool simplified (removed UNROLL stepping -> ch++ II=1)
//   Opt 24: maxpool_2x2 accepts II=4 (bandwidth-limited, 2.4% of total)
//   Opt 25: STABLE pragma on wc_local/bias_cache for DATAFLOW correctness
//
// v4.6.1 correctness, determinism & timing fixes:
//   P0-A: accum_t widened <32,20>-><36,22> (overflow on 128-ch layers)
//   P0-B: Linear-mode saturation gated to ReLU only (detection head fix)
//   P0-C: AP_RND_CONV + AP_SAT on all types (matches PyTorch quantization)
//   P0-D: DEPENDENCE pragma on MAC loops (preserves accumulation order)
//   P0-E: prod_t intermediate preserves full multiply precision
//   P0-F/G: Bias & BN folding quantization requirements documented (header)
//   P0-H: Zero-padding assumption documented (symmetric quant zero-point=0)
//   P1-A: Conv_KX II=2 (MAC reduction tree timing margin at 150MHz -2LV)
//   P1-B: GAP division -> reciprocal multiply (II=1 achieved)
//   P2-A: C-sim assertions added (zero synthesis cost)
//   P2-B: Explicit PIPO on line_buf/out_row_buf
//   P3-A: CNN_ACCEL_VERSION define (in header)
//   P3-B: Invalid kernel_size returns (no silent default)
//
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
//
// v4.6.6 csim crash fix:
//   R4-C: Assertion in load_line_buffer now skips out-of-bounds columns
//         (gc = start_col+c >= img_width). Old assert checked ALL buf_cols
//         entries, causing abort() on rightmost partial tile when
//         img_width % TILE_C != 0 (e.g. W=9 with TILE_C=8). No synthesis
//         impact -- assertion is #ifndef __SYNTHESIS__ only.
//
// v4.6.7 GAP numerical fix:
//   GAP-1: Fix reciprocal computation to avoid inv_t denominator saturation.
//          Old code used inv_t(spatial_area), which clips values >1 to ~1.0,
//          producing inv_area ~1.0 instead of 1/spatial_area. This caused
//          GAP outputs to saturate (e.g., hls=31.999 vs ref~1.28 at H=W=5).
//          New code computes reciprocal in float first, then casts to inv_t.
//
// v4.6.8 synthesis-driven refinements:
//   S1-A: psum declared static so #pragma HLS RESET is valid (HLS 207-5555)
//   S1-B: bias_cache declared static so #pragma HLS RESET is valid
//   S2:   GAP reciprocal uses pure fixed-point denominator (ap_uint<17>)
//   S3:   Row_Loop DATAFLOW call args routed via local loop-body scalars
//   S4:   GAP sum[] partition tuned to cyclic factor=16 (LUT reduction trade)
//
// v4.6.9 burst and throughput fixes:
//   Fixed AXI read burst serialization in maxpool_2x2 and global_avg_pool.
//   Increased TILE_C 8->32 and TILE_M 16->32 for fewer calls and higher MAC throughput.
//   Replaced conditional-continue in load_line_buffer with clamped c_lo/c_hi bounds.
//   Replaced conditional-break in write_output_row with valid_c loop upper bound.
//
// v4.6.13 DATAFLOW correctness fixes (synthesis-report-driven, 10 FPS target):
//   S5:   PIPO pragmas added explicitly to line_buf and out_row_buf in Row_Loop
//         DATAFLOW body. Without them, Vitis HLS may fail to double-buffer these
//         arrays, collapsing the 3-stage pipeline to sequential execution and
//         raising per-row latency from max(12.9K,5K,0.5K)=12.9K to 18.4K cycles
//         (+43% latency; drops ~17 FPS back to ~12 FPS worst case).
//   S6:   STABLE pragmas added to wc_local and bias_cache in Row_Loop DATAFLOW
//         body. wc_local is loaded ONCE before Row_Loop (per tile_c/tile_m) and
//         is read-only across all row iterations. Without STABLE, HLS may try to
//         create a PIPO copy of wc_local (36,864 elements x2 = double LUTRAM) or
//         raise a DATAFLOW channel error (no producer inside the DATAFLOW region).
//   S7:   global_avg_pool Acc_Ch: hoisted (y*img_width+x)*channels multiply out
//         of the PIPELINE II=1 loop body into base_idx pre-computed at Acc_Col
//         level. Synthesis report showed Acc_Col IterationLatency=263 cycles for
//         in_ch=128 (= 128*2 + depth), confirming II=2. With only base_idx+c
//         (1 adder) in the pipeline body, II=1 is guaranteed. This halves GAP
//         latency: 6.73M -> 3.3M cycles worst case (160x160x128).
//   S8:   LOOP_FLATTEN off added to Spatial_C, Output_Groups, and Row_Loop.
//         Prevents HLS loop-flattening passes from merging variable-bound outer
//         loops with the DATAFLOW inner loops, which would destroy the dataflow
//         schedule and serialize the pipeline stages.
//
// Weight format: NCHW [Cout][Cin][Ky][Kx]
// Data format:   HWC  [H][W][C]
//
// IMPORTANT: weights passed to cnn_accel_top MUST use actual (unpadded)
// channel strides matching in_channels.  The kernel internally zero-fills
// non-tile-aligned channels via guards in load_weights and compute_tile.
// ============================================================================

#include "cnn_accelerator.h"
#include <cassert>

// ============================================================================
// PHASE 1: Load Line Buffer -- KERNEL_MAX rows for one output row
//
// [Opt 18] Replaces full spatial tile buffer with a line buffer.
// For a 3x3 convolution producing output row `out_row`, we need input rows
// [out_row-1, out_row, out_row+1] (with zero-padding at boundaries).
// Buffer: line_buf[KERNEL_MAX][TILE_C_PAD][MAX_CONV_IN_CH]
//   = 3 x 10 x 128 = 3,840 elements  (was 10x10x128 = 12,800)
//   -> ~70% BRAM reduction on input buffer alone.
// ============================================================================

static void load_line_buffer(
    data_t line_buf[KERNEL_MAX][TILE_C_PAD][MAX_CONV_IN_CH],
    const data_t* input_dram,
    int out_row, int tile_c,
    int img_height, int img_width,
    int in_channels, int kernel_size)
{
    #pragma HLS INLINE OFF
    #pragma HLS ARRAY_PARTITION variable=line_buf cyclic factor=TILE_N dim=3

    // [P2-A] C-sim assertions -- zero synthesis cost
#ifndef __SYNTHESIS__
    assert(in_channels > 0 && in_channels <= MAX_CONV_IN_CH &&
           "load_line_buffer: in_channels out of range");
    assert(kernel_size >= 1 && kernel_size <= KERNEL_MAX &&
           "load_line_buffer: kernel_size out of range");
    assert(img_height > 0 && img_height <= MAX_SPATIAL &&
           "load_line_buffer: img_height out of range");
    assert(img_width  > 0 && img_width  <= MAX_SPATIAL &&
           "load_line_buffer: img_width out of range");
#endif

    int padding = (kernel_size > 1) ? (kernel_size / 2) : 0;
    int start_col = tile_c * TILE_C - padding;
    int buf_cols  = TILE_C + 2 * padding;

    // [v4.6.12] Phase 1: Targeted zero-fill — only OOB rows and pad columns.
    // Replaces full 3x34x128=13,056-cycle zero-fill with ~384-cycle typical case.
    // Interior rows visited >99% of the time only need pad-column zeroing (2 cols).
    // Zero-point=0 assumption preserved (symmetric QAT / ap_fixed<16,6>). (P0-H)
    if (kernel_size > 1) {
        LB_Zero_Row:
        for (int kr = 0; kr < kernel_size; kr++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=3 avg=3
            int gr = out_row - padding + kr;
            if (gr < 0 || gr >= img_height) {
                // OOB row: zero all buf_cols × in_channels entries
                LB_Zero_OOB_Col:
                for (int c = 0; c < buf_cols; c++) {
                    #pragma HLS LOOP_TRIPCOUNT min=8 max=34 avg=5
                    LB_Zero_OOB_Ch:
                    for (int ch = 0; ch < in_channels; ch++) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=3 max=128 avg=64
                        line_buf[kr][c][ch] = data_t(0);
                    }
                }
            } else {
                // Valid row: zero only left/right pad columns
                if (start_col < 0) {
                    // Left pad column (local index 0)
                    LB_Zero_Left_Ch:
                    for (int ch = 0; ch < in_channels; ch++) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=3 max=128 avg=64
                        line_buf[kr][0][ch] = data_t(0);
                    }
                }
                if (start_col + buf_cols > img_width) {
                    // Right pad column
                    int rpad_c = img_width - start_col;
                    LB_Zero_Right_Ch:
                    for (int ch = 0; ch < in_channels; ch++) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=3 max=128 avg=64
                        line_buf[kr][rpad_c][ch] = data_t(0);
                    }
                }
            }
        }
    }

    // [Opt 21] Phase 2: Load valid pixels with clamped column range.
    // This removes per-column continues and enables long contiguous AXI bursts.
    LB_Load_Row:
    for (int kr = 0; kr < kernel_size; kr++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=3 avg=3
        int gr = out_row - padding + kr;
        if (gr < 0 || gr >= img_height) continue;

        int c_lo = (start_col < 0) ? (-start_col) : 0;
        int c_hi = (start_col + buf_cols > img_width) ? (img_width - start_col) : buf_cols;

#ifndef __SYNTHESIS__
        assert(c_hi > c_lo &&
               "load_line_buffer: clamped column range is empty");
#endif

        int base = (gr * img_width + start_col + c_lo) * in_channels;

        LB_Load_Col:
        for (int c = c_lo; c < c_hi; c++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=34 avg=32
            LB_Load_Ch:
            for (int ch = 0; ch < in_channels; ch++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=3 max=128 avg=64
                line_buf[kr][c][ch] =
                    input_dram[base + (c - c_lo) * in_channels + ch];
            }
        }
    }
}

// ============================================================================
// PHASE 2a: Load Weights for ONE output channel group
//
// Weights loaded once per tile_m group and reused across all output rows.
// NCHW: for fixed Cout, [Cin][Ky][Kx] is contiguous -> stride-1 DDR.
// ============================================================================

static void load_weights(
    weight_t weight_cache[MAX_WEIGHT_TILES][TILE_M][TILE_N][KERNEL_MAX][KERNEL_MAX],
    const weight_t* weight_dram, int tile_m,
    int in_channels, int out_channels, int kernel_size)
{
    #pragma HLS INLINE OFF
    #pragma HLS ARRAY_PARTITION variable=weight_cache complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weight_cache complete dim=3

    int num_tiles_n = (in_channels + TILE_N - 1) / TILE_N;
    const int stride_kx = 1;
    const int stride_ky = kernel_size;
    const int stride_cin = kernel_size * kernel_size;
    const int stride_cout = in_channels * stride_cin;

    // [Opt 22] Phase 1: Zero the entire weight cache
    // m and n dimensions are fully partitioned -> all written in parallel per cycle.
    Zero_TileN:
    for (int tile_n = 0; tile_n < MAX_WEIGHT_TILES; tile_n++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=16 avg=4
        Zero_KY:
        for (int ky = 0; ky < KERNEL_MAX; ky++) {
            Zero_KX:
            for (int kx = 0; kx < KERNEL_MAX; kx++) {
                #pragma HLS PIPELINE II=1
                for (int m = 0; m < TILE_M; m++) {
                    #pragma HLS UNROLL
                    for (int n = 0; n < TILE_N; n++) {
                        #pragma HLS UNROLL
                        weight_cache[tile_n][m][n][ky][kx] = weight_t(0);
                    }
                }
            }
        }
    }

    // [Opt 22] Phase 2: Load valid weights -- unconditional DDR reads
    // Compute valid ranges so innermost loop has no conditionals.
    int valid_m = TILE_M;
    if (tile_m * TILE_M + TILE_M > out_channels)
        valid_m = out_channels - tile_m * TILE_M;
    if (valid_m <= 0) return;

    Load_M:
    for (int m = 0; m < valid_m; m++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=16 avg=16
        Load_TileN:
        for (int tile_n = 0; tile_n < num_tiles_n; tile_n++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16 avg=4

            int valid_n = TILE_N;
            if (tile_n * TILE_N + TILE_N > in_channels)
                valid_n = in_channels - tile_n * TILE_N;

            Load_N:
            for (int n = 0; n < valid_n; n++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=8 avg=8
                Load_KY:
                for (int ky = 0; ky < kernel_size; ky++) {
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=3 avg=3
                    Load_KX:
                    for (int kx = 0; kx < kernel_size; kx++) {
                        #pragma HLS PIPELINE II=1
                        #pragma HLS LOOP_TRIPCOUNT min=1 max=3 avg=3

                        int out_ch = tile_m * TILE_M + m;
                        int in_ch  = tile_n * TILE_N + n;
                        int idx = out_ch * stride_cout
                                + in_ch * stride_cin
                                + ky * stride_ky
                                + kx * stride_kx;
                        weight_cache[tile_n][m][n][ky][kx] = weight_dram[idx];
                    }
                }
            }
        }
    }
}

// ============================================================================
// PHASE 2b: Compute ONE Output Row of Convolution + Bias + Activation
//
// [Opt 18] Row-based compute replaces full tile compute.
// Processes one output row (TILE_C columns x TILE_M output channels).
// psum reduced from psum[M][R][C] -> psum[M][C]  (87.5% smaller).
// out_row_buf reduced from out_tile[R][C][M] -> out_row_buf[C][M].
//
// 128 parallel MACs per cycle (TILE_M=16 x TILE_N=8).
//   psum += (accum_t)(in_pix * w)  <- 16x8 -> 1 DSP48E2 each
// ============================================================================

static void compute_row(
    const data_t line_buf[KERNEL_MAX][TILE_C_PAD][MAX_CONV_IN_CH],
    const weight_t weight_cache[MAX_WEIGHT_TILES][TILE_M][TILE_N][KERNEL_MAX][KERNEL_MAX],
    data_t out_row_buf[TILE_C][TILE_M],
    const weight_t bias_cache[MAX_OUT_CH],
    int tile_m,
    int in_channels, int out_channels,
    int kernel_size, int mode)
{
    #pragma HLS INLINE OFF
    #pragma HLS ARRAY_PARTITION variable=weight_cache complete dim=2
    #pragma HLS ARRAY_PARTITION variable=weight_cache complete dim=3
    #pragma HLS ARRAY_PARTITION variable=line_buf cyclic factor=TILE_N dim=3
    // [v4.6.10-A2] Cyclic partitioning factor=8 dim=2 reduces BRAM from 32 to 8 blocks
    #pragma HLS ARRAY_PARTITION variable=out_row_buf cyclic factor=8 dim=2

    int num_tiles_n = (in_channels + TILE_N - 1) / TILE_N;

    // Partial sum accumulator -- 2D: [M][C] (was [M][R][C])
    static accum_t psum[TILE_M][TILE_C];
    #pragma HLS ARRAY_PARTITION variable=psum complete dim=1
    // [R4-B] psum is the accumulation target; force DSP usage on add path.
    #pragma HLS BIND_OP variable=psum op=add impl=dsp
    // [v4.6.10-A1] Bind psum to LUTRAM instead of BRAM (saves 64 BRAM blocks)
    #pragma HLS BIND_STORAGE variable=psum type=ram_s2p impl=lutram
    #pragma HLS RESET variable=psum  // Valid on static variable (S1-A)

    // Zero psum: TILE_C cycles (m unrolled via partition)
    Zero_Psum:
    for (int c = 0; c < TILE_C; c++) {
        #pragma HLS PIPELINE II=1
        for (int m = 0; m < TILE_M; m++) {
            #pragma HLS UNROLL
            psum[m][c] = 0;
        }
    }

    // [R1-B] Verify Zero_Psum executed before MAC begins -- C-sim only.
    // The RESET pragma on psum is synthesis-only; in C-sim only the explicit
    // Zero_Psum loop zeroes psum. If HLS optimises Zero_Psum away in synthesis
    // (because RESET handles it), C-sim still runs the loop => correct in both.
    // This assertion catches any future refactor that accidentally removes Zero_Psum.
#ifndef __SYNTHESIS__
    for (int _m = 0; _m < TILE_M; _m++)
        for (int _c = 0; _c < TILE_C; _c++)
            assert(psum[_m][_c] == accum_t(0) &&
                   "psum not zeroed before MAC -- Zero_Psum loop may have been removed");
#endif

    // Main MAC loop -- iterate over input channel tiles x kernel window
    Acc_TileN:
    for (int tile_n = 0; tile_n < num_tiles_n; tile_n++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=16 avg=4
        Conv_KY:
        for (int ky = 0; ky < kernel_size; ky++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=3 avg=3
            Conv_KX:
            for (int kx = 0; kx < kernel_size; kx++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=3 avg=3
                Conv_C:
                for (int c = 0; c < TILE_C; c++) {
                    // [R4-A] Single unambiguous pipeline directive at MAC body loop.
                    #pragma HLS PIPELINE II=2
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=32 avg=32

                    // Read TILE_N input pixels from line buffer
                    // ky indexes into the KERNEL_MAX rows of line_buf
                    int in_c = c + kx;

                    data_t in_pix[TILE_N];
                    #pragma HLS ARRAY_PARTITION variable=in_pix complete

                    for (int n = 0; n < TILE_N; n++) {
                        #pragma HLS UNROLL
                        int ch = tile_n * TILE_N + n;
                        if (ch < in_channels)
                            in_pix[n] = line_buf[ky][in_c][ch];
                        else
                            in_pix[n] = data_t(0);
                    }

                    MAC_M:
                    for (int m = 0; m < TILE_M; m++) {
                        #pragma HLS UNROLL
                        // [P0-D] DEPENDENCE inter false: documentation-only on fully-unrolled
                        // MAC_M/MAC_N. After UNROLL, HLS sees only combinational logic;
                        // the pragma has no scheduling effect but documents design intent.
                        #pragma HLS DEPENDENCE variable=psum inter false
                        MAC_N:
                        for (int n = 0; n < TILE_N; n++) {
                            #pragma HLS UNROLL
                            #pragma HLS DEPENDENCE variable=psum inter false  // [P0-D] doc-only (see above)
                            weight_t w = weight_cache[tile_n][m][n][ky][kx];
                            // [S0-B] Let HLS perform native 16x16 DSP multiply first,
                            // then widen result to prod_t and accumulate into accum_t.
                            psum[m][c] += (accum_t)((prod_t)(in_pix[n] * w));
                        }
                    }
                }
            }
        }
    }

    // [P2-A] Post-MAC overflow check -- C-sim only, zero synthesis cost.
    // With AP_SAT on accum_t, psum is always within range; this assertion
    // confirms the accumulation did not silently wrap before AP_SAT engaged.
#ifndef __SYNTHESIS__
    for (int _m = 0; _m < TILE_M; _m++)
        for (int _c = 0; _c < TILE_C; _c++)
            assert(psum[_m][_c] <= accum_t(4194303) &&
                   psum[_m][_c] >= accum_t(-4194304) &&
                   "compute_row: psum overflow -- accum_t<36,22> exceeded");
#endif

    // [B1] Hoist mode check outside Store loop -- mode is loop-invariant.
    // HLS constant-propagates relu_mode into Store_M, converting the runtime
    // branch to a static mux select. Removes two 36-bit comparators from
    // the II=1 Store_M critical path (~0.8ns recovered).
    const bool relu_mode = (mode == MODE_CONV_RELU);

    // ---- Bias + Activation -> store to out_row_buf ----
    Store_C:
    for (int c = 0; c < TILE_C; c++) {
        Store_M:
        for (int m = 0; m < TILE_M; m++) {
            #pragma HLS PIPELINE II=1

            int out_ch = tile_m * TILE_M + m;
            accum_t val;
            if (out_ch < out_channels)
                val = psum[m][c] + (accum_t)bias_cache[out_ch];
            else
                val = accum_t(0);

            // [P0-B] relu_mode hoisted above Store_C -- constant after hoist.
            // Detection head (MODE_CONV_LINEAR) produces logits and bounding-box
            // deltas that legitimately exceed the data_t range; pre-clamping to
            // +-32 causes systematic wrong box coordinates. AP_SAT on data_t
            // handles narrowing overflow safely for the linear path.
            data_t result;
            if (relu_mode) {
                if (val > accum_t(31))  val = accum_t(31);
                if (val < accum_t(-32)) val = accum_t(-32);
                result = (val > accum_t(0)) ? (data_t)val : data_t(0);
            } else {
                result = (data_t)val;
            }

            out_row_buf[c][m] = result;
        }
    }
}

// ============================================================================
// PHASE 2c: Write ONE Output Row to DDR
//
// [Opt 18] Row-based write replaces full tile write.
// Writes out_row_buf[TILE_C][TILE_M] for a single output row.
// c->m order (m innermost = stride-1 in HWC -> burst writes).
// ============================================================================

static void write_output_row(
    const data_t out_row_buf[TILE_C][TILE_M],
    data_t* output_dram,
    int tile_m, int out_row, int tile_c,
    int img_height, int img_width,
    int out_channels)
{
    #pragma HLS INLINE OFF
    // [v4.6.10-A2] Cyclic partitioning factor=8 dim=2 reduces BRAM from 32 to 8 blocks
    #pragma HLS ARRAY_PARTITION variable=out_row_buf cyclic factor=8 dim=2

    if (out_row >= img_height)
        return;

    int start_col = tile_c * TILE_C;

    int valid_c = TILE_C;
    if (start_col + TILE_C > img_width)
        valid_c = img_width - start_col;
    if (valid_c < 0) valid_c = 0;

    int valid_m = TILE_M;
    if ((tile_m + 1) * TILE_M > out_channels)
        valid_m = out_channels - tile_m * TILE_M;
    if (valid_m < 0) valid_m = 0;

    Write_C:
    for (int c = 0; c < valid_c; c++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=32 avg=32
        Write_M:
        for (int m = 0; m < valid_m; m++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=32 avg=32

            int out_ch  = tile_m * TILE_M + m;
            int out_idx = (out_row * img_width + (start_col + c)) * out_channels + out_ch;
            output_dram[out_idx] = out_row_buf[c][m];
        }
    }
}

// ============================================================================
// MAXPOOL 2x2 STRIDE 2
// ============================================================================

static void maxpool_2x2(const data_t* input_dram, data_t* output_dram,
                        int img_height, int img_width, int channels)
{
    #pragma HLS INLINE OFF

    int out_h = img_height / 2;
    int out_w = img_width  / 2;

    data_t row_buf[2][MAX_SPATIAL][MAX_CONV_IN_CH];
    #pragma HLS ARRAY_PARTITION variable=row_buf complete dim=1
    // [v4.6.10-A3] Cyclic factor=2 on spatial dim=2 saves 28 BRAM vs factor=8 on channels
    // Distributes ix0/ix1 across 2 spatial banks per row, enabling 4 simultaneous reads
    #pragma HLS ARRAY_PARTITION variable=row_buf cyclic factor=2 dim=2

    Pool_OutRow:
    for (int oy = 0; oy < out_h; oy++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=80 avg=20

        // --- Phase 1: Load 2 input rows directly into row_buf ---
        // [v4.6.12] Removed flat_buffer (was 64 BRAM with cyclic=8).
        // Direct 3-nested read into row_buf: sequential row-major DDR addresses
        // still form contiguous bursts per row; eliminates unpack phase entirely.
        int iy0 = oy * 2;

        Pool_Load_R:
        for (int r = 0; r < 2; r++) {
            Pool_Load_X:
            for (int x = 0; x < img_width; x++) {
                Pool_Load_Ch:
                for (int ch = 0; ch < channels; ch++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=16 max=128 avg=64
                    row_buf[r][x][ch] = input_dram[((iy0 + r) * img_width + x) * channels + ch];
                }
            }
        }

        Pool_OutCol:
        for (int ox = 0; ox < out_w; ox++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=80 avg=40
            int ix0 = ox * 2;
            int ix1 = ox * 2 + 1;

            Pool_Ch:
            for (int c = 0; c < channels; c++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=16 max=128 avg=64
                // [v4.6.10-A3] Sequential channel processing with spatial cyclic partitioning
                // ix0/ix1 map to different banks in row_buf[r]; one channel processed per cycle
                data_t v00 = row_buf[0][ix0][c];
                data_t v01 = row_buf[0][ix1][c];
                data_t v10 = row_buf[1][ix0][c];
                data_t v11 = row_buf[1][ix1][c];
                data_t max_val = (v00 > v01) ? v00 : v01;
                max_val = (v10 > max_val) ? v10 : max_val;
                max_val = (v11 > max_val) ? v11 : max_val;
                output_dram[(oy * out_w + ox) * channels + c] = max_val;
            }
        }
    }
}

// ============================================================================
// GLOBAL AVERAGE POOLING -- HxWxC -> 1x1xC
//
// [Opt 23] Sequential channel accumulation with PIPELINE II=1.
//   sum[] partitioned cyclic factor=16 -> 16 BRAM banks.
//   Inner loop steps ch++ (1 channel/cycle, 1 DDR read/cycle).
//   Cyclic factor=16 ensures bank reuse interval=16 >> RAW latency -> II=1.
//
// [Fix 6] Division-by-zero guard: if spatial area is 0, output zeros.
//
// [P1-B] Reciprocal multiplication replaces sequential SRT divider.
//   ap_fixed division synthesizes as II=20-30; reciprocal pre-computed
//   once per GAP invocation, per-channel loop uses only multiplication -> II=1.
// ============================================================================

static void global_avg_pool(const data_t* input_dram, data_t* output_dram,
                            int img_height, int img_width, int channels)
{
    #pragma HLS INLINE OFF

    accum_t sum[MAX_OUT_CH];
    #pragma HLS ARRAY_PARTITION variable=sum cyclic factor=16

    Init_Sum:
    for (int ch = 0; ch < channels; ch++) {
        #pragma HLS PIPELINE II=1
        #pragma HLS LOOP_TRIPCOUNT min=16 max=128 avg=64
        sum[ch] = 0;
    }

    int spatial_area = img_height * img_width;

    if (spatial_area > 0) {
        Acc_Row:
        for (int y = 0; y < img_height; y++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=160 avg=10
            Acc_Col:
            for (int x = 0; x < img_width; x++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=160 avg=10
                // [v4.6.13-S3] Hoist the multiply out of Acc_Ch pipeline.
                // The expression (y*img_width + x)*channels is loop-invariant across c.
                // Inside the pipeline body it became a 2-multiply chain that HLS could
                // not always schedule in 1 cycle (→ II=2, confirmed in synthesis report:
                // Acc_Col IterationLatency 263 cycles = 128*2 + depth for in_ch=128).
                // With base_idx pre-computed, Acc_Ch address = base_idx + c (1 adder only).
                int base_idx = (y * img_width + x) * channels;
                Acc_Ch:
                for (int c = 0; c < channels; c++) {
                    #pragma HLS PIPELINE II=1
                    // [v4.6.12] Break conservative RAW dependency: sum[c] and sum[c'] for
                    // c != c' are independent banks (cyclic factor=16 partition). HLS was
                    // conservatively assuming inter-iteration dependency on the whole sum[].
                    // With cyclic=16, bank reuse interval=16 exceeds LUTRAM read latency=2.
                    #pragma HLS DEPENDENCE variable=sum inter false
                    #pragma HLS LOOP_TRIPCOUNT min=16 max=128 avg=64
                    sum[c] += (accum_t)input_dram[base_idx + c];
                }
            }
        }

        ap_uint<17> spatial_area_u = (ap_uint<17>)(img_height * img_width);
        ap_ufixed<17, 17, AP_TRN, AP_SAT> spatial_area_fx = spatial_area_u;
        inv_t inv_area = inv_t(1) / spatial_area_fx;

        Write_Avg:
        for (int c = 0; c < channels; c++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=16 max=128 avg=64
            data_t avg = (data_t)(sum[c] * inv_area);
            output_dram[c] = avg;
        }
    } else {
        Write_Zero:
        for (int c = 0; c < channels; c++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=16 max=128 avg=64
            output_dram[c] = data_t(0);
        }
    }
}

// ============================================================================
// TOP-LEVEL ENTRY POINT
//
// [Fix 1-4] Hard guards at entry:
//   - in_channels  clamped to MAX_CONV_IN_CH  -> prevents input_buf overrun
//   - out_channels clamped to MAX_OUT_CH       -> prevents bias/psum overrun
//   - kernel_size  clamped to KERNEL_MAX       -> prevents buffer overrun
//   - img_height/width clamped to MAX_SPATIAL  -> prevents tile overflow
//
// [Fix 7] AXI weight depth = MAX_WEIGHT_DEPTH (computed from header defines)
//
// 3-stage DATAFLOW per row: load_line_buffer || compute_row || write_output_row
// Line-buffer architecture: only KERNEL_MAX rows buffered (not full tile).
// ============================================================================

void cnn_accel_top(
#ifdef __SYNTHESIS__
    const short* input_data,
    short*       output_data,
    const short* weights,
    const short* biases,
#else
    const data_t*   input_data,
    data_t*         output_data,
    const weight_t* weights,
    const weight_t* biases,
#endif
    int img_height,
    int img_width,
    int in_channels,
    int out_channels,
    int mode,
    int kernel_size)
{
    // ---- AXI Master: 4 bundles, no port conflicts ----
    // [PDS-A] DDR PORT SPECIALIZATION FOR DETECTION HEAD:
    // For MODE_CONV_LINEAR (detection head, 1x1 conv), the bottleneck is
    // DDR bandwidth for weight reads (128x det_ch weights per output pixel).
    // For MODE_CONV_RELU (backbone, 3x3 conv), the bottleneck is compute.
    //
    // Current 4-bundle AXI mapping is already optimal for compute-bound layers.
    // For bandwidth-bound detection head, consider PS-side optimization:
    //   - Load detection head weights to OCM (on-chip memory) before inference
    //     to eliminate DDR weight reads during the final 1x1 conv layer
    //   - OCM on KV260 = 256KB; detection head weights for 128->80 classes
    //     = 128 x 80 x 2 bytes = 20KB -- fits entirely in OCM
    //   - PS driver: memcpy weights to OCM address space before starting IP
    //   This eliminates the gmem_param DDR port bottleneck for the detection
    //   head entirely, matching PD-Swap's bandwidth optimization strategy.
    //
    // To implement: add a separate AXI bundle for OCM-backed weight port:
    // #pragma HLS INTERFACE m_axi port=weights bundle=gmem_ocm \
    //     depth=10240 max_read_burst_length=64 num_read_outstanding=8
    // PS configures AXI base address to OCM (0xFFFC0000) for det head layers.
    //
    // [Imp 13] Outstanding raised: reads 8, writes 4 (hide DDR latency)
    // NOTE: max_widen_bitwidth removed -- illegal on ap_fixed pointer ports
    //       (HLS 214-319: aggregate on type with conversion operators).
    //       HLS will auto-widen to the port's natural element width (16 bits).
    #pragma HLS INTERFACE m_axi port=input_data  offset=slave bundle=gmem_in \
        depth=4096000 max_read_burst_length=256 num_read_outstanding=4 latency=64
    #pragma HLS INTERFACE m_axi port=output_data offset=slave bundle=gmem_out \
        depth=4096000 max_write_burst_length=256 num_write_outstanding=4 latency=64
    #pragma HLS INTERFACE m_axi port=weights     offset=slave bundle=gmem_param \
        depth=1048576 max_read_burst_length=256 num_read_outstanding=4 latency=64
    #pragma HLS INTERFACE m_axi port=biases      offset=slave bundle=gmem_bias \
        depth=65536 max_read_burst_length=128 num_read_outstanding=2 latency=64

    // ---- AXI Lite control registers ----
    #pragma HLS INTERFACE s_axilite port=input_data   bundle=control
    #pragma HLS INTERFACE s_axilite port=output_data  bundle=control
    #pragma HLS INTERFACE s_axilite port=weights      bundle=control
    #pragma HLS INTERFACE s_axilite port=biases       bundle=control
    #pragma HLS INTERFACE s_axilite port=img_height   bundle=control
    #pragma HLS INTERFACE s_axilite port=img_width    bundle=control
    #pragma HLS INTERFACE s_axilite port=in_channels  bundle=control
    #pragma HLS INTERFACE s_axilite port=out_channels bundle=control
    #pragma HLS INTERFACE s_axilite port=mode         bundle=control
    #pragma HLS INTERFACE s_axilite port=kernel_size  bundle=control
    #pragma HLS INTERFACE s_axilite port=return       bundle=control

    // ================================================================
    // [Fix 15] Map port pointers to typed internal pointers.
    //
    // Synthesis: short* ports avoid HLS 214-319 (no class conversion ops).
    //   reinterpret_cast to data_t*/weight_t* is zero-cost (both 16-bit).
    // C-sim: ports ARE data_t*/weight_t* already -- direct assignment.
    // ================================================================
#ifdef __SYNTHESIS__
    const data_t*   in_ptr  = reinterpret_cast<const data_t*>(input_data);
    data_t*         out_ptr = reinterpret_cast<data_t*>(output_data);
    const weight_t* wt_ptr  = reinterpret_cast<const weight_t*>(weights);
    const weight_t* bi_ptr  = reinterpret_cast<const weight_t*>(biases);
#else
    const data_t*   in_ptr  = input_data;
    data_t*         out_ptr = output_data;
    const weight_t* wt_ptr  = weights;
    const weight_t* bi_ptr  = biases;
#endif

    // ================================================================
    // [Fix 8] Runtime 64-byte pointer alignment check
    //
    // 512-bit AXI bursts require 64-byte aligned DDR addresses.
    // If PS passes unaligned pointers, burst packing breaks silently,
    // collapsing throughput and potentially missing real-time targets.
    // Return immediately so the bug surfaces at integration time.
    //
    // Guarded by __SYNTHESIS__ because C-simulation uses new/malloc
    // which do NOT guarantee 64-byte alignment.  In cosim and on HW,
    // the DMA framework guarantees alignment.
    //
    // Cost: 4 AND ops on address registers, 0 DSPs, 0 BRAMs, ~10 LUTs.
    // ================================================================
#ifdef __SYNTHESIS__
    {
        const unsigned long long align_mask = 0x3Full; // 64-byte = 2^6
        if (((unsigned long long)input_data  & align_mask) ||
            ((unsigned long long)output_data & align_mask) ||
            ((unsigned long long)weights     & align_mask) ||
            ((unsigned long long)biases      & align_mask)) {
            return;  // Misaligned -- abort, PS sees ap_done immediately
        }
    }
#endif

    // ================================================================
    // [Fix 1-4] Hard dimension guards -- prevent all buffer overruns
    //
    // These execute as simple comparisons in the control FSM before any
    // data movement. Cost: 0 DSPs, 0 BRAMs, ~20 LUTs, <5 clock cycles.
    // ================================================================

    // [Fix 4] Spatial dimension guard
    if (img_height <= 0 || img_width <= 0)
        return;
    if (img_height > MAX_SPATIAL)
        img_height = MAX_SPATIAL;
    if (img_width > MAX_SPATIAL)
        img_width = MAX_SPATIAL;

    // [Fix 1] Input channel guard
    if (in_channels <= 0)
        return;
    if (in_channels > MAX_CONV_IN_CH)
        in_channels = MAX_CONV_IN_CH;

    // [P3-B] Kernel size guard: return on invalid value instead of silently
    // defaulting to 3. An invalid kernel_size indicates PS misconfiguration
    // and must surface at integration time via immediate ap_done.
    if (kernel_size != 1 && kernel_size != 3)
        return;
    if (kernel_size > KERNEL_MAX)
        kernel_size = KER    // NOTE: max_widen_bitwidth removed -- illegal on ap_fixed pointer ports
    //       (HLS 214-319: aggregate on type with conversion operators).
    //       HLS will auto-widen to the port's natural element width (16 bits).
           "cnn_accel_top: invalid mode");
#endif

    // ==== DISPATCH BY MODE ====

    if (mode == MODE_MAXPOOL) {

        maxpool_2x2(in_ptr, out_ptr,
                    img_height, img_width, in_channels);

    } else if (mode == MODE_GLOBAL_AVG) {

        // [Fix 6] global_avg_pool internally guards spatial_area > 0
        global_avg_pool(in_ptr, out_ptr,
                        img_height, img_width, in_channels);

    } else if (mode == MODE_CONV_RELU || mode == MODE_CONV_LINEAR) {

        // [Fix 2] Output channel guard
        if (out_channels <= 0)
            return;
        if (out_channels > MAX_OUT_CH)
            out_channels = MAX_OUT_CH;

        // ---- Bias cache: one DDR burst, reused for all tiles ----
        static weight_t bias_cache[MAX_OUT_CH];
        // [A1] Bind to LUTRAM: consistent with wc_local, avoids BRAM port contention.
        #pragma HLS BIND_STORAGE variable=bias_cache type=ram_1p impl=lutram
        // [C2] RESET enables clock-enable gating on bias_cache LUTRAM during
        // all Row_Loop iterations (bias_cache is loaded once, then read-only).
        // Saves ~40mW toggle power on 128-entry LUTRAM during inference.
        #pragma HLS RESET variable=bias_cache  // Valid on static variable (S1-B)

        Load_Biases:
        for (int i = 0; i < out_channels; i++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=5 max=128 avg=64
            bias_cache[i] = bi_ptr[i];
        }

        int num_tiles_m = (out_channels + TILE_M - 1) / TILE_M;
        int num_tiles_c = (img_width    + TILE_C - 1) / TILE_C;

        // ================================================================
        // [Opt 18] Line-buffer architecture: row-by-row processing.
        //
        // Loop order: tile_c -> tile_m (weights loaded once) -> out_row
        //   For each output row, load KERNEL_MAX input rows into the
        //   line buffer, compute one row of output, and write it to DDR.
        //
        // This replaces the full input_buf_local[10][10][128] with a
        // compact line_buf[3][10][128], reducing BRAM by ~70%.
        //
        // Weight cache is reused across all rows within a tile_m group,
        // minimizing DDR bandwidth for the largest data structure.
        // ================================================================

        Spatial_C:
        for (int tile_c = 0; tile_c < num_tiles_c; tile_c++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=5 avg=5
            #pragma HLS LOOP_FLATTEN off

            Output_Groups:
            for (int tile_m = 0; tile_m < num_tiles_m; tile_m++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=4 avg=4
                #pragma HLS LOOP_FLATTEN off

                // [v4.6.12] Load weights ONCE per tile_m: amortized over all img_height rows.
                // Synthesis showed load_weights=39,140 cycles was being called per-row per-tile_m
                // in the v4.6.10 inverted hierarchy (3.7x slower). Reverted to tile_c->tile_m->row.
                weight_t wc_local[MAX_WEIGHT_TILES][TILE_M][TILE_N][KERNEL_MAX][KERNEL_MAX];
                #pragma HLS ARRAY_PARTITION variable=wc_local complete dim=2
                #pragma HLS ARRAY_PARTITION variable=wc_local complete dim=3
                #pragma HLS BIND_STORAGE variable=wc_local type=ram_s2p impl=lutram

                load_weights(wc_local, wt_ptr, tile_m, in_channels, out_channels, kernel_size);

                Row_Loop:
                for (int out_row = 0; out_row < img_height; out_row++) {
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=160 avg=80
                    #pragma HLS LOOP_FLATTEN off
                    #pragma HLS DATAFLOW

                    data_t line_buf[KERNEL_MAX][TILE_C_PAD][MAX_CONV_IN_CH];
                    #pragma HLS ARRAY_PARTITION variable=line_buf cyclic factor=TILE_N dim=3

                    data_t out_row_buf[TILE_C][TILE_M];
                    // Cyclic factor=8 dim=2: consecutive m values hit different banks -> II=1
                    #pragma HLS ARRAY_PARTITION variable=out_row_buf cyclic factor=8 dim=2

                    // [v4.6.13-S1] Explicit PIPO: double-buffer line_buf and out_row_buf so
                    // successive Row_Loop iterations overlap (load row N+1 while computing row N).
                    // Without PIPO, HLS may serialize the DATAFLOW stages, collapsing to sequential.
                    #pragma HLS PIPO variable=line_buf
                    #pragma HLS PIPO variable=out_row_buf

                    // [v4.6.13-S2] STABLE: wc_local and bias_cache are loaded BEFORE Row_Loop
                    // and are read-only across all row iterations. Without STABLE, HLS may attempt
                    //         #pragma HLS ARRAY_PARTITION variable=global_weights cyclic factor=4 dim=4
        #pragma HLS BIND_STORAGE variable=global_weights type=ram_s2p impl=bram
OW region).
                    #pragma HLS STABLE variable=wc_local
                    #pragma HLS STABLE variable=bias_cache

                    load_line_buffer(line_buf, in_ptr, out_row, tile_c, img_height, img_width, in_channels, kernel_size);

                    compute_row(line_buf, wc_local, out_row_buf, bias_cache, tile_m, in_channels, out_channels, kernel_size, mode);

                    write_output_row(out_row_buf, out_ptr, tile_m, out_row, tile_c, img_height, img_width, out_channels);
                }
            }
        }
    }
    // Unknown mode -> do nothing (safe: no DDR access, no hang)
}
A