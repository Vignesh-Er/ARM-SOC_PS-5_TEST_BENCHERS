// ============================================================================
// CNN Accelerator — Production Implementation (v4.6)
// Target: Xilinx Kria K26 SOM (xck26-sfvc784-2LV-c)
//
// Modes: Conv(3×3/1×1)+ReLU, Conv+Linear, MaxPool, GlobalAvgPool
//
// v4.1 hardware safety fixes over v4:
//   Fix 1: Guard in_channels  ≤ MAX_CONV_IN_CH  (prevents BRAM overrun)
//   Fix 2: Guard out_channels ≤ MAX_OUT_CH       (prevents bias/psum overrun)
//   Fix 3: Clamp kernel_size  ≤ KERNEL_MAX       (prevents buffer overrun)
//   Fix 4: Guard img_height/width ≤ MAX_SPATIAL   (prevents tile overflow)
//   Fix 5: global_avg_pool sum[] partitioned      (fixes pipeline II hazard)
//   Fix 6: Division-by-zero guard in global_avg   (prevents hardware hang)
//   Fix 7: AXI weight depth = MAX_WEIGHT_DEPTH    (computed, not literal)
//   Fix 8: Runtime 64-byte pointer alignment check (abort if misaligned)
//          [v4.4] Now guarded by __SYNTHESIS__: new/malloc not 64B-aligned
//   Fix 9: AXI weight depth safety-margined        (1M, decoupled from tensor)
//
// v4.3 performance improvements:
//   Imp 10: MAC parallelism doubled: TILE_N 4→8 (128 MACs, 10% DSP)
//   Imp 11: 3-stage DATAFLOW: load_weights ‖ compute_tile ‖ write_output
//   Imp 12: GAP: cyclic factor 4→16, 4-way parallel accumulation
//   Imp 13: AXI outstanding raised: reads 2→8, writes 2→4
//
// v4.4 functional correctness fixes:
//   Fix 10: Alignment check gated by __SYNTHESIS__ (C-sim uses heap alloc)
//   Fix 11: Saturation clamp before accum_t→data_t narrowing cast
//           (prevents AP_WRAP sign-flip on overflow: accum<32,20>→data<16,6>)
//   Fix 12: load_weights PIPELINE moved to innermost loop (Load_KX)
//           (avoids HLS loop-flatten failure on variable kernel_size bounds)
//   Fix 14: Removed max_widen_bitwidth=512 from all m_axi ports
//           (HLS 214-319: illegal aggregate on ap_fixed with conversion ops)
//
// v4.5 architectural optimizations:
//   Opt 14: accum_t reduced <48,24>→<32,20> (shorter DSP cascade path)
//   Opt 15: BIND_STORAGE ram_2p removed (let HLS auto-infer BRAM/LUTRAM)
//   Opt 16: BIND_OP impl=dsp forced on MAC psum (DSP48E2 utilization ~45%)
//   Opt 17: DATAFLOW removed from Output_Groups (sequential execution,
//           eliminates scheduler collapse on variable-bound tile_m loop)
//   Opt 18: Full input tile buffer replaced with line buffer architecture
//           input_buf_local[10][10][128] → line_buf[3][10][128] (70% smaller)
//           compute_tile → compute_row (psum [M][R][C] → [M][C], 87% smaller)
//           out_tile[R][C][M] → out_row_buf[C][M] (87% smaller)
//           Loop: tile_c → tile_m (weight reuse) → out_row (line buffer)
//
// v4.6 pipeline & burst optimizations:
//   Opt 19: DATAFLOW in Row_Loop (load ‖ compute ‖ write via PIPO buffers)
//   Opt 20: wc_local → LUTRAM (BIND_STORAGE impl=lutram, frees ~128 BRAMs)
//   Opt 21: load_line_buffer burst-safe (pre-zero + unconditional DDR reads)
//   Opt 22: load_weights burst-safe (pre-zero + valid-range-only reads)
//   Opt 23: global_avg_pool simplified (removed UNROLL stepping → ch++ II=1)
//   Opt 24: maxpool_2x2 accepts II=4 (bandwidth-limited, 2.4% of total)
//   Opt 25: STABLE pragma on wc_local/bias_cache for DATAFLOW correctness
//
// Weight format: NCHW [Cout][Cin][Ky][Kx]
// Data format:   HWC  [H][W][C]
//
// IMPORTANT: weights passed to cnn_accel_top MUST use actual (unpadded)
// channel strides matching in_channels.  The kernel internally zero-fills
// non-tile-aligned channels via guards in load_weights and compute_tile.
// ============================================================================

#include "cnn_accelerator.h"

// ============================================================================
// PHASE 1: Load Line Buffer — KERNEL_MAX rows for one output row
//
// [Opt 18] Replaces full spatial tile buffer with a line buffer.
// For a 3×3 convolution producing output row `out_row`, we need input rows
// [out_row-1, out_row, out_row+1] (with zero-padding at boundaries).
// Buffer: line_buf[KERNEL_MAX][TILE_C_PAD][MAX_CONV_IN_CH]
//   = 3 × 10 × 128 = 3,840 elements  (was 10×10×128 = 12,800)
//   → ~70% BRAM reduction on input buffer alone.
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

    int padding = (kernel_size > 1) ? (kernel_size / 2) : 0;
    int start_col = tile_c * TILE_C - padding;
    int buf_cols  = TILE_C + 2 * padding;

    // [Opt 21] Phase 1: Zero the entire line buffer (unconditional, burst-safe)
    LB_Zero_Row:
    for (int kr = 0; kr < kernel_size; kr++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=3 avg=3
        LB_Zero_Col:
        for (int c = 0; c < buf_cols; c++) {
            #pragma HLS LOOP_TRIPCOUNT min=8 max=10 avg=10
            LB_Zero_Ch:
            for (int ch = 0; ch < in_channels; ch++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=3 max=128 avg=32
                line_buf[kr][c][ch] = data_t(0);
            }
        }
    }

    // [Opt 21] Phase 2: Load valid pixels — unconditional DDR reads (burst-friendly)
    // Conditionals moved to outer loops; innermost LB_Load_Ch is branch-free.
    LB_Load_Row:
    for (int kr = 0; kr < kernel_size; kr++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=3 avg=3
        int gr = out_row - padding + kr;
        if (gr < 0 || gr >= img_height) continue;

        LB_Load_Col:
        for (int c = 0; c < buf_cols; c++) {
            #pragma HLS LOOP_TRIPCOUNT min=8 max=10 avg=10
            int gc = start_col + c;
            if (gc < 0 || gc >= img_width) continue;

            int base_idx = (gr * img_width + gc) * in_channels;

            LB_Load_Ch:
            for (int ch = 0; ch < in_channels; ch++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=3 max=128 avg=32
                line_buf[kr][c][ch] = input_dram[base_idx + ch];
            }
        }
    }
}

// ============================================================================
// PHASE 2a: Load Weights for ONE output channel group
//
// Weights loaded once per tile_m group and reused across all output rows.
// NCHW: for fixed Cout, [Cin][Ky][Kx] is contiguous → stride-1 DDR.
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

    // [Opt 22] Phase 1: Zero the entire weight cache
    // m and n dimensions are fully partitioned → all written in parallel per cycle.
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

    // [Opt 22] Phase 2: Load valid weights — unconditional DDR reads
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
                        int idx = ((out_ch * in_channels + in_ch)
                                  * kernel_size + ky) * kernel_size + kx;
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
// Processes one output row (TILE_C columns × TILE_M output channels).
// psum reduced from psum[M][R][C] → psum[M][C]  (87.5% smaller).
// out_row_buf reduced from out_tile[R][C][M] → out_row_buf[C][M].
//
// 128 parallel MACs per cycle (TILE_M=16 × TILE_N=8).
//   psum += (accum_t)(in_pix * w)  ← 16×8 → 1 DSP48E2 each
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
    #pragma HLS ARRAY_PARTITION variable=out_row_buf complete dim=2

    int num_tiles_n = (in_channels + TILE_N - 1) / TILE_N;

    // Partial sum accumulator — 2D: [M][C] (was [M][R][C])
    accum_t psum[TILE_M][TILE_C];
    #pragma HLS ARRAY_PARTITION variable=psum complete dim=1
    #pragma HLS BIND_OP variable=psum op=mul impl=dsp

    // Zero psum: TILE_C cycles (m unrolled via partition)
    Zero_Psum:
    for (int c = 0; c < TILE_C; c++) {
        #pragma HLS PIPELINE II=1
        for (int m = 0; m < TILE_M; m++) {
            #pragma HLS UNROLL
            psum[m][c] = 0;
        }
    }

    // Main MAC loop — iterate over input channel tiles × kernel window
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
                    #pragma HLS PIPELINE II=1

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
                        MAC_N:
                        for (int n = 0; n < TILE_N; n++) {
                            #pragma HLS UNROLL
                            weight_t w = weight_cache[tile_n][m][n][ky][kx];
                            psum[m][c] += (accum_t)(in_pix[n] * w);
                        }
                    }
                }
            }
        }
    }

    // ---- Bias + Activation → store to out_row_buf ----
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
                val = 0;

            // Saturate accumulator to data_t representable range
            // before narrowing cast (prevents AP_WRAP sign-flip).
            // data_t = ap_fixed<16,6>: range [-32, +31.999]
            if (val > accum_t(31)) val = accum_t(31);
            if (val < accum_t(-32)) val = accum_t(-32);

            data_t result;
            if (mode == MODE_CONV_RELU)
                result = (val > accum_t(0)) ? (data_t)val : data_t(0);
            else
                result = (data_t)val;

            out_row_buf[c][m] = result;
        }
    }
}

// ============================================================================
// PHASE 2c: Write ONE Output Row to DDR
//
// [Opt 18] Row-based write replaces full tile write.
// Writes out_row_buf[TILE_C][TILE_M] for a single output row.
// c→m order (m innermost = stride-1 in HWC → burst writes).
// ============================================================================

static void write_output_row(
    const data_t out_row_buf[TILE_C][TILE_M],
    data_t* output_dram,
    int tile_m, int out_row, int tile_c,
    int img_height, int img_width,
    int out_channels)
{
    #pragma HLS INLINE OFF
    #pragma HLS ARRAY_PARTITION variable=out_row_buf complete dim=2

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
        #pragma HLS LOOP_TRIPCOUNT min=1 max=8 avg=8
        Write_M:
        for (int m = 0; m < valid_m; m++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16 avg=16

            int out_ch = tile_m * TILE_M + m;
            int out_idx = (out_row * img_width + (start_col + c))
                        * out_channels + out_ch;
            output_dram[out_idx] = out_row_buf[c][m];
        }
    }
}

// ============================================================================
// MAXPOOL 2×2 STRIDE 2
// ============================================================================

static void maxpool_2x2(const data_t* input_dram, data_t* output_dram,
                        int img_height, int img_width, int channels)
{
    #pragma HLS INLINE OFF

    int out_h = img_height / 2;
    int out_w = img_width  / 2;

    Pool_Row:
    for (int oh = 0; oh < out_h; oh++) {
        #pragma HLS LOOP_TRIPCOUNT min=10 max=80 avg=40
        Pool_Col:
        for (int ow = 0; ow < out_w; ow++) {
            #pragma HLS LOOP_TRIPCOUNT min=10 max=80 avg=40
            Pool_Ch:
            for (int c = 0; c < channels; c++) {
                #pragma HLS PIPELINE II=1
                #pragma HLS LOOP_TRIPCOUNT min=16 max=128 avg=32

                int ih = oh * 2;
                int iw = ow * 2;

                data_t v00 = input_dram[(ih     * img_width + iw    ) * channels + c];
                data_t v01 = input_dram[(ih     * img_width + iw + 1) * channels + c];
                data_t v10 = input_dram[((ih+1) * img_width + iw    ) * channels + c];
                data_t v11 = input_dram[((ih+1) * img_width + iw + 1) * channels + c];

                data_t m0 = (v00 > v01) ? v00 : v01;
                data_t m1 = (v10 > v11) ? v10 : v11;
                output_dram[(oh * out_w + ow) * channels + c] = (m0 > m1) ? m0 : m1;
            }
        }
    }
}

// ============================================================================
// GLOBAL AVERAGE POOLING — H×W×C → 1×1×C
//
// [Opt 23] Sequential channel accumulation with PIPELINE II=1.
//   sum[] partitioned cyclic factor=16 → 16 BRAM banks.
//   Inner loop steps ch++ (1 channel/cycle, 1 DDR read/cycle).
//   Cyclic factor=16 ensures bank reuse interval=16 >> RAW latency → II=1.
//
// [Fix 6] Division-by-zero guard: if spatial area is 0, output zeros.
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

    // [Fix 6] Guard: if either spatial dimension is 0, skip accumulation
    int spatial_area = img_height * img_width;

    if (spatial_area > 0) {
        Acc_Row:
        for (int r = 0; r < img_height; r++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=20 avg=20
            Acc_Col:
            for (int c = 0; c < img_width; c++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=20 avg=20
                Acc_Ch:
                for (int ch = 0; ch < channels; ch++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=16 max=128 avg=64

                    int idx = (r * img_width + c) * channels + ch;
                    sum[ch] += (accum_t)input_dram[idx];
                }
            }
        }

        accum_t divisor = (accum_t)spatial_area;

        Write_Avg:
        for (int ch = 0; ch < channels; ch++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=16 max=128 avg=64
            output_dram[ch] = (data_t)(sum[ch] / divisor);
        }
    } else {
        // Spatial area is 0 → output zeros (safe fallback)
        Write_Zero:
        for (int ch = 0; ch < channels; ch++) {
            #pragma HLS PIPELINE II=1
            #pragma HLS LOOP_TRIPCOUNT min=16 max=128 avg=64
            output_dram[ch] = data_t(0);
        }
    }
}

// ============================================================================
// TOP-LEVEL ENTRY POINT
//
// [Fix 1-4] Hard guards at entry:
//   - in_channels  clamped to MAX_CONV_IN_CH  → prevents input_buf overrun
//   - out_channels clamped to MAX_OUT_CH       → prevents bias/psum overrun
//   - kernel_size  clamped to KERNEL_MAX       → prevents buffer overrun
//   - img_height/width clamped to MAX_SPATIAL  → prevents tile overflow
//
// [Fix 7] AXI weight depth = MAX_WEIGHT_DEPTH (computed from header defines)
//
// 3-stage DATAFLOW per row: load_line_buffer ‖ compute_row ‖ write_output_row
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
    // [Imp 13] Outstanding raised: reads 8, writes 4 (hide DDR latency)
    // NOTE: max_widen_bitwidth removed — illegal on ap_fixed pointer ports
    //       (HLS 214-319: aggregate on type with conversion operators).
    //       HLS will auto-widen to the port's natural element width (16 bits).
    #pragma HLS INTERFACE m_axi port=input_data  offset=slave bundle=gmem_in \
        depth=3276800 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS INTERFACE m_axi port=output_data offset=slave bundle=gmem_out \
        depth=3276800 max_write_burst_length=64 num_write_outstanding=4
    #pragma HLS INTERFACE m_axi port=weights     offset=slave bundle=gmem_param \
        depth=1048576 max_read_burst_length=64 num_read_outstanding=8
    #pragma HLS INTERFACE m_axi port=biases      offset=slave bundle=gmem_bias \
        depth=128 max_read_burst_length=32 num_read_outstanding=4

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
    // C-sim: ports ARE data_t*/weight_t* already — direct assignment.
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
            return;  // Misaligned — abort, PS sees ap_done immediately
        }
    }
#endif

    // ================================================================
    // [Fix 1-4] Hard dimension guards — prevent all buffer overruns
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

    // [Fix 3] Kernel size clamp (accepts 1 or 3 only; default to 3)
    if (kernel_size != 1 && kernel_size != 3)
        kernel_size = 3;
    if (kernel_size > KERNEL_MAX)
        kernel_size = KERNEL_MAX;

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
        weight_t bias_cache[MAX_OUT_CH];

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
        // Loop order: tile_c → tile_m (weights loaded once) → out_row
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
            #pragma HLS LOOP_TRIPCOUNT min=1 max=20 avg=5

            // ---- Output channel groups: load weights once per group ----
            Output_Groups:
            for (int tile_m = 0; tile_m < num_tiles_m; tile_m++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=8 avg=4

                weight_t wc_local[MAX_WEIGHT_TILES][TILE_M][TILE_N][KERNEL_MAX][KERNEL_MAX];
                #pragma HLS ARRAY_PARTITION variable=wc_local complete dim=2
                #pragma HLS ARRAY_PARTITION variable=wc_local complete dim=3
                #pragma HLS BIND_STORAGE variable=wc_local type=ram_s2p impl=lutram

                load_weights(wc_local, wt_ptr, tile_m,
                            in_channels, out_channels, kernel_size);

                // ---- Row-by-row processing with line buffer + DATAFLOW ----
                // [Opt 19] DATAFLOW enables pipelined overlap:
                //   load_line_buffer(N) || compute_row(N-1) || write_output_row(N-2)
                //   line_buf and out_row_buf become PIPO buffers automatically.
                //   wc_local and bias_cache are read-only → marked STABLE.
                Row_Loop:
                for (int out_row = 0; out_row < img_height; out_row++) {
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=160 avg=40
                    #pragma HLS DATAFLOW
                    #pragma HLS STABLE variable=wc_local
                    #pragma HLS STABLE variable=bias_cache

                    data_t line_buf[KERNEL_MAX][TILE_C_PAD][MAX_CONV_IN_CH];
                    #pragma HLS ARRAY_PARTITION variable=line_buf cyclic factor=TILE_N dim=3

                    data_t out_row_buf[TILE_C][TILE_M];
                    #pragma HLS ARRAY_PARTITION variable=out_row_buf complete dim=2

                    load_line_buffer(line_buf, in_ptr,
                                    out_row, tile_c,
                                    img_height, img_width,
                                    in_channels, kernel_size);

                    compute_row(line_buf, wc_local,
                               out_row_buf, bias_cache,
                               tile_m,
                               in_channels, out_channels,
                               kernel_size, mode);

                    write_output_row(out_row_buf, out_ptr,
                                   tile_m, out_row, tile_c,
                                   img_height, img_width,
                                   out_channels);
                }
            }
        }
    }
    // Unknown mode → do nothing (safe: no DDR access, no hang)
}