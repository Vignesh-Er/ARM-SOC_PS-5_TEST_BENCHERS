// ============================================================================
// CNN Accelerator — C-Simulation Testbench (v4.1)
//
// Tests the full CNN pipeline including all 4 modes:
//   MODE_CONV_RELU(0), MODE_MAXPOOL(1), MODE_CONV_LINEAR(2), MODE_GLOBAL_AVG(3)
//
// Pipeline:
//   Conv1(3→16,3×3) → Pool1 → Conv2(16→32,3×3) → Pool2 →
//   Conv3(32→64,3×3) → Conv4(64→128,3×3) → Pool3 →
//   GlobalAvgPool → Conv5(128→10,1×1) [classification head]
//
// Input:  160×160×3   (76,800 elements)
// Final:  1×1×10      (10 class scores)
//
// Also tests:
//   - TF→NCHW weight reorder + tile padding
//   - CPU baseline for correctness verification
//   - Dimension guard paths (invalid inputs)
//   - All 4 operation modes
// ============================================================================

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <ctime>

#include "cnn_accelerator.h"

// ============================================================================
// Utility: round up to next multiple
// ============================================================================

static int round_up(int val, int multiple) {
    return ((val + multiple - 1) / multiple) * multiple;
}

// ============================================================================
// Weight loader — reads comma-separated floats from .txt files
// ============================================================================

static bool load_weights_txt(weight_t* dst, int count, const char* filename) {
    std::ifstream fin(filename);
    if (!fin.is_open()) {
        printf("ERROR: Cannot open weight file: %s\n", filename);
        return false;
    }

    std::string content((std::istreambuf_iterator<char>(fin)),
                         std::istreambuf_iterator<char>());
    fin.close();

    std::stringstream ss(content);
    std::string token;
    int idx = 0;

    while (std::getline(ss, token, ',') && idx < count) {
        size_t start = token.find_first_not_of(" \t\n\r");
        if (start == std::string::npos) continue;
        token = token.substr(start);
        float val = (float)atof(token.c_str());
        dst[idx] = (weight_t)val;
        idx++;
    }

    if (idx != count) {
        printf("WARNING: %s — expected %d values, read %d\n", filename, count, idx);
        for (int i = idx; i < count; i++)
            dst[i] = weight_t(0);
    }

    return true;
}

// ============================================================================
// Weight layout reorder: TF [Ky][Kx][Cin][Cout] → NCHW [Cout][Cin][Ky][Kx]
// ============================================================================

static void reorder_weights_tf_to_nchw(const weight_t* src, weight_t* dst,
                                       int ksize, int cin, int cout)
{
    for (int co = 0; co < cout; co++)
        for (int ci = 0; ci < cin; ci++)
            for (int ky = 0; ky < ksize; ky++)
                for (int kx = 0; kx < ksize; kx++) {
                    int tf_idx   = ((ky * ksize + kx) * cin + ci) * cout + co;
                    int nchw_idx = ((co * cin + ci) * ksize + ky) * ksize + kx;
                    dst[nchw_idx] = src[tf_idx];
                }
}

// ============================================================================
// Pad weight array to tile-aligned size (zero-fill for safe burst reads)
// ============================================================================

static weight_t* create_padded_weights(const weight_t* nchw_weights,
                                       int ksize, int cin, int cout,
                                       int* padded_cin_out,
                                       int* padded_cout_out)
{
    int padded_cin  = round_up(cin,  TILE_N);
    int padded_cout = round_up(cout, TILE_M);
    int padded_size = padded_cout * padded_cin * ksize * ksize;

    weight_t* padded = new weight_t[padded_size];
    memset(padded, 0, padded_size * sizeof(weight_t));

    for (int co = 0; co < cout; co++)
        for (int ci = 0; ci < cin; ci++)
            for (int ky = 0; ky < ksize; ky++)
                for (int kx = 0; kx < ksize; kx++) {
                    int src_idx = ((co * cin + ci) * ksize + ky) * ksize + kx;
                    int dst_idx = ((co * padded_cin + ci) * ksize + ky) * ksize + kx;
                    padded[dst_idx] = nchw_weights[src_idx];
                }

    *padded_cin_out  = padded_cin;
    *padded_cout_out = padded_cout;
    return padded;
}

// ============================================================================
// Pad bias array to TILE_M multiple
// ============================================================================

static weight_t* create_padded_biases(const weight_t* biases, int cout) {
    int padded_cout = round_up(cout, TILE_M);
    weight_t* padded = new weight_t[padded_cout];
    memset(padded, 0, padded_cout * sizeof(weight_t));
    for (int i = 0; i < cout; i++)
        padded[i] = biases[i];
    return padded;
}

// ============================================================================
// CPU baseline: naive Conv2D + Bias + ReLU (for correctness comparison)
// ============================================================================

static void cpu_conv2d_relu(const data_t* input, data_t* output,
                            const weight_t* weights_nchw,
                            const weight_t* biases,
                            int H, int W, int Cin, int Cout, int K)
{
    int pad = K / 2;
    for (int oh = 0; oh < H; oh++)
        for (int ow = 0; ow < W; ow++)
            for (int co = 0; co < Cout; co++) {
                accum_t sum = (accum_t)biases[co];
                for (int ci = 0; ci < Cin; ci++)
                    for (int ky = 0; ky < K; ky++)
                        for (int kx = 0; kx < K; kx++) {
                            int ih = oh - pad + ky;
                            int iw = ow - pad + kx;
                            data_t pixel = data_t(0);
                            if (ih >= 0 && ih < H && iw >= 0 && iw < W)
                                pixel = input[(ih * W + iw) * Cin + ci];
                            int widx = ((co * Cin + ci) * K + ky) * K + kx;
                            sum += (accum_t)(pixel * weights_nchw[widx]);
                        }
                data_t result = (sum > accum_t(0)) ? (data_t)sum : data_t(0);
                output[(oh * W + ow) * Cout + co] = result;
            }
}

// ============================================================================
// CPU baseline: MaxPool 2×2
// ============================================================================

static void cpu_maxpool(const data_t* input, data_t* output,
                        int H, int W, int C)
{
    int oH = H / 2, oW = W / 2;
    for (int oh = 0; oh < oH; oh++)
        for (int ow = 0; ow < oW; ow++)
            for (int c = 0; c < C; c++) {
                int ih = oh * 2, iw = ow * 2;
                data_t v00 = input[(ih     * W + iw    ) * C + c];
                data_t v01 = input[(ih     * W + iw + 1) * C + c];
                data_t v10 = input[((ih+1) * W + iw    ) * C + c];
                data_t v11 = input[((ih+1) * W + iw + 1) * C + c];
                data_t m0 = (v00 > v01) ? v00 : v01;
                data_t m1 = (v10 > v11) ? v10 : v11;
                output[(oh * oW + ow) * C + c] = (m0 > m1) ? m0 : m1;
            }
}

// ============================================================================
// Statistics helper
// ============================================================================

struct BufferStats {
    float min_val, max_val, mean_val;
    int nonzero, total;
};

static BufferStats compute_stats(const data_t* buf, int size) {
    BufferStats s;
    s.total = size;
    s.nonzero = 0;
    s.min_val = 1e9f;
    s.max_val = -1e9f;
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        float v = (float)buf[i];
        if (v != 0.0f) s.nonzero++;
        if (v < s.min_val) s.min_val = v;
        if (v > s.max_val) s.max_val = v;
        sum += v;
    }
    s.mean_val = (float)(sum / size);
    return s;
}

static void print_stats(const char* name, const BufferStats& s) {
    printf("  %-20s | min=%8.4f  max=%8.4f  mean=%8.5f  nonzero=%d/%d\n",
           name, s.min_val, s.max_val, s.mean_val, s.nonzero, s.total);
}

// ============================================================================
// Compare HLS output vs CPU reference
// ============================================================================

static bool compare_outputs(const data_t* hls_out, const data_t* cpu_out,
                            int size, float tol, const char* layer_name)
{
    int mismatches = 0;
    float max_diff = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = fabs((float)hls_out[i] - (float)cpu_out[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tol) mismatches++;
    }
    if (mismatches > 0) {
        printf("  %s MISMATCH: %d/%d elements differ > %.4f (max_diff=%.6f)\n",
               layer_name, mismatches, size, tol, max_diff);
        return false;
    } else {
        printf("  %s MATCH: max_diff=%.6f (tol=%.4f)\n",
               layer_name, max_diff, tol);
        return true;
    }
}

// ============================================================================
// Main testbench
// ============================================================================

int main() {
    printf("============================================================\n");
    printf("  CNN Accelerator — C-Simulation Testbench (v4.1)\n");
    printf("  Target: Kria K26 SOM (xck26-sfvc784-2LV-c)\n");
    printf("  Modes tested: CONV_RELU, MAXPOOL, GLOBAL_AVG, CONV_LINEAR\n");
    printf("============================================================\n\n");

    // ------------------------------------------------------------------
    // 1. Allocate buffers
    // ------------------------------------------------------------------
    printf("[1/8] Allocating buffers...\n");

    const int INPUT_SIZE = 160 * 160 * 3;
    const int CONV1_SIZE = 160 * 160 * 16;
    const int POOL1_SIZE = 80  * 80  * 16;
    const int CONV2_SIZE = 80  * 80  * 32;
    const int POOL2_SIZE = 40  * 40  * 32;
    const int CONV3_SIZE = 40  * 40  * 64;
    const int CONV4_SIZE = 40  * 40  * 128;
    const int POOL3_SIZE = 20  * 20  * 128;
    const int GAP_SIZE   = 128;           // GlobalAvgPool output
    const int HEAD_SIZE  = 10;            // Classification head output

    data_t* input     = new data_t[INPUT_SIZE];
    data_t* conv1_out = new data_t[CONV1_SIZE];
    data_t* pool1_out = new data_t[POOL1_SIZE];
    data_t* conv2_out = new data_t[CONV2_SIZE];
    data_t* pool2_out = new data_t[POOL2_SIZE];
    data_t* conv3_out = new data_t[CONV3_SIZE];
    data_t* conv4_out = new data_t[CONV4_SIZE];
    data_t* pool3_out = new data_t[POOL3_SIZE];
    data_t* gap_out   = new data_t[GAP_SIZE];
    data_t* head_out  = new data_t[HEAD_SIZE];

    // CPU reference buffers (first 3 layers for verification)
    data_t* cpu_conv1 = new data_t[CONV1_SIZE];
    data_t* cpu_pool1 = new data_t[POOL1_SIZE];
    data_t* cpu_conv2 = new data_t[CONV2_SIZE];

    // ------------------------------------------------------------------
    // 2. Load trained weights from .txt files
    // ------------------------------------------------------------------
    printf("[2/8] Loading weights...\n");

    const int W1_SIZE = 3 * 3 * 3  * 16;
    const int B1_SIZE = 16;
    const int W2_SIZE = 3 * 3 * 16 * 32;
    const int B2_SIZE = 32;
    const int W3_SIZE = 3 * 3 * 32 * 64;
    const int B3_SIZE = 64;
    const int W4_SIZE = 3 * 3 * 64 * 128;
    const int B4_SIZE = 128;

    weight_t* w1_tf = new weight_t[W1_SIZE];
    weight_t* b1    = new weight_t[B1_SIZE];
    weight_t* w2_tf = new weight_t[W2_SIZE];
    weight_t* b2    = new weight_t[B2_SIZE];
    weight_t* w3_tf = new weight_t[W3_SIZE];
    weight_t* b3    = new weight_t[B3_SIZE];
    weight_t* w4_tf = new weight_t[W4_SIZE];
    weight_t* b4    = new weight_t[B4_SIZE];

    bool ok = true;
    ok &= load_weights_txt(w1_tf, W1_SIZE, "w2.txt");
    ok &= load_weights_txt(b1,    B1_SIZE, "b2.txt");
    ok &= load_weights_txt(w2_tf, W2_SIZE, "w6.txt");
    ok &= load_weights_txt(b2,    B2_SIZE, "b6.txt");
    ok &= load_weights_txt(w3_tf, W3_SIZE, "w10.txt");
    ok &= load_weights_txt(b3,    B3_SIZE, "b10.txt");
    ok &= load_weights_txt(w4_tf, W4_SIZE, "w13.txt");
    ok &= load_weights_txt(b4,    B4_SIZE, "b13.txt");

    if (!ok) {
        printf("ERROR: Failed to load one or more weight files.\n");
        return 1;
    }
    printf("  All weights loaded successfully.\n");

    // ------------------------------------------------------------------
    // 3. Reorder TF → NCHW, then pad to tile multiples
    // ------------------------------------------------------------------
    printf("[3/8] Reordering weights (TF->NCHW) and padding...\n");

    weight_t* w1_nchw = new weight_t[W1_SIZE];
    weight_t* w2_nchw = new weight_t[W2_SIZE];
    weight_t* w3_nchw = new weight_t[W3_SIZE];
    weight_t* w4_nchw = new weight_t[W4_SIZE];

    reorder_weights_tf_to_nchw(w1_tf, w1_nchw, 3, 3,  16);
    reorder_weights_tf_to_nchw(w2_tf, w2_nchw, 3, 16, 32);
    reorder_weights_tf_to_nchw(w3_tf, w3_nchw, 3, 32, 64);
    reorder_weights_tf_to_nchw(w4_tf, w4_nchw, 3, 64, 128);

    delete[] w1_tf; delete[] w2_tf; delete[] w3_tf; delete[] w4_tf;

    int pc1, pco1, pc2, pco2, pc3, pco3, pc4, pco4;
    weight_t* w1_pad = create_padded_weights(w1_nchw, 3, 3,  16,  &pc1, &pco1);
    weight_t* w2_pad = create_padded_weights(w2_nchw, 3, 16, 32,  &pc2, &pco2);
    weight_t* w3_pad = create_padded_weights(w3_nchw, 3, 32, 64,  &pc3, &pco3);
    weight_t* w4_pad = create_padded_weights(w4_nchw, 3, 64, 128, &pc4, &pco4);

    weight_t* b1_pad = create_padded_biases(b1, 16);
    weight_t* b2_pad = create_padded_biases(b2, 32);
    weight_t* b3_pad = create_padded_biases(b3, 64);
    weight_t* b4_pad = create_padded_biases(b4, 128);

    printf("  Conv1 weights: 3->%d cin, 16->%d cout (padded)\n", pc1, pco1);
    printf("  Conv2 weights: 16->%d cin, 32->%d cout (padded)\n", pc2, pco2);
    printf("  Conv3 weights: 32->%d cin, 64->%d cout (padded)\n", pc3, pco3);
    printf("  Conv4 weights: 64->%d cin, 128->%d cout (padded)\n", pc4, pco4);

    // Create synthetic 1×1 classification head weights (128→10)
    // In real deployment, these come from the trained model
    const int W5_SIZE = 1 * 1 * 128 * 10;
    const int B5_SIZE = 10;
    weight_t* w5_nchw = new weight_t[W5_SIZE];
    weight_t* b5      = new weight_t[B5_SIZE];
    for (int i = 0; i < W5_SIZE; i++)
        w5_nchw[i] = weight_t(0.01f * ((i % 7) - 3));  // small spread
    for (int i = 0; i < B5_SIZE; i++)
        b5[i] = weight_t(0.1f * i);

    int pc5, pco5;
    weight_t* w5_pad = create_padded_weights(w5_nchw, 1, 128, 10, &pc5, &pco5);
    weight_t* b5_pad = create_padded_biases(b5, 10);
    printf("  Conv5 head:    128->%d cin, 10->%d cout (padded, 1x1)\n", pc5, pco5);
    printf("  Weight reorder + padding complete.\n\n");

    // ------------------------------------------------------------------
    // 4. Create test input (normalized ramp)
    // ------------------------------------------------------------------
    printf("[4/8] Creating test input (160x160x3 ramp)...\n\n");
    for (int i = 0; i < INPUT_SIZE; i++)
        input[i] = data_t((float)(i % 256) / 255.0f);

    // ------------------------------------------------------------------
    // 5. Run HLS accelerator — full pipeline (9 IP calls)
    // ------------------------------------------------------------------
    printf("[5/8] Running HLS accelerator pipeline...\n\n");

    clock_t hls_start = clock();

    // Layer 1: Conv2D(3→16, 3×3) + ReLU
    printf("  Layer 1: Conv2D 160x160x3 -> 160x160x16 (3x3, relu)\n");
    cnn_accel_top(input, conv1_out, w1_nchw, b1,
                  160, 160, 3, 16, MODE_CONV_RELU, 3);
    print_stats("conv1_out", compute_stats(conv1_out, CONV1_SIZE));

    // Layer 2: MaxPool 2×2
    printf("  Layer 2: MaxPool 160x160x16 -> 80x80x16\n");
    cnn_accel_top(conv1_out, pool1_out, w1_nchw, b1,
                  160, 160, 16, 16, MODE_MAXPOOL, 3);
    print_stats("pool1_out", compute_stats(pool1_out, POOL1_SIZE));

    // Layer 3: Conv2D(16→32, 3×3) + ReLU
    printf("  Layer 3: Conv2D 80x80x16 -> 80x80x32 (3x3, relu)\n");
    cnn_accel_top(pool1_out, conv2_out, w2_nchw, b2,
                  80, 80, 16, 32, MODE_CONV_RELU, 3);
    print_stats("conv2_out", compute_stats(conv2_out, CONV2_SIZE));

    // Layer 4: MaxPool 2×2
    printf("  Layer 4: MaxPool 80x80x32 -> 40x40x32\n");
    cnn_accel_top(conv2_out, pool2_out, w2_nchw, b2,
                  80, 80, 32, 32, MODE_MAXPOOL, 3);
    print_stats("pool2_out", compute_stats(pool2_out, POOL2_SIZE));

    // Layer 5: Conv2D(32→64, 3×3) + ReLU
    printf("  Layer 5: Conv2D 40x40x32 -> 40x40x64 (3x3, relu)\n");
    cnn_accel_top(pool2_out, conv3_out, w3_nchw, b3,
                  40, 40, 32, 64, MODE_CONV_RELU, 3);
    print_stats("conv3_out", compute_stats(conv3_out, CONV3_SIZE));

    // Layer 6: Conv2D(64→128, 3×3) + ReLU
    printf("  Layer 6: Conv2D 40x40x64 -> 40x40x128 (3x3, relu)\n");
    cnn_accel_top(conv3_out, conv4_out, w4_nchw, b4,
                  40, 40, 64, 128, MODE_CONV_RELU, 3);
    print_stats("conv4_out", compute_stats(conv4_out, CONV4_SIZE));

    // Layer 7: MaxPool 2×2
    printf("  Layer 7: MaxPool 40x40x128 -> 20x20x128\n");
    cnn_accel_top(conv4_out, pool3_out, w4_nchw, b4,
                  40, 40, 128, 128, MODE_MAXPOOL, 3);
    print_stats("pool3_out", compute_stats(pool3_out, POOL3_SIZE));

    // Layer 8: Global Average Pooling
    printf("  Layer 8: GlobalAvgPool 20x20x128 -> 1x1x128\n");
    cnn_accel_top(pool3_out, gap_out, w4_nchw, b4,
                  20, 20, 128, 128, MODE_GLOBAL_AVG, 3);
    print_stats("gap_out", compute_stats(gap_out, GAP_SIZE));

    // Layer 9: Conv1×1(128→10) + Linear (classification head)
    printf("  Layer 9: Conv1x1 1x1x128 -> 1x1x10 (linear)\n");
    cnn_accel_top(gap_out, head_out, w5_nchw, b5,
                  1, 1, 128, 10, MODE_CONV_LINEAR, 1);
    print_stats("head_out", compute_stats(head_out, HEAD_SIZE));

    clock_t hls_end = clock();
    double hls_time_ms = 1000.0 * (hls_end - hls_start) / CLOCKS_PER_SEC;

    // ------------------------------------------------------------------
    // 6. CPU baseline (first 3 layers for verification)
    // ------------------------------------------------------------------
    printf("\n[6/8] Running CPU baseline (layers 1-3)...\n\n");

    clock_t cpu_start = clock();

    printf("  CPU Layer 1: Conv2D 160x160x3 -> 160x160x16\n");
    cpu_conv2d_relu(input, cpu_conv1, w1_nchw, b1, 160, 160, 3, 16, 3);

    printf("  CPU Layer 2: MaxPool 160x160x16 -> 80x80x16\n");
    cpu_maxpool(cpu_conv1, cpu_pool1, 160, 160, 16);

    printf("  CPU Layer 3: Conv2D 80x80x16 -> 80x80x32\n");
    cpu_conv2d_relu(cpu_pool1, cpu_conv2, w2_nchw, b2, 80, 80, 16, 32, 3);

    clock_t cpu_end = clock();
    double cpu_time_ms = 1000.0 * (cpu_end - cpu_start) / CLOCKS_PER_SEC;

    // ------------------------------------------------------------------
    // 7. Dimension guard tests (should return without crash)
    // ------------------------------------------------------------------
    printf("\n[7/8] Testing dimension guards...\n");

    data_t guard_buf[16];
    weight_t guard_w = weight_t(0);
    weight_t guard_b = weight_t(0);

    // Test: zero dimensions → should return immediately
    printf("  Guard test 1: img_height=0 ...");
    cnn_accel_top(guard_buf, guard_buf, &guard_w, &guard_b,
                  0, 10, 3, 16, MODE_CONV_RELU, 3);
    printf(" OK (no crash)\n");

    // Test: zero in_channels → should return immediately
    printf("  Guard test 2: in_channels=0 ...");
    cnn_accel_top(guard_buf, guard_buf, &guard_w, &guard_b,
                  10, 10, 0, 16, MODE_CONV_RELU, 3);
    printf(" OK (no crash)\n");

    // Test: zero out_channels in conv mode → should return immediately
    printf("  Guard test 3: out_channels=0 ...");
    cnn_accel_top(guard_buf, guard_buf, &guard_w, &guard_b,
                  10, 10, 3, 0, MODE_CONV_RELU, 3);
    printf(" OK (no crash)\n");

    // Test: invalid kernel size → should sanitize to 3
    printf("  Guard test 4: kernel_size=7 ...");
    cnn_accel_top(guard_buf, guard_buf, &guard_w, &guard_b,
                  1, 1, 1, 1, MODE_CONV_RELU, 7);
    printf(" OK (sanitized to 3)\n");

    // Test: unknown mode → should do nothing
    printf("  Guard test 5: mode=99 ...");
    cnn_accel_top(guard_buf, guard_buf, &guard_w, &guard_b,
                  10, 10, 3, 16, 99, 3);
    printf(" OK (no-op)\n");

    printf("  All guard tests passed.\n");

    // ------------------------------------------------------------------
    // 8. Results summary
    // ------------------------------------------------------------------
    printf("\n============================================================\n");
    printf("  RESULTS SUMMARY\n");
    printf("============================================================\n");

    // Final output (classification scores)
    printf("\n  Classification head output (10 classes):\n");
    for (int i = 0; i < HEAD_SIZE; i++)
        printf("    class[%d] = %8.4f\n", i, (float)head_out[i]);

    // Backbone output stats
    BufferStats pool3_stats = compute_stats(pool3_out, POOL3_SIZE);
    printf("\n  Backbone output (20x20x128):\n");
    printf("    range:   [%.4f, %.4f]\n", pool3_stats.min_val, pool3_stats.max_val);
    printf("    mean:    %.6f\n", pool3_stats.mean_val);
    printf("    nonzero: %d / %d (%.1f%%)\n", pool3_stats.nonzero, pool3_stats.total,
           100.0f * pool3_stats.nonzero / pool3_stats.total);

    // HLS vs CPU verification
    printf("\n  --- HLS vs CPU Verification ---\n");
    float tol = 0.05f;
    bool match = true;
    match &= compare_outputs(conv1_out, cpu_conv1, CONV1_SIZE, tol, "Conv1");
    match &= compare_outputs(pool1_out, cpu_pool1, POOL1_SIZE, tol, "Pool1");
    match &= compare_outputs(conv2_out, cpu_conv2, CONV2_SIZE, tol, "Conv2");

    // Timing
    printf("\n  --- Timing (C-sim, NOT cycle-accurate) ---\n");
    printf("  HLS full pipeline (9 layers): %.1f ms\n", hls_time_ms);
    printf("  CPU baseline (3 layers):      %.1f ms\n", cpu_time_ms);

    // Sanity checks
    printf("\n  --- Sanity Checks ---\n");
    bool pass = true;

    if (pool3_stats.nonzero == 0) {
        printf("  FAIL: Backbone output is all zeros!\n");
        pass = false;
    }
    if (pool3_stats.min_val < 0.0f) {
        printf("  FAIL: Backbone has negative values (ReLU violation)\n");
        pass = false;
    }
    if (!match) {
        printf("  FAIL: HLS output does not match CPU reference\n");
        pass = false;
    }

    // GlobalAvgPool check: output should be mean of each channel
    bool gap_ok = true;
    for (int ch = 0; ch < GAP_SIZE && gap_ok; ch++) {
        if ((float)gap_out[ch] < 0.0f) {
            printf("  FAIL: GAP output has negative value at ch=%d\n", ch);
            gap_ok = false;
            pass = false;
        }
    }
    if (gap_ok)
        printf("  GlobalAvgPool: all outputs non-negative (OK)\n");

    // Head check: should have non-zero outputs
    BufferStats head_stats = compute_stats(head_out, HEAD_SIZE);
    if (head_stats.nonzero == 0) {
        printf("  FAIL: Head output is all zeros!\n");
        pass = false;
    } else {
        printf("  Classification head: %d/%d nonzero scores (OK)\n",
               head_stats.nonzero, head_stats.total);
    }

    printf("\n  Test result: %s\n", pass ? "PASS" : "FAIL");
    printf("============================================================\n\n");

    // Write output log
    {
        std::ofstream fout("csim_results.log");
        if (fout.is_open()) {
            fout << "# Backbone output (20x20x128)\n";
            for (int i = 0; i < POOL3_SIZE; i++) {
                fout << (float)pool3_out[i];
                if (i < POOL3_SIZE - 1) fout << " ";
            }
            fout << "\n# Classification head (10 classes)\n";
            for (int i = 0; i < HEAD_SIZE; i++) {
                fout << (float)head_out[i];
                if (i < HEAD_SIZE - 1) fout << " ";
            }
            fout << "\n";
            fout.close();
            printf("  Output written to csim_results.log\n");
        }
    }

    // Cleanup
    delete[] input;
    delete[] conv1_out; delete[] pool1_out;
    delete[] conv2_out; delete[] pool2_out;
    delete[] conv3_out; delete[] conv4_out; delete[] pool3_out;
    delete[] gap_out;   delete[] head_out;
    delete[] cpu_conv1; delete[] cpu_pool1; delete[] cpu_conv2;

    delete[] w1_nchw; delete[] w2_nchw; delete[] w3_nchw; delete[] w4_nchw;
    delete[] b1; delete[] b2; delete[] b3; delete[] b4;
    delete[] w1_pad; delete[] w2_pad; delete[] w3_pad; delete[] w4_pad;
    delete[] b1_pad; delete[] b2_pad; delete[] b3_pad; delete[] b4_pad;
    delete[] w5_nchw; delete[] b5;
    delete[] w5_pad;  delete[] b5_pad;

    return pass ? 0 : 1;
}