// ============================================================
// FOD MLP accelerator - HLS implementation
//
// Phase-by-phase, correctness-first implementation of:
//
//   64 INT8 -> Dense(W1,b1) -> ReLU -> requant INT8 (32)
//           -> Dense(W2,b2) -> ReLU -> requant INT8 (16)
//           -> Dense(W3,b3) -> INT32 logits (4)
//           -> argmax -> class ID
//
// This is intentionally written as plain, unpipelined,
// unpartitioned C++ so that Vitis HLS C simulation is easy to
// debug against the Python golden reference. Optimization
// (pipelining, array partitioning, AXI interfaces, streaming)
// is deliberately deferred to a later phase.
//
// See the accompanying explanation for a discussion of the
// numerical details (rounding, casting, overflow) that must be
// respected for HLS results to exactly match the Python
// integer_mlp() golden reference.
// ============================================================

#include "mlp.h"
#include <cmath>

// ------------------------------------------------------------
// requantize()
//
// Reproduces:  round(value_int32 * scale_num / scale_den)
// followed by a narrowing cast to int8_t, exactly mirroring the
// Python line:
//
//   A_int8 = np.round(A * S_Z / S_A).astype(np.int8)
//
// Two numerical details matter here:
//
// 1. ROUNDING MODE
//    C's round() rounds halfway cases AWAY FROM ZERO
//    (e.g. round(2.5) == 3.0, round(-2.5) == -3.0).
//    numpy.round() (and therefore np.round in the golden
//    reference) instead uses ROUND-HALF-TO-EVEN ("banker's
//    rounding"): round(2.5) == 2.0, round(3.5) == 4.0.
//    We must NOT use round()/roundf() here, or results can
//    differ on exact .5 ties. Instead we use rint(), which
//    rounds according to the current floating-point rounding
//    mode. The IEEE-754 default rounding mode is
//    round-to-nearest-with-ties-to-even, which is exactly what
//    numpy uses, and Vitis HLS does not change this default in
//    C simulation. This is the detail most likely to cause a
//    silent mismatch if overlooked.
//
// 2. CAST-ON-OVERFLOW BEHAVIOR
//    The golden reference does NOT clip A1_int8 / A2_int8 to
//    [-128, 127] before casting - it relies on the activation
//    scale (S_A1, S_A2) having been calibrated so that values
//    normally fit in INT8 range, and simply lets
//    numpy's .astype(np.int8) do a narrowing (wrapping) cast
//    for any value that doesn't fit. A narrowing cast of an
//    out-of-range value to a signed integer type is technically
//    implementation-defined in C/C++, but on every mainstream
//    platform (and in Vitis HLS) it performs the same two's
//    complement truncation/wraparound that numpy performs, so
//    (int8_t)(int32_t)rounded_value reproduces numpy's behavior.
//    We therefore intentionally do NOT clip here, to stay
//    bit-faithful to the golden reference. If you later decide
//    you want saturating (clipped) behavior instead of wrapping,
//    that is a deliberate deviation from the current golden
//    reference and should be validated against a Python
//    reference that also clips.
//
// 3. PRECISION OF THE SCALES
//    S_Z1/S_A1/S_Z2/S_A2 are stored in mlp_weights.h as 32-bit
//    'float' (this is a limitation of the header file itself,
//    not something introduced here). The original Python session
//    computed S_Z*/S_A* and the ratio S_Z/S_A in double precision
//    before ever narrowing them to float for the header. To stay
//    as close as possible to that original double-precision
//    computation (and minimize additional rounding error on top
//    of what the header already introduced), the arithmetic below
//    is done in 'double', with the float constants promoted to
//    double automatically. We are NOT recalculating or replacing
//    any value from mlp_weights.h - we are just using them at
//    higher precision during the intermediate computation before
//    rounding to the final integer.
// ------------------------------------------------------------
static inline int8_t requantize(int32_t value, float scale_num, float scale_den)
{
    double scaled = (double)value * (double)scale_num / (double)scale_den;
    double rounded = rint(scaled); // round-half-to-even, matches np.round
    return (int8_t)(int32_t)rounded; // wrapping cast, matches np.astype(int8)
}

// ------------------------------------------------------------
// Phase 1: Dense 64 -> 32, ReLU, requantize to INT8
// ------------------------------------------------------------
void layer1(const int8_t input[INPUT_SIZE], int8_t output[HIDDEN1_SIZE])
{
LAYER1_OUT:
    for (int j = 0; j < HIDDEN1_SIZE; j++)
    {
        // INT8 x INT8 -> INT32 accumulation.
        // Casting both operands to int32_t before multiplying
        // guarantees 32-bit multiply/accumulate regardless of
        // int promotion rules, matching:
        //   X_int8.astype(np.int32) @ W1_int8.astype(np.int32)
        int32_t acc = 0;
    LAYER1_MAC:
        for (int i = 0; i < INPUT_SIZE; i++)
        {
            acc += (int32_t)input[i] * (int32_t)W1[i][j];
        }

        // Add INT32 bias (already quantized in mlp_weights.h)
        acc += b1[j];

        // ReLU
        if (acc < 0)
        {
            acc = 0;
        }

        // Requantize INT32 -> INT8 using the layer's accumulator
        // scale (S_Z1) and the next layer's input activation scale
        // (S_A1), exactly like: round(A1 * S_Z1 / S_A1)
        output[j] = requantize(acc, S_Z1, S_A1);
    }
}

// ------------------------------------------------------------
// Phase 2: Dense 32 -> 16, ReLU, requantize to INT8
// ------------------------------------------------------------
void layer2(const int8_t input[HIDDEN1_SIZE], int8_t output[HIDDEN2_SIZE])
{
LAYER2_OUT:
    for (int j = 0; j < HIDDEN2_SIZE; j++)
    {
        int32_t acc = 0;
    LAYER2_MAC:
        for (int i = 0; i < HIDDEN1_SIZE; i++)
        {
            acc += (int32_t)input[i] * (int32_t)W2[i][j];
        }

        acc += b2[j];

        if (acc < 0)
        {
            acc = 0;
        }

        output[j] = requantize(acc, S_Z2, S_A2);
    }
}

// ------------------------------------------------------------
// Phase 3: Dense 16 -> 4. No ReLU, no requantization - the
// output is the final INT32 logit vector, matching:
//   Z3 = A2_int8 @ W3_int8 + b3_int32
// ------------------------------------------------------------
void layer3(const int8_t input[HIDDEN2_SIZE], int32_t logits[OUTPUT_SIZE])
{
LAYER3_OUT:
    for (int j = 0; j < OUTPUT_SIZE; j++)
    {
        int32_t acc = 0;
    LAYER3_MAC:
        for (int i = 0; i < HIDDEN2_SIZE; i++)
        {
            acc += (int32_t)input[i] * (int32_t)W3[i][j];
        }

        acc += b3[j];

        logits[j] = acc;
    }
}

// ------------------------------------------------------------
// Phase 4: argmax over 4 INT32 logits.
//
// numpy.argmax returns the index of the FIRST occurrence of the
// maximum value when there is a tie. We reproduce that by only
// updating best_idx on a STRICT '>' comparison (never '>=').
// ------------------------------------------------------------
int argmax4(const int32_t logits[OUTPUT_SIZE])
{
    int best_idx = 0;
    int32_t best_val = logits[0];

ARGMAX_LOOP:
    for (int i = 1; i < OUTPUT_SIZE; i++)
    {
        if (logits[i] > best_val)
        {
            best_val = logits[i];
            best_idx = i;
        }
    }

    return best_idx;
}

// ------------------------------------------------------------
// Top-level accelerator function.
//
// Plain function-call structure, no AXI/streaming pragmas yet.
// Intermediate activations are kept as local arrays so each
// phase can be inspected independently in the debugger / by
// adding printfs while bringing the design up.
// ------------------------------------------------------------
void mlp_top(
    const int8_t input[INPUT_SIZE],
    int32_t logits[OUTPUT_SIZE],
    int &pred_class)
{
    int8_t a1[HIDDEN1_SIZE];
    int8_t a2[HIDDEN2_SIZE];

    layer1(input, a1);
    layer2(a1, a2);
    layer3(a2, logits);

    pred_class = argmax4(logits);
}
