// ============================================================
// Testbench for mlp_top()
//
// Runs the HLS top-level function on known INT8 input vectors
// and compares the resulting INT32 logits and predicted class
// against golden-reference values.
//
// IMPORTANT - where these golden values come from:
// This testbench does NOT invent expected values. The 6 vectors
// below (3 structural edge cases: all-zero, all +127, all -127,
// plus 3 pseudo-random vectors) were run through a Python
// re-implementation of the EXACT algorithm described in your
// integer_mlp() golden reference, using the EXACT weights,
// biases and scales parsed directly out of your mlp_weights.h
// (no values were recomputed or guessed). That gives bit-exact
// expected_logits[]/expected_class values to compare against.
//
// These are useful for exercising the datapath (zero input,
// saturated positive/negative input, generic input) but they are
// NOT drawn from your real image dataset. For a stronger check,
// replace/extend TEST_VECTORS below with real X_test_int8 rows
// and the matching Z3_golden / pred_golden values printed by your
// notebook's integer_mlp() function (see cell that prints
// "Sample i: ... Logits=...") - the struct layout below is built
// so you can paste those in directly.
// ============================================================

#include "mlp.h"
#include <cstdio>

struct TestCase
{
    const char *name;
    int8_t input[INPUT_SIZE];
    int32_t expected_logits[OUTPUT_SIZE];
    int expected_class;
};

static const TestCase TEST_VECTORS[] = {
    {
        "all_zero_input",
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {-322, -490, 168, 294},
        3
    },
    {
        "all_plus_127",
        {127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
         127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
         127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127,
         127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127, 127},
        {7924, -23932, -318, -4811},
        0
    },
    {
        "all_minus_127",
        {-127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
         -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
         -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127,
         -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127, -127},
        {14904, -4627, -2159, -1682},
        0
    },
    {
        "random_1",
        {-105, 70, 39, -16, -17, 91, -106, 50, -76, -103, 7, 121, 60, 67, 55, 73,
         3, -95, 87, -13, 0, -33, -81, 109, 72, 37, -25, 82, 12, -14, -13, -70,
         -104, 14, 99, -111, 91, 84, -57, 34, -85, 66, 51, -37, -110, 120, -14, 100,
         45, 71, 66, -78, -35, -8, -1, -116, 12, -88, 62, 47, 108, 62, -34, 119},
        {-4765, -8416, 10173, -14141},
        2
    },
    {
        "random_2",
        {-23, -44, 103, -33, -108, -8, 75, -79, -9, -94, 48, -6, -43, -70, 16, 43,
         112, -16, -87, 85, 33, 51, -103, -48, 68, 85, -17, 78, 87, -29, 102, -54,
         -66, 47, 35, -92, 85, -77, 78, -126, 76, 73, 71, 42, -7, 52, -57, 72,
         14, -10, 1, 18, -118, -92, -65, -98, -15, 43, 39, -7, 91, 17, -107, 68},
        {-8343, 19736, -7852, 17426},
        1
    },
    {
        "random_3",
        {19, 34, 17, 14, -104, 15, 75, -50, 26, -120, -39, -16, 123, -73, -57, -23,
         126, 90, -119, -68, 82, -113, 91, -56, 107, -53, -17, 41, -95, 15, 1, 72,
         127, 42, -23, -24, -21, 80, -46, -85, -42, -122, -100, -105, 69, 57, 50, -10,
         55, -86, 102, 0, 112, -89, -1, 50, -1, -14, -85, -30, -67, -51, 47, 33},
        {-7375, -2568, 4380, 5206},
        3
    },
};

static const int NUM_TEST_VECTORS = sizeof(TEST_VECTORS) / sizeof(TEST_VECTORS[0]);

int main()
{
    int failures = 0;

    for (int t = 0; t < NUM_TEST_VECTORS; t++)
    {
        const TestCase &tc = TEST_VECTORS[t];

        int32_t logits[OUTPUT_SIZE];
        int pred_class = -1;

        mlp_top(tc.input, logits, pred_class);

        printf("---------------------------------------------\n");
        printf("Test vector: %s\n", tc.name);

        printf("  HLS logits     : [");
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            printf("%6d%s", logits[i], (i < OUTPUT_SIZE - 1) ? ", " : "");
        }
        printf("]\n");

        printf("  Expected logits: [");
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            printf("%6d%s", tc.expected_logits[i], (i < OUTPUT_SIZE - 1) ? ", " : "");
        }
        printf("]\n");

        printf("  HLS class      : %d\n", pred_class);
        printf("  Expected class : %d\n", tc.expected_class);

        bool logits_match = true;
        for (int i = 0; i < OUTPUT_SIZE; i++)
        {
            if (logits[i] != tc.expected_logits[i])
            {
                logits_match = false;
            }
        }
        bool class_match = (pred_class == tc.expected_class);

        if (logits_match && class_match)
        {
            printf("  Result         : PASS\n");
        }
        else
        {
            printf("  Result         : FAIL\n");
            failures++;
        }
    }

    printf("===============================================\n");
    if (failures == 0)
    {
        printf("ALL %d TEST VECTORS PASSED\n", NUM_TEST_VECTORS);
    }
    else
    {
        printf("%d / %d TEST VECTORS FAILED\n", failures, NUM_TEST_VECTORS);
    }

    // Return 0 on success, nonzero on failure so Vitis HLS C
    // simulation reports the correct PASS/FAIL status.
    return (failures == 0) ? 0 : 1;
}
