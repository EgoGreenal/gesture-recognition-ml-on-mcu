/* main_template.c -- Day 13 STM32U5 deployment template for C1j INT8 streaming.
 *
 * Drop-in replacement for CubeIDE-generated Core/Src/main.c. Preserves CubeMX
 * "USER CODE BEGIN ... USER CODE END" regions for HAL_Init / SystemClock_Config /
 * MX_*_Init etc. (you'll need to keep those after CubeMX code-gen runs).
 *
 * UART protocol with host_send_jester.py:
 *   Board:  USART2 (ST-LINK virtual COM, 115200 baud) -- printf debug
 *           USART1 (host data link, 921600 baud)      -- binary frames + result
 *
 *   Per clip (T=8 frames):
 *     Host  ->  Board   : send 'S' (start-of-clip sync byte)
 *     Host  ->  Board   : send 4096 bytes (1 frame = 64*64*1 int8) ... x 8 frames
 *     Board ->  Host    : per frame, send "F %d %lu\n" (frame_idx, cycles)  [debug]
 *     Board ->  Host    : end of clip, send "R %d %d %lu\n" (pred, exit_frame, total_cycles)
 *
 * Early-exit policy: S1, min_exit_frame=5, thresh=0.85 (Day 12 chosen).
 *
 * NOTE: This is a template. You MUST:
 *   1. Keep CubeMX-generated SystemClock_Config / MX_GPIO_Init / MX_USART1/2_UART_Init
 *      etc. from the auto-generated main.c (paste them at bottom of this file or merge)
 *   2. Set system clock to 160 MHz (default 80 -- needs PLL re-config)
 *   3. Confirm USART1 baud = 921600 in CubeMX
 *   4. Verify the generated header name matches stai_student_c1j_ptq_int8.h
 *      (rename if you used a different network name in X-CUBE-AI)
 */
#include "main.h"
#include "stai.h"
#include "student_c1j_ptq_int8.h"
#include "student_c1j_ptq_int8_data.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/* ===== Constants ===== */
#define K_CLASSES           27
#define T_FRAMES            8
#define MIN_EXIT_FRAME      5     /* Day 12 chosen: floor obs >= 0.75 */
#define S1_THRESH           0.85f /* Day 12 chosen: max-softmax threshold */

/* Frame size: 64x64x1 int8 */
#define FRAME_SIZE_BYTES    STAI_STUDENT_C1J_PTQ_INT8_IN_1_SIZE_BYTES  /* 4096 */

/* Output 7 = logits (27-class) -- see student_c1j_ptq_int8.h line 337 */
#define LOGITS_OUT_IDX      6     /* zero-based index of logits in outputs[] */
#define LOGITS_SCALE        STAI_STUDENT_C1J_PTQ_INT8_OUT_7_SCALE     /* 0.188f */
#define LOGITS_ZP           STAI_STUDENT_C1J_PTQ_INT8_OUT_7_ZERO_POINT /* 66 */

/* ===== TSM cache pair mapping ===== */
/* Matched by shape + scale + zp between IN_n and OUT_m (see student_c1j_ptq_int8.h).
 * Each pair: after ai_run, copy outputs[OUT_IDX] -> inputs[IN_IDX] for next frame.
 *
 * INDICES ARE ZERO-BASED (the stai API returns arrays starting at index 0;
 * the IN_n / OUT_n macros in the header are 1-based).
 *
 * Verified via shape + scale match (see day13_checklist.md "wrong cache mapping" row
 * for verification procedure if board results disagree with host).
 */
typedef struct {
    int in_idx;           /* 0-based index in inputs[] (= IN_n - 1, with n in {2..11}) */
    int out_idx;          /* 0-based index in outputs[] (= OUT_m - 1, with m in {1..11}\{7}) */
    int size_bytes;       /* size of this cache buffer (in==out) */
    const char *site_name; /* for debug printf */
} cache_pair_t;

static const cache_pair_t CACHE_PAIRS[] = {
    /* IN_n  OUT_m  bytes  site (b<N> = TSM block N) */
    /* in_idx, out_idx, bytes, name */
    {  1,    2,    512, "b2"  },   /* 16x16x2 */
    {  2,    7,    128, "b4"  },   /*  8x8 x2 */
    {  3,    5,    128, "b5"  },   /*  8x8 x2 */
    {  4,    1,     64, "b7"  },   /*  4x4 x4 */
    {  5,    4,     64, "b8"  },   /*  4x4 x4 */
    {  6,   10,     64, "b9"  },   /*  4x4 x4 */
    {  7,    3,     96, "b11" },   /*  4x4 x6 */
    {  8,    8,     96, "b12" },   /*  4x4 x6 */
    {  9,    0,     40, "b14" },   /*  2x2 x10 */
    { 10,    9,     40, "b15" },   /*  2x2 x10 */
};
#define N_CACHE_PAIRS  (sizeof(CACHE_PAIRS)/sizeof(CACHE_PAIRS[0]))

/* Cache input ZP (for cold-start init -- int8 zero is at ZP not 0) */
static const int8_t CACHE_INIT_ZP[N_CACHE_PAIRS] = {
    STAI_STUDENT_C1J_PTQ_INT8_IN_2_ZERO_POINT,
    STAI_STUDENT_C1J_PTQ_INT8_IN_3_ZERO_POINT,
    STAI_STUDENT_C1J_PTQ_INT8_IN_4_ZERO_POINT,
    STAI_STUDENT_C1J_PTQ_INT8_IN_5_ZERO_POINT,
    STAI_STUDENT_C1J_PTQ_INT8_IN_6_ZERO_POINT,
    STAI_STUDENT_C1J_PTQ_INT8_IN_7_ZERO_POINT,
    STAI_STUDENT_C1J_PTQ_INT8_IN_8_ZERO_POINT,
    STAI_STUDENT_C1J_PTQ_INT8_IN_9_ZERO_POINT,
    STAI_STUDENT_C1J_PTQ_INT8_IN_10_ZERO_POINT,
    STAI_STUDENT_C1J_PTQ_INT8_IN_11_ZERO_POINT,
};

/* ===== Globals ===== */
UART_HandleTypeDef huart1;  /* host data link */
UART_HandleTypeDef huart2;  /* ST-LINK virtual COM debug */

/* Network context (struct aligned to STAI requirements) -- CubeMX auto-generates
 * an aligned activations + weights area; we just need the network struct itself */
__attribute__((aligned(STAI_STUDENT_C1J_PTQ_INT8_CONTEXT_ALIGNMENT)))
static uint8_t g_network_context_buf[STAI_STUDENT_C1J_PTQ_INT8_CONTEXT_SIZE];
static stai_network *g_network = (stai_network *)g_network_context_buf;

/* Activations buffer (preallocated as required by STAI_FLAG_INPUTS|OUTPUTS).
 * Size from header. 68 KB for C1j. */
__attribute__((aligned(8)))
static uint8_t g_activations[STAI_STUDENT_C1J_PTQ_INT8_ACTIVATIONS_SIZE_BYTES];

/* Cached pointers from get_inputs / get_outputs (set once at init) */
static int8_t *g_inputs[STAI_STUDENT_C1J_PTQ_INT8_IN_NUM];
static int8_t *g_outputs[STAI_STUDENT_C1J_PTQ_INT8_OUT_NUM];

/* ===== printf retarget to USART2 (debug) ===== */
int __io_putchar(int ch)
{
    HAL_UART_Transmit(&huart2, (uint8_t *)&ch, 1, HAL_MAX_DELAY);
    return ch;
}

/* ===== DWT cycle counter ===== */
static void dwt_init(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
    DWT->CYCCNT = 0;
}

/* ===== Softmax + S1 early-exit ===== */
static int int8_softmax_argmax_check(const int8_t *q_logits,
                                       float scale, int zp,
                                       int *out_argmax,
                                       float *out_maxp)
{
    /* Dequantize 27 logits to fp32 + numerically stable softmax */
    float fp[K_CLASSES];
    float maxlog = -INFINITY;
    for (int i = 0; i < K_CLASSES; ++i) {
        fp[i] = ((float)q_logits[i] - (float)zp) * scale;
        if (fp[i] > maxlog) maxlog = fp[i];
    }
    float sum = 0.0f;
    int argmax = 0;
    for (int i = 0; i < K_CLASSES; ++i) {
        fp[i] = expf(fp[i] - maxlog);
        sum += fp[i];
    }
    if (sum < 1e-9f) sum = 1e-9f;
    float inv = 1.0f / sum;
    float maxp = 0.0f;
    for (int i = 0; i < K_CLASSES; ++i) {
        fp[i] *= inv;
        if (fp[i] > maxp) { maxp = fp[i]; argmax = i; }
    }
    *out_argmax = argmax;
    *out_maxp = maxp;
    return (maxp > S1_THRESH) ? 1 : 0;
}

/* ===== AI init ===== */
static int ai_init(void)
{
    stai_return_code rc;

    /* Init network into g_network_context_buf */
    rc = stai_student_c1j_ptq_int8_init(g_network);
    if (rc != STAI_SUCCESS) {
        printf("[ERROR] stai_init returned %d\n", (int)rc);
        return -1;
    }

    /* Set activations buffer */
    stai_ptr act_ptr = (stai_ptr)g_activations;
    stai_size n_act = 1;
    rc = stai_student_c1j_ptq_int8_set_activations(g_network, &act_ptr, n_act);
    if (rc != STAI_SUCCESS) {
        printf("[ERROR] set_activations returned %d\n", (int)rc);
        return -1;
    }

    /* Set weights buffer (data is in student_c1j_ptq_int8_data.c).
     * Real symbol from header: `const uint64_t g_student_c1j_ptq_int8_weights_array[92576]`.
     * Cast through uintptr_t to silence -Wcast-qual when handing const data to STAI. */
    stai_ptr w_ptr = (stai_ptr)(uintptr_t)g_student_c1j_ptq_int8_weights_array;
    stai_size n_w = 1;
    rc = stai_student_c1j_ptq_int8_set_weights(g_network, &w_ptr, n_w);
    if (rc != STAI_SUCCESS) {
        printf("[ERROR] set_weights returned %d\n", (int)rc);
        return -1;
    }

    /* Cache the input/output ptr arrays (preallocated inside activations) */
    stai_size n_in = STAI_STUDENT_C1J_PTQ_INT8_IN_NUM;
    rc = stai_student_c1j_ptq_int8_get_inputs(g_network,
                                              (stai_ptr *)g_inputs, &n_in);
    if (rc != STAI_SUCCESS || n_in != STAI_STUDENT_C1J_PTQ_INT8_IN_NUM) {
        printf("[ERROR] get_inputs returned %d n=%lu\n", (int)rc, (unsigned long)n_in);
        return -1;
    }
    stai_size n_out = STAI_STUDENT_C1J_PTQ_INT8_OUT_NUM;
    rc = stai_student_c1j_ptq_int8_get_outputs(g_network,
                                               (stai_ptr *)g_outputs, &n_out);
    if (rc != STAI_SUCCESS || n_out != STAI_STUDENT_C1J_PTQ_INT8_OUT_NUM) {
        printf("[ERROR] get_outputs returned %d n=%lu\n", (int)rc, (unsigned long)n_out);
        return -1;
    }

    printf("[ai_init] OK: %u inputs (%u bytes total), %u outputs (%u bytes total)\n",
           STAI_STUDENT_C1J_PTQ_INT8_IN_NUM,
           STAI_STUDENT_C1J_PTQ_INT8_IN_SIZE_BYTES,
           STAI_STUDENT_C1J_PTQ_INT8_OUT_NUM,
           STAI_STUDENT_C1J_PTQ_INT8_OUT_SIZE_BYTES);
    return 0;
}

/* Reset TSM cache buffers to int8 zero (per-port zero_point) */
static void reset_caches(void)
{
    for (unsigned i = 0; i < N_CACHE_PAIRS; ++i) {
        const cache_pair_t *p = &CACHE_PAIRS[i];
        memset(g_inputs[p->in_idx], CACHE_INIT_ZP[i], p->size_bytes);
    }
}

/* After each frame's ai_run, copy outputs[OUT_IDX] -> inputs[IN_IDX] for next frame */
static void roll_caches(void)
{
    for (unsigned i = 0; i < N_CACHE_PAIRS; ++i) {
        const cache_pair_t *p = &CACHE_PAIRS[i];
        memcpy(g_inputs[p->in_idx], g_outputs[p->out_idx], p->size_bytes);
    }
}

/* ===== Per-clip processing ===== */
static int process_one_clip(void)
{
    uint8_t sync;

    /* Wait for 'S' (start-of-clip sync byte) */
    if (HAL_UART_Receive(&huart1, &sync, 1, HAL_MAX_DELAY) != HAL_OK) return -1;
    if (sync != 'S') {
        printf("[WARN] bad sync byte 0x%02x (expected 'S')\n", sync);
        return -2;
    }

    reset_caches();
    int pred = -1, exit_frame = T_FRAMES - 1;
    uint32_t total_cycles = 0;
    int exited = 0;

    for (int t = 0; t < T_FRAMES; ++t) {
        /* Receive frame_t (4096 bytes int8) directly into IN_1 buffer */
        if (HAL_UART_Receive(&huart1, (uint8_t *)g_inputs[0],
                              FRAME_SIZE_BYTES, HAL_MAX_DELAY) != HAL_OK)
            return -3;

        /* Run inference (sync mode) */
        DWT->CYCCNT = 0;
        stai_return_code rc = stai_student_c1j_ptq_int8_run(g_network, STAI_MODE_SYNC);
        uint32_t cycles = DWT->CYCCNT;
        total_cycles += cycles;

        if (rc != STAI_SUCCESS) {
            printf("[ERROR] ai_run frame %d returned %d\n", t, (int)rc);
            return -4;
        }

        /* Tell host this frame is done + cycles */
        char buf[64];
        int n = snprintf(buf, sizeof(buf), "F %d %lu\n", t, (unsigned long)cycles);
        HAL_UART_Transmit(&huart1, (uint8_t *)buf, n, HAL_MAX_DELAY);

        /* Roll caches for next frame */
        roll_caches();

        if (!exited) {
            /* Apply S1 mf=5 thresh=0.85 */
            int argmax = -1;
            float maxp = 0.0f;
            int s1_fire = int8_softmax_argmax_check(
                (const int8_t *)g_outputs[LOGITS_OUT_IDX],
                LOGITS_SCALE, LOGITS_ZP, &argmax, &maxp);
            if (t >= MIN_EXIT_FRAME && s1_fire) {
                pred = argmax;
                exit_frame = t;
                exited = 1;
                /* Continue loop to keep cache state consistent if host sends all T;
                 * if host knows about exit then it could skip; for simplicity host
                 * always sends T=8 frames and board just stops applying early-exit */
            }
            if (t == T_FRAMES - 1 && !exited) {
                pred = argmax;
                exit_frame = T_FRAMES - 1;
            }
        }
    }

    /* Send final result */
    char buf[80];
    int n = snprintf(buf, sizeof(buf), "R %d %d %lu\n",
                     pred, exit_frame, (unsigned long)total_cycles);
    HAL_UART_Transmit(&huart1, (uint8_t *)buf, n, HAL_MAX_DELAY);
    return 0;
}

/* ===== Main entry ===== */
int main(void)
{
    /* MCU Configuration -- CubeMX-generated, keep as-is */
    HAL_Init();
    SystemClock_Config();      /* MUST set SYSCLK = 160 MHz */
    MX_GPIO_Init();
    MX_USART1_UART_Init();     /* host data, 921600 baud */
    MX_USART2_UART_Init();     /* ST-LINK debug, 115200 baud */

    dwt_init();

    printf("\n[Boot] MLonMCU gesture C1j INT8 streaming ready\n");
    printf("[Boot] SYSCLK = %lu Hz\n", HAL_RCC_GetSysClockFreq());
    printf("[Boot] Flash weights = %u bytes (%u KB)\n",
           STAI_STUDENT_C1J_PTQ_INT8_WEIGHT_1_SIZE_BYTES,
           STAI_STUDENT_C1J_PTQ_INT8_WEIGHT_1_SIZE_BYTES / 1024);
    printf("[Boot] SRAM activations = %u bytes (%u KB)\n",
           STAI_STUDENT_C1J_PTQ_INT8_ACTIVATIONS_SIZE_BYTES,
           STAI_STUDENT_C1J_PTQ_INT8_ACTIVATIONS_SIZE_BYTES / 1024);

    if (ai_init() != 0) {
        printf("[FATAL] ai_init failed, halting\n");
        Error_Handler();
    }

    printf("[Boot] Entering main loop, waiting for clips on USART1...\n");

    while (1) {
        int rc = process_one_clip();
        if (rc != 0) {
            printf("[WARN] process_one_clip returned %d\n", rc);
            HAL_Delay(100);  /* Brief pause before retry */
        }
    }
}

/* ===== Stub for CubeMX-generated functions =====
 * Replace these stubs with whatever CubeMX produces. The CubeMX-generated
 * Core/Src/main.c has full implementations of:
 *   - SystemClock_Config()  -- set up SYSCLK = 160 MHz via PLL
 *   - MX_GPIO_Init()        -- LEDs, debug pins
 *   - MX_USART1_UART_Init() -- BAUD = 921600
 *   - MX_USART2_UART_Init() -- BAUD = 115200, used for printf
 *   - Error_Handler()
 *
 * Copy them verbatim from the CubeMX-generated main.c into this file (or
 * keep this file as your main.c and merge CubeMX's USER CODE BEGIN/END blocks).
 */
extern void SystemClock_Config(void);
extern void MX_GPIO_Init(void);
extern void MX_USART1_UART_Init(void);
extern void MX_USART2_UART_Init(void);
extern void Error_Handler(void);
