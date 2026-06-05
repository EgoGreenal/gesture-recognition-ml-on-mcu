/***** Includes *****/
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "mxc.h"
#include "mxc_device.h"
#include "mxc_delay.h"

#include "uart.h"
#include "led.h"
#include "board.h"

#include "camera.h"
#include "utils.h"
#include "dma.h"

#include "mxc.h"
#include "cnn.h"
#include "early_exit_policy.h"

// #define BUTTON


/*
If BUTTON is defined, you'll need to push PB1 to capture an image frame.  Otherwise, images
will be captured continuously.
*/
// #define BUTTON

#define CAMERA_FREQ (10 * 1000 * 1000)


#define IMAGE_XRES 64
#define IMAGE_YRES 64
#define IMAGE_PIXELS (IMAGE_XRES * IMAGE_YRES)
#define GESTURE_FRAMES 12
#define TARGET_INFERENCE_FPS 12
#define TARGET_DISPLAY_FPS 4
#define DISPLAY_EVERY_N (TARGET_INFERENCE_FPS / TARGET_DISPLAY_FPS)
#define FRAME_PERIOD_US (1000000 / TARGET_INFERENCE_FPS)
#define FRAME_TIMER MXC_TMR1

#define CON_BAUD 2000000
#define ENABLE_EARLY_EXIT_POLICY 0
#define USE_SIGNED_MOTION_INPUT 1
#define ENABLE_STAGE_PROFILE 1
#define PROFILE_TIMER MXC_TMR2
#define MODEL_NAME "jestermotion12res_signed_qat"
#define EARLY_EXIT_FIRST_PREFIX 8
#define EARLY_EXIT_SECOND_PREFIX 10
#define EARLY_EXIT_FULL_PREFIX 12
#define EARLY_EXIT_ACCEPT_COOLDOWN_FRAMES 4

volatile uint32_t cnn_time; // Stopwatch

static uint8_t gesture_frames[GESTURE_FRAMES][IMAGE_PIXELS];
static uint32_t cnn_input_0[IMAGE_PIXELS];
static uint32_t cnn_input_4[IMAGE_PIXELS];
static uint32_t cnn_input_8[IMAGE_PIXELS];
static int frame_head;
static int frames_seen;
static uint32_t inference_frame_count;
#if ENABLE_EARLY_EXIT_POLICY
static int policy_age;
static int policy_cooldown_frames;
#endif
static int last_best_class = 25;
static q15_t last_best_score;
static int32_t last_best_logit;

typedef struct {
    uint32_t capture_us;
    uint32_t get_us;
    uint32_t history_us;
    uint32_t load_us;
    uint32_t cnn_us;
    uint32_t softmax_us;
    uint32_t post_us;
    uint32_t frame_us;
} profile_times_t;

static const char *gesture_labels[CNN_NUM_OUTPUTS] = {
    "Swiping Left",
    "Swiping Right",
    "Swiping Down",
    "Swiping Up",
    "Pushing Hand Away",
    "Pulling Hand In",
    "Sliding Two Fingers Left",
    "Sliding Two Fingers Right",
    "Sliding Two Fingers Down",
    "Sliding Two Fingers Up",
    "Pushing Two Fingers Away",
    "Pulling Two Fingers In",
    "Rolling Hand Forward",
    "Rolling Hand Backward",
    "Turning Hand Clockwise",
    "Turning Hand Counterclockwise",
    "Zooming In With Full Hand",
    "Zooming Out With Full Hand",
    "Zooming In With Two Fingers",
    "Zooming Out With Two Fingers",
    "Thumb Up",
    "Thumb Down",
    "Shaking Hand",
    "Stop Sign",
    "Drumming Fingers",
    "No gesture",
    "Doing other things",
};

void fail(void)
{
  printf("\n*** FAIL ***\n\n");
  while (1);
}


static uint8_t rgb565_to_gray(uint8_t hi, uint8_t lo)
{
    uint8_t r5 = (hi & 0xf8) >> 3;
    uint8_t g6 = ((hi & 0x07) << 3) | ((lo & 0xe0) >> 5);
    uint8_t b5 = lo & 0x1f;
    uint16_t r8 = (uint16_t)r5 * 255 / 31;
    uint16_t g8 = (uint16_t)g6 * 255 / 63;
    uint16_t b8 = (uint16_t)b5 * 255 / 31;

    return (uint8_t)((77 * r8 + 150 * g8 + 29 * b8) >> 8);
}

static void update_frame_history(uint8_t *raw)
{
    uint8_t *dst = gesture_frames[frame_head];

    for (int i = 0; i < IMAGE_PIXELS; i++) {
        dst[i] = rgb565_to_gray(raw[i * 2], raw[i * 2 + 1]);
    }

    if (frames_seen == 0) {
        for (int f = 1; f < GESTURE_FRAMES; f++) {
            memcpy(gesture_frames[f], dst, IMAGE_PIXELS);
        }
    }

    frame_head = (frame_head + 1) % GESTURE_FRAMES;
    if (frames_seen < GESTURE_FRAMES) {
        frames_seen++;
    }
}

void load_input(int prefix_planes)
{
    int8_t *input_0 = (int8_t *)cnn_input_0;
    int8_t *input_4 = (int8_t *)cnn_input_4;
    int8_t *input_8 = (int8_t *)cnn_input_8;

    for (int i = 0; i < IMAGE_PIXELS; i++) {
        int center_index = (frame_head + (GESTURE_FRAMES / 2)) % GESTURE_FRAMES;
        input_0[i * 4 + 0] = (int8_t)(gesture_frames[center_index][i] - 128);

        for (int f = 0; f < GESTURE_FRAMES - 1; f++) {
            int prev_index = (frame_head + f) % GESTURE_FRAMES;
            int curr_index = (frame_head + f + 1) % GESTURE_FRAMES;
            int diff = (int)gesture_frames[curr_index][i] - (int)gesture_frames[prev_index][i];
            int8_t motion;

            if ((f + 1) >= prefix_planes) {
                motion = 0;
            } else if (USE_SIGNED_MOTION_INPUT) {
                motion = (int8_t)(diff / 2);
            } else {
                motion = (int8_t)((diff < 0 ? -diff : diff) - 128);
            }

            if (f < 3) {
                input_0[i * 4 + f + 1] = motion;
            } else if (f < 7) {
                input_4[i * 4 + f - 3] = motion;
            } else {
                input_8[i * 4 + f - 7] = motion;
            }
        }
    }

    memcpy32((uint32_t *)0x50400000, cnn_input_0, IMAGE_PIXELS);
    memcpy32((uint32_t *)0x50408000, cnn_input_4, IMAGE_PIXELS);
    memcpy32((uint32_t *)0x50410000, cnn_input_8, IMAGE_PIXELS);
}

static uint8_t *latest_gray_frame(void)
{
    int history_index = (frame_head + GESTURE_FRAMES - 1) % GESTURE_FRAMES;
    return gesture_frames[history_index];
}

static uint32_t profile_cpu_mhz(void)
{
    uint32_t mhz = SystemCoreClock / 1000000UL;
    return mhz == 0 ? 1 : mhz;
}

static uint32_t profile_sum_us(const profile_times_t *profile)
{
    return profile->capture_us + profile->get_us + profile->history_us + profile->load_us +
           profile->cnn_us + profile->softmax_us + profile->post_us;
}


// Classification layer:
static int32_t ml_data[CNN_NUM_OUTPUTS];
static q15_t ml_softmax[CNN_NUM_OUTPUTS];

void softmax_layer(void)
{
  cnn_unload((uint32_t *) ml_data);
  softmax_q17p14_q15((const q31_t *) ml_data, CNN_NUM_OUTPUTS, ml_softmax);
}

static void format_result(char *result, size_t result_len, uint32_t frame_index, int send_image,
                          uint32_t cnn_us, const char *policy, int prefix, int accepted,
                          int skipped, int best_class, q15_t best_score,
                          const profile_times_t *profile)
{
    int digs = (1000 * best_score + 0x4000) >> 15;
    int tens = digs % 10;
    digs = digs / 10;

    snprintf(result, result_len,
             "Classification results:\nmodel=%s frame=%lu packet=%c cnn_us=%lu policy=%s prefix=%d accepted=%d skipped=%d prof_us=%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu mhz=%lu\nClass %d: %d.%d%%\n%s\n",
             MODEL_NAME,
             (unsigned long)frame_index, send_image ? 'i' : 'r',
             (unsigned long)cnn_us, policy, prefix, accepted, skipped,
             (unsigned long)profile->capture_us,
             (unsigned long)profile->get_us,
             (unsigned long)profile->history_us,
             (unsigned long)profile->load_us,
             (unsigned long)profile->cnn_us,
             (unsigned long)profile->softmax_us,
             (unsigned long)profile->post_us,
             (unsigned long)profile->frame_us,
             (unsigned long)profile_cpu_mhz(),
             best_class, digs, tens, gesture_labels[best_class]);
}

#if ENABLE_EARLY_EXIT_POLICY
static int should_accept_early(int prefix, int class_id, q15_t score)
{
    uint16_t threshold = early_exit_threshold_for(prefix, class_id);

    if (threshold == EARLY_EXIT_DISABLED_THRESHOLD) {
        return 0;
    }

    return ((uint16_t)score) >= threshold;
}
#endif

void process_img(uint32_t frame_index, int send_image, uint32_t capture_wait_us)
{
    uint8_t *raw;
    uint32_t imgLen;
    uint32_t w, h;
    uint32_t cnn_us;
    int i;
    int prefix = EARLY_EXIT_FULL_PREFIX;
    int accepted = 0;
    int skipped = 0;
    const char *policy = "full";
    int best_class = last_best_class;
    q15_t best_score = -32768;
    int32_t best_logit = last_best_logit;
    char result[256];
    profile_times_t profile = {0};

    profile.capture_us = capture_wait_us;

    // Get the details of the image from the camera driver.
#if ENABLE_STAGE_PROFILE
    MXC_TMR_SW_Start(PROFILE_TIMER);
#endif
    camera_get_image(&raw, &imgLen, &w, &h);
#if ENABLE_STAGE_PROFILE
    profile.get_us = MXC_TMR_SW_Stop(PROFILE_TIMER);
    MXC_TMR_SW_Start(PROFILE_TIMER);
#endif

    update_frame_history(raw);
#if ENABLE_STAGE_PROFILE
    profile.history_us = MXC_TMR_SW_Stop(PROFILE_TIMER);
#endif

#if ENABLE_EARLY_EXIT_POLICY
    if (policy_cooldown_frames > 0) {
        policy_cooldown_frames--;
        skipped = 1;
        cnn_us = 0;
        policy = "cooldown";
        prefix = 0;
        best_class = last_best_class;
        best_score = last_best_score;
        best_logit = last_best_logit;
        if (policy_cooldown_frames == 0) {
            policy_age = 0;
        }
        goto send_packet;
    }

    policy_age++;
    if (policy_age < EARLY_EXIT_FIRST_PREFIX) {
        skipped = 1;
        cnn_us = 0;
        policy = "warmup";
        prefix = policy_age;
        best_class = last_best_class;
        best_score = last_best_score;
        best_logit = last_best_logit;
        goto send_packet;
    }

    if (policy_age == EARLY_EXIT_FIRST_PREFIX) {
        prefix = EARLY_EXIT_FIRST_PREFIX;
        policy = "prefix8";
    } else if (policy_age == EARLY_EXIT_SECOND_PREFIX) {
        prefix = EARLY_EXIT_SECOND_PREFIX;
        policy = "prefix10";
    } else if (policy_age >= EARLY_EXIT_FULL_PREFIX) {
        prefix = EARLY_EXIT_FULL_PREFIX;
        policy = "full";
    } else {
        skipped = 1;
        cnn_us = 0;
        policy = "wait";
        prefix = policy_age;
        best_class = last_best_class;
        best_score = last_best_score;
        best_logit = last_best_logit;
        goto send_packet;
    }
#endif

#if ENABLE_STAGE_PROFILE
    MXC_TMR_SW_Start(PROFILE_TIMER);
#endif
    load_input(prefix); // Load prefix-masked 12-plane gesture input
#if ENABLE_STAGE_PROFILE
    profile.load_us = MXC_TMR_SW_Stop(PROFILE_TIMER);
#endif
    cnn_time = 0;
    cnn_start(); // Start CNN processing

    while (cnn_time == 0)
      MXC_LP_EnterSleepMode(); // Wait for CNN

    cnn_us = cnn_time;
    profile.cnn_us = cnn_us;
#if ENABLE_STAGE_PROFILE
    MXC_TMR_SW_Start(PROFILE_TIMER);
#endif
    softmax_layer();
#if ENABLE_STAGE_PROFILE
    profile.softmax_us = MXC_TMR_SW_Stop(PROFILE_TIMER);
    MXC_TMR_SW_Start(PROFILE_TIMER);
#endif

    for (i = 0; i < CNN_NUM_OUTPUTS; i++) {
        if (ml_softmax[i] > best_score) {
            best_score = ml_softmax[i];
            best_class = i;
            best_logit = ml_data[i];
        }
    }

#if ENABLE_EARLY_EXIT_POLICY
    if (prefix == EARLY_EXIT_FIRST_PREFIX || prefix == EARLY_EXIT_SECOND_PREFIX) {
        accepted = should_accept_early(prefix, best_class, best_score);
        if (accepted) {
            policy_cooldown_frames = EARLY_EXIT_ACCEPT_COOLDOWN_FRAMES;
        }
    } else if (prefix == EARLY_EXIT_FULL_PREFIX) {
        policy_age = 0;
    }
#endif

    last_best_class = best_class;
    last_best_score = best_score;
    last_best_logit = best_logit;
#if ENABLE_STAGE_PROFILE
    profile.post_us = MXC_TMR_SW_Stop(PROFILE_TIMER);
#endif

#if ENABLE_EARLY_EXIT_POLICY
send_packet:
#endif
    profile.cnn_us = cnn_us;
    profile.frame_us = profile_sum_us(&profile);
    format_result(result, sizeof(result), frame_index, send_image, cnn_us, policy, prefix,
                  accepted, skipped, best_class, best_score, &profile);

    if (send_image) {
        (void)raw;
        (void)imgLen;
        utils_send_img_to_pc(latest_gray_frame(), IMAGE_PIXELS, w, h, (uint8_t*)"GRAY8");
        utils_send_results_to_pc(result);
    } else {
        utils_send_result_packet_to_pc(result);
    }
}

// *****************************************************************************
int main(void)
{
    int ret = 0;
    int slaveAddress;
    int id;
    int dma_channel;

    /* Enable cache */
    MXC_ICC_Enable(MXC_ICC0);

    /* Set system clock to 100 MHz */
    MXC_SYS_Clock_Select(MXC_SYS_CLOCK_IPO);
    SystemCoreClockUpdate();

    // Initialize DMA for camera interface
    MXC_DMA_Init();
    dma_channel = MXC_DMA_AcquireChannel();

    mxc_uart_regs_t *ConsoleUart = MXC_UART_GET_UART(CONSOLE_UART);

    if ((ret = MXC_UART_Init(ConsoleUart, CON_BAUD, MXC_UART_APB_CLK)) != E_NO_ERROR) {
        return ret;
    }

    // Initialize the camera driver.
    camera_init(CAMERA_FREQ);
    printf("\n\nCamera initialized\n");

    slaveAddress = camera_get_slave_address();
    printf("Camera I2C slave address: %02x\n", slaveAddress);

    // Obtain the manufacturer ID of the camera.
    ret = camera_get_manufacture_id(&id);

    if (ret != STATUS_OK) {
        printf("Error returned from reading camera id. Error %d\n", ret);
        return -1;
    }

    printf("Camera ID detected: %04x\n", id);


    ret = camera_setup(IMAGE_XRES, IMAGE_YRES, PIXFORMAT_RGB565, FIFO_FOUR_BYTE, USE_DMA,
                       dma_channel); // RGB565

    if (ret != STATUS_OK) {
        printf("Error returned from setting up camera. Error %d\n", ret);
        return -1;
    }

    camera_write_reg(0x11, 0x80);

    MXC_Delay(SEC(1));

    printf("Configuring CNN gesture accelerator\n");
    cnn_enable(MXC_S_GCR_PCLKDIV_CNNCLKSEL_PCLK, MXC_S_GCR_PCLKDIV_CNNCLKDIV_DIV1);
    cnn_init();
    cnn_load_weights();
    cnn_load_bias();
    cnn_configure();

    // Start capturing a first camera image frame.
    printf("Starting 12 fps inference / 4 fps image display stream at %d baud\n", CON_BAUD);
#ifdef BUTTON
    while (!PB_Get(0)) {}
#endif

    MXC_TMR_SW_Start(FRAME_TIMER);
    camera_start_capture_image();

    while (1) {
        // Check if image is acquired.
        if (camera_is_image_rcv())
        {
            uint32_t frame_elapsed_us;
            uint32_t capture_wait_us;
            uint32_t frame_index = inference_frame_count++;
            int send_image = ((frame_index % DISPLAY_EVERY_N) == 0);

            capture_wait_us = MXC_TMR_TO_Elapsed(FRAME_TIMER);

            // Process the image, send it through the UART console.
            process_img(frame_index, send_image, capture_wait_us);

            frame_elapsed_us = MXC_TMR_SW_Stop(FRAME_TIMER);
            if (frame_elapsed_us < FRAME_PERIOD_US) {
                MXC_Delay(FRAME_PERIOD_US - frame_elapsed_us);
            }

            // Prepare for another frame capture.
            LED_Toggle(LED_GREEN);
#ifdef BUTTON
            while (!PB_Get(0)) {}
#endif
            MXC_TMR_SW_Start(FRAME_TIMER);
            camera_start_capture_image();
        }
    }

    return ret;
}
