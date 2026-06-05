
/**
  ******************************************************************************
  * @file    app_x-cube-ai.c
  * @author  X-CUBE-AI C code generator
  * @brief   AI program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */

 /*
  * Description
  *   v1.0 - Minimum template to show how to use the Embedded Client API
  *          model. Only one input and one output is supported. All
  *          memory resources are allocated statically (AI_NETWORK_XX, defines
  *          are used).
  *          Re-target of the printf function is out-of-scope.
  *   v2.0 - add multiple IO and/or multiple heap support
  *
  *   For more information, see the embeded documentation:
  *
  *       [1] %X_CUBE_AI_DIR%/Documentation/index.html
  *
  *   X_CUBE_AI_DIR indicates the location where the X-CUBE-AI pack is installed
  *   typical : C:\Users\[user_name]\STM32Cube\Repository\STMicroelectronics\X-CUBE-AI\7.1.0
  */

#ifdef __cplusplus
 extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/

#if defined ( __ICCARM__ )
#elif defined ( __CC_ARM ) || ( __GNUC__ )
#endif

/* System headers */
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <string.h>

#include "app_x-cube-ai.h"
#include "main.h"
#include "ai_datatypes_defines.h"
#include "student_c1j_ptq_int8.h"
#include "student_c1j_ptq_int8_data.h"

/* USER CODE BEGIN includes */
#include "stm32u5xx_hal.h"   /* for DWT->CYCCNT (streaming-step timing) */
/* USER CODE END includes */

/* IO buffers ----------------------------------------------------------------*/

#if !defined(AI_STUDENT_C1J_PTQ_INT8_INPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_in_1[AI_STUDENT_C1J_PTQ_INT8_IN_1_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_in_2[AI_STUDENT_C1J_PTQ_INT8_IN_2_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_in_3[AI_STUDENT_C1J_PTQ_INT8_IN_3_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_in_4[AI_STUDENT_C1J_PTQ_INT8_IN_4_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_in_5[AI_STUDENT_C1J_PTQ_INT8_IN_5_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_in_6[AI_STUDENT_C1J_PTQ_INT8_IN_6_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_in_7[AI_STUDENT_C1J_PTQ_INT8_IN_7_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_in_8[AI_STUDENT_C1J_PTQ_INT8_IN_8_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_in_9[AI_STUDENT_C1J_PTQ_INT8_IN_9_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_in_10[AI_STUDENT_C1J_PTQ_INT8_IN_10_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_in_11[AI_STUDENT_C1J_PTQ_INT8_IN_11_SIZE_BYTES];
ai_i8* data_ins[AI_STUDENT_C1J_PTQ_INT8_IN_NUM] = {
data_in_1,
data_in_2,
data_in_3,
data_in_4,
data_in_5,
data_in_6,
data_in_7,
data_in_8,
data_in_9,
data_in_10,
data_in_11
};
#else
ai_i8* data_ins[AI_STUDENT_C1J_PTQ_INT8_IN_NUM] = {
NULL,
NULL,
NULL,
NULL,
NULL,
NULL,
NULL,
NULL,
NULL,
NULL,
NULL
};
#endif

#if !defined(AI_STUDENT_C1J_PTQ_INT8_OUTPUTS_IN_ACTIVATIONS)
AI_ALIGNED(4) ai_i8 data_out_1[AI_STUDENT_C1J_PTQ_INT8_OUT_1_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_out_2[AI_STUDENT_C1J_PTQ_INT8_OUT_2_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_out_3[AI_STUDENT_C1J_PTQ_INT8_OUT_3_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_out_4[AI_STUDENT_C1J_PTQ_INT8_OUT_4_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_out_5[AI_STUDENT_C1J_PTQ_INT8_OUT_5_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_out_6[AI_STUDENT_C1J_PTQ_INT8_OUT_6_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_out_7[AI_STUDENT_C1J_PTQ_INT8_OUT_7_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_out_8[AI_STUDENT_C1J_PTQ_INT8_OUT_8_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_out_9[AI_STUDENT_C1J_PTQ_INT8_OUT_9_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_out_10[AI_STUDENT_C1J_PTQ_INT8_OUT_10_SIZE_BYTES];
AI_ALIGNED(4) ai_i8 data_out_11[AI_STUDENT_C1J_PTQ_INT8_OUT_11_SIZE_BYTES];
ai_i8* data_outs[AI_STUDENT_C1J_PTQ_INT8_OUT_NUM] = {
data_out_1,
data_out_2,
data_out_3,
data_out_4,
data_out_5,
data_out_6,
data_out_7,
data_out_8,
data_out_9,
data_out_10,
data_out_11
};
#else
ai_i8* data_outs[AI_STUDENT_C1J_PTQ_INT8_OUT_NUM] = {
NULL,
NULL,
NULL,
NULL,
NULL,
NULL,
NULL,
NULL,
NULL,
NULL,
NULL
};
#endif

/* Activations buffers -------------------------------------------------------*/

AI_ALIGNED(32)
static uint8_t pool0[AI_STUDENT_C1J_PTQ_INT8_DATA_ACTIVATION_1_SIZE];

ai_handle data_activations0[] = {pool0};

/* AI objects ----------------------------------------------------------------*/

static ai_handle student_c1j_ptq_int8 = AI_HANDLE_NULL;

static ai_buffer* ai_input;
static ai_buffer* ai_output;

static void ai_log_err(const ai_error err, const char *fct)
{
  /* USER CODE BEGIN log */
  if (fct)
    printf("TEMPLATE - Error (%s) - type=0x%02x code=0x%02x\r\n", fct,
        err.type, err.code);
  else
    printf("TEMPLATE - Error - type=0x%02x code=0x%02x\r\n", err.type, err.code);

  do {} while (1);
  /* USER CODE END log */
}

static int ai_boostrap(ai_handle *act_addr)
{
  ai_error err;

  /* Create and initialize an instance of the model */
  err = ai_student_c1j_ptq_int8_create_and_init(&student_c1j_ptq_int8, act_addr, NULL);
  if (err.type != AI_ERROR_NONE) {
    ai_log_err(err, "ai_student_c1j_ptq_int8_create_and_init");
    return -1;
  }

  ai_input = ai_student_c1j_ptq_int8_inputs_get(student_c1j_ptq_int8, NULL);
  ai_output = ai_student_c1j_ptq_int8_outputs_get(student_c1j_ptq_int8, NULL);

#if defined(AI_STUDENT_C1J_PTQ_INT8_INPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-inputs" option is used, memory buffer can be
   *  used from the activations buffer. This is not mandatory.
   */
  for (int idx=0; idx < AI_STUDENT_C1J_PTQ_INT8_IN_NUM; idx++) {
	data_ins[idx] = ai_input[idx].data;
  }
#else
  for (int idx=0; idx < AI_STUDENT_C1J_PTQ_INT8_IN_NUM; idx++) {
	  ai_input[idx].data = data_ins[idx];
  }
#endif

#if defined(AI_STUDENT_C1J_PTQ_INT8_OUTPUTS_IN_ACTIVATIONS)
  /*  In the case where "--allocate-outputs" option is used, memory buffer can be
   *  used from the activations buffer. This is no mandatory.
   */
  for (int idx=0; idx < AI_STUDENT_C1J_PTQ_INT8_OUT_NUM; idx++) {
	data_outs[idx] = ai_output[idx].data;
  }
#else
  for (int idx=0; idx < AI_STUDENT_C1J_PTQ_INT8_OUT_NUM; idx++) {
	ai_output[idx].data = data_outs[idx];
  }
#endif

  return 0;
}

static int ai_run(void)
{
  ai_i32 batch;

  batch = ai_student_c1j_ptq_int8_run(student_c1j_ptq_int8, ai_input, ai_output);
  if (batch != 1) {
    ai_log_err(ai_student_c1j_ptq_int8_get_error(student_c1j_ptq_int8),
        "ai_student_c1j_ptq_int8_run");
    return -1;
  }

  return 0;
}

/* USER CODE BEGIN 2 */
int acquire_and_process_data(ai_i8* data[])
{
  /* fill the inputs of the c-model
  for (int idx=0; idx < AI_STUDENT_C1J_PTQ_INT8_IN_NUM; idx++ )
  {
      data[idx] = ....
  }

  */
  return 0;
}

int post_process(ai_i8* data[])
{
  /* process the predictions
  for (int idx=0; idx < AI_STUDENT_C1J_PTQ_INT8_OUT_NUM; idx++ )
  {
      data[idx] = ....
  }

  */
  return 0;
}

/* ----------------------------------------------------------------------------
 * C1j streaming step helpers (added for ai-v1)
 *
 * Cache pair binding (0-based indices into ai_input[]/ai_output[]):
 *   in[1]  <- out[2]   cache_b2   16x16x2 = 512 B
 *   in[2]  <- out[7]   cache_b4    8x8x2  = 128 B
 *   in[3]  <- out[5]   cache_b5    8x8x2  = 128 B
 *   in[4]  <- out[1]   cache_b7    4x4x4  =  64 B
 *   in[5]  <- out[4]   cache_b8    4x4x4  =  64 B
 *   in[6]  <- out[10]  cache_b9    4x4x4  =  64 B
 *   in[7]  <- out[3]   cache_b11   4x4x6  =  96 B
 *   in[8]  <- out[8]   cache_b12   4x4x6  =  96 B
 *   in[9]  <- out[0]   cache_b14   2x2x10 =  40 B
 *   in[10] <- out[9]   cache_b15   2x2x10 =  40 B
 *   in[0]  = frame_in0  (64x64x1 = 4096 B)
 *   out[6] = logits     (27 INT8)
 *
 * Bindings derived from student_c1j_ptq_int8_generate_report.txt by matching
 * tensor shapes + quantization params (cache_in and cache_out of the same
 * shift block share an identical QLinear scale/zero-point).
 * -------------------------------------------------------------------------- */

#define AI_C1J_LOGITS_OUT_IDX  6
#define AI_C1J_LOGITS_LEN      27
#define AI_C1J_FRAME_BYTES     AI_STUDENT_C1J_PTQ_INT8_IN_1_SIZE_BYTES  /* 4096 */

static const uint8_t  s_cache_in_idx [10] = { 1, 2, 3, 4, 5,  6, 7, 8, 9, 10};
static const uint8_t  s_cache_out_idx[10] = { 2, 7, 5, 1, 4, 10, 3, 8, 0,  9};
static const uint16_t s_cache_bytes  [10] = {512,128,128,64,64,64, 96,96,40, 40};

static const uint16_t s_input_bytes[AI_STUDENT_C1J_PTQ_INT8_IN_NUM] = {
  AI_STUDENT_C1J_PTQ_INT8_IN_1_SIZE_BYTES,
  AI_STUDENT_C1J_PTQ_INT8_IN_2_SIZE_BYTES,
  AI_STUDENT_C1J_PTQ_INT8_IN_3_SIZE_BYTES,
  AI_STUDENT_C1J_PTQ_INT8_IN_4_SIZE_BYTES,
  AI_STUDENT_C1J_PTQ_INT8_IN_5_SIZE_BYTES,
  AI_STUDENT_C1J_PTQ_INT8_IN_6_SIZE_BYTES,
  AI_STUDENT_C1J_PTQ_INT8_IN_7_SIZE_BYTES,
  AI_STUDENT_C1J_PTQ_INT8_IN_8_SIZE_BYTES,
  AI_STUDENT_C1J_PTQ_INT8_IN_9_SIZE_BYTES,
  AI_STUDENT_C1J_PTQ_INT8_IN_10_SIZE_BYTES,
  AI_STUDENT_C1J_PTQ_INT8_IN_11_SIZE_BYTES,
};

/* Zero every model input (frame + all 10 cache_in tensors).  Call this once
 * before starting a new clip / sequence so the first step sees a clean cache. */
void ai_streaming_reset(void)
{
  if (!ai_input) return;
  for (int i = 0; i < AI_STUDENT_C1J_PTQ_INT8_IN_NUM; i++) {
    if (ai_input[i].data) {
      memset(ai_input[i].data, 0, s_input_bytes[i]);
    }
  }
}

/* Run one streaming inference step.
 *   frame_in    : 4096 INT8 bytes (NHWC, 1x64x64x1).  May be NULL -> use zeros.
 *   logits_out  : 27 INT8 bytes, caller-allocated.
 *   cycles_out  : optional, filled with DWT cycles spent inside ai_run().
 * Returns 0 on success, negative on failure. */
int ai_streaming_step(const int8_t* frame_in, int8_t* logits_out, uint32_t* cycles_out)
{
  if (!student_c1j_ptq_int8 || !ai_input || !ai_output) return -1;

  /* 1) load current frame into in[0] (NULL -> zero frame for dummy mode) */
  if (frame_in) {
    memcpy(ai_input[0].data, frame_in, AI_C1J_FRAME_BYTES);
  } else {
    memset(ai_input[0].data, 0, AI_C1J_FRAME_BYTES);
  }
  /* cache inputs already hold previous step's cache_out (or zeros if reset) */

  /* 2) run inference under DWT-cycle timing */
  uint32_t t0 = DWT->CYCCNT;
  ai_i32 batch = ai_student_c1j_ptq_int8_run(student_c1j_ptq_int8, ai_input, ai_output);
  uint32_t t1 = DWT->CYCCNT;
  if (cycles_out) *cycles_out = (uint32_t)(t1 - t0);

  if (batch != 1) {
    ai_error err = ai_student_c1j_ptq_int8_get_error(student_c1j_ptq_int8);
    printf("AI run error type=0x%02x code=0x%02x\r\n", err.type, err.code);
    return -2;
  }

  /* 3) copy logits */
  if (logits_out) {
    memcpy(logits_out, ai_output[AI_C1J_LOGITS_OUT_IDX].data, AI_C1J_LOGITS_LEN);
  }

  /* 4) feed cache_out back as next-step cache_in for each shift block */
  for (int k = 0; k < 10; k++) {
    int8_t* dst = (int8_t*)ai_input [s_cache_in_idx [k]].data;
    int8_t* src = (int8_t*)ai_output[s_cache_out_idx[k]].data;
    if (dst && src && dst != src) {
      memcpy(dst, src, s_cache_bytes[k]);
    }
  }
  return 0;
}

/* True after MX_X_CUBE_AI_Init() has successfully initialised the network. */
int ai_streaming_ready(void)
{
  return (student_c1j_ptq_int8 != AI_HANDLE_NULL) ? 1 : 0;
}
/* USER CODE END 2 */

/* Entry points --------------------------------------------------------------*/

void MX_X_CUBE_AI_Init(void)
{
    /* USER CODE BEGIN 5 */
  printf("\r\nTEMPLATE - initialization\r\n");

  ai_boostrap(data_activations0);
    /* USER CODE END 5 */
}

void MX_X_CUBE_AI_Process(void)
{
    /* USER CODE BEGIN 6 */
  int res = -1;

  printf("TEMPLATE - run - main loop\r\n");

  if (student_c1j_ptq_int8) {

    do {
      /* 1 - acquire and pre-process input data */
      res = acquire_and_process_data(data_ins);
      /* 2 - process the data - call inference engine */
      if (res == 0)
        res = ai_run();
      /* 3- post-process the predictions */
      if (res == 0)
        res = post_process(data_outs);
    } while (res==0);
  }

  if (res) {
    ai_error err = {AI_ERROR_INVALID_STATE, AI_ERROR_CODE_NETWORK};
    ai_log_err(err, "Process has FAILED");
  }
    /* USER CODE END 6 */
}
#ifdef __cplusplus
}
#endif
