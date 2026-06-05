/**
  ******************************************************************************
  * @file    student_c1j_ptq_int8.c
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-05-13T19:14:01+0200
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#include "ai_lite_inspect.h"
#include "ai_platform_interface.h"
#include "layers.h"
#include "core_convert.h"
#include "student_c1j_ptq_int8.h"
#include "student_c1j_ptq_int8_details.h"
#include "student_c1j_ptq_int8_data.h"
#include "stai_events.h"

#include "lite_operators.h"

#include "ai_lite_inspect.h"
/*****************************************************************************/
#define STAI_INTERNAL_API_MAJOR               (1)
#define STAI_INTERNAL_API_MINOR               (0)
#define STAI_INTERNAL_API_MICRO               (0)

#define STAI_MAGIC                            (0xB1C00100)

/*****************************************************************************/
#define _STAI_CONCAT_ARG(a, b)     a ## b
#define STAI_CONCAT(a, b)         _STAI_CONCAT_ARG(a, b)

/*!  STAI_CAST SECTION                       *********************************/
#define STAI_CAST(type, expr) \
  ((type)(expr))


/*****************************************************************************/
#define STAI_SIZE(_size) \
  ((stai_size)(_size))

/*****************************************************************************/
#define STAI_INIT_BUFFER(_flags, _size, _address) \
  { \
    .size = (_size), \
    .address = (uintptr_t)(_address), \
    .flags = (_flags), \
  }

#define STAI_INIT_TENSOR(_name, _flags, _fmt, _size_bytes, _shape, _scale, _zeropoint) \
  { \
    .size_bytes = (_size_bytes), \
    .flags = (_flags), \
    .format = (stai_format)(_fmt), \
    .shape = STAI_PACK(_shape), \
    .scale = STAI_PACK(_scale), \
    .zeropoint = STAI_PACK(_zeropoint), \
    .name = (_name) \
  }

#define STAI_INIT_ARRAY(_size, _ptr) \
  { .size = STAI_SIZE(_size), .data = STAI_PACK(_ptr) }


#define STAI_CAST_ARRAY(_type, _size, _ptr) \
  { .size = STAI_SIZE(_size), .data = (_type)STAI_PACK(_ptr) }


#define STAI_DECLARE_ARRAY(_type, _size, ...) \
  { .size = STAI_SIZE(_size), .data = (_type[_size]) { STAI_PACK(__VA_ARGS__) } }


#define STAI_EMPTY_ARRAY() \
  { .size = 0, .data = NULL }


#define STAI_INIT_VERSION(_major, _minor, _micro) \
  { .major = (_major), .minor = (_minor), .micro = (_micro), .reserved = 0x0 }

/*****************************************************************************/
/**  Getters and setters  **/

#define STAI_GET_ARRAY_SIZE(nd_array) \
  (nd_array.size)


#define STAI_GET_ARRAY_ELEM(nd_array, pos) \
  (nd_array.data[(pos)])

#define _STAI_SET_ERROR(net_ctx, cond, value, exit) { \
  if (!(net_ctx)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE; } \
  if (((uintptr_t)net_ctx) & (_STAI_CONTEXT_ALIGNMENT-1)) { return STAI_ERROR_NETWORK_INVALID_CONTEXT_ALIGNMENT; } \
  if (((value) >= STAI_ERROR_GENERIC) && (cond)) { \
    if ((net_ctx)->_return_code == STAI_SUCCESS) { \
      (net_ctx)->_return_code = (value); \
    } \
    return (exit); \
  } \
}

/*****************************************************************************/
/* TODO REMOVE THESE TWO MACROS */
#define STAI_EVENT_NODE_START_CB
#define STAI_EVENT_NODE_STOP_CB

#ifdef STAI_EVENT_NODE_START_CB
#ifndef _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB
  #define _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _start_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(const stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_START, (const void*)&_start_event); \
  }
#endif
#else
  #define _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_START_CB */

#ifdef STAI_EVENT_NODE_STOP_CB
#ifndef _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB
  #define _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
  if (net_ctx->_callback) { \
    const stai_event_node_start_stop _stop_event = { \
      .node_id=(_node_id), \
      .buffers={ \
        .size=(_buffers_size), \
        .data=(stai_ptr const*)(stai_ptr[_buffers_size])STAI_PACK(__VA_ARGS__) \
      } \
    }; \
    net_ctx->_callback(net_ctx->_callback_cookie, STAI_EVENT_NODE_STOP, (const void*)&_stop_event); \
  }
#endif
#else
  #define _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(_node_id, _buffers_size, ...) \
    do { /* _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB() */ } while(0);
#endif      /* STAI_EVENT_NODE_STOP_CB */


/*****************************************************************************/
#define _STAI_STUDENT_C1J_PTQ_INT8_MODEL_SIGNATURE     "0x6a9962b9e2de4212a31af378766543f7"
#define _STAI_STUDENT_C1J_PTQ_INT8_DATETIME            "2026-05-13T19:14:01+0200"
#define _STAI_STUDENT_C1J_PTQ_INT8_COMPILE_DATETIME    __DATE__ " " __TIME__

#define _STAI_CONTEXT_ALIGNMENT        STAI_STUDENT_C1J_PTQ_INT8_CONTEXT_ALIGNMENT

/*****************************************************************************/
#define g_student_c1j_ptq_int8_activations_1     (NULL)




#if defined(HAVE_STUDENT_C1J_PTQ_INT8_INFO)
/*****************************************************************************/
static const stai_network_info g_student_c1j_ptq_int8_info = {
  .model_signature = _STAI_STUDENT_C1J_PTQ_INT8_MODEL_SIGNATURE,
  .c_compile_datetime = _STAI_STUDENT_C1J_PTQ_INT8_COMPILE_DATETIME,
  .c_model_name = STAI_STUDENT_C1J_PTQ_INT8_MODEL_NAME,
  .c_model_datetime = _STAI_STUDENT_C1J_PTQ_INT8_DATETIME,
  .c_model_signature = 0x0,
  .runtime_version = STAI_INIT_VERSION(12, 0, 0),
  .tool_version = STAI_INIT_VERSION(4, 0, 0),
  .api_version = STAI_INIT_VERSION(1, 0, 0),
  .n_macc = STAI_STUDENT_C1J_PTQ_INT8_MACC_NUM,
  .n_nodes = STAI_STUDENT_C1J_PTQ_INT8_NODES_NUM,
  .flags = STAI_STUDENT_C1J_PTQ_INT8_FLAGS,
  .n_inputs = STAI_STUDENT_C1J_PTQ_INT8_IN_NUM,
  .n_outputs = STAI_STUDENT_C1J_PTQ_INT8_OUT_NUM,
  .n_activations = STAI_STUDENT_C1J_PTQ_INT8_ACTIVATIONS_NUM,
  .n_weights = STAI_STUDENT_C1J_PTQ_INT8_WEIGHTS_NUM,
  .n_states = STAI_STUDENT_C1J_PTQ_INT8_STATES_NUM,
  .inputs = (stai_tensor[STAI_STUDENT_C1J_PTQ_INT8_IN_NUM]) {
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_IN_1_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_IN_1_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_IN_1_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_IN_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 64, 64, 1),
      STAI_DECLARE_ARRAY(float, 1, 0.007843137718737125f),
      STAI_DECLARE_ARRAY(int16_t, 1, -1)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_IN_2_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_IN_2_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_IN_2_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_IN_2_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 16, 16, 2),
      STAI_DECLARE_ARRAY(float, 1, 0.04220737889409065f),
      STAI_DECLARE_ARRAY(int16_t, 1, 4)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_IN_3_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_IN_3_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_IN_3_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_IN_3_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 8, 8, 2),
      STAI_DECLARE_ARRAY(float, 1, 0.04509761556982994f),
      STAI_DECLARE_ARRAY(int16_t, 1, -2)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_IN_4_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_IN_4_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_IN_4_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_IN_4_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 8, 8, 2),
      STAI_DECLARE_ARRAY(float, 1, 0.06923279166221619f),
      STAI_DECLARE_ARRAY(int16_t, 1, 17)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_IN_5_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_IN_5_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_IN_5_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_IN_5_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 4, 4, 4),
      STAI_DECLARE_ARRAY(float, 1, 0.05053471401333809f),
      STAI_DECLARE_ARRAY(int16_t, 1, -3)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_IN_6_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_IN_6_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_IN_6_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_IN_6_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 4, 4, 4),
      STAI_DECLARE_ARRAY(float, 1, 0.06210874021053314f),
      STAI_DECLARE_ARRAY(int16_t, 1, -13)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_IN_7_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_IN_7_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_IN_7_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_IN_7_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 4, 4, 4),
      STAI_DECLARE_ARRAY(float, 1, 0.0738293007016182f),
      STAI_DECLARE_ARRAY(int16_t, 1, -6)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_IN_8_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_IN_8_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_IN_8_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_IN_8_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 4, 4, 6),
      STAI_DECLARE_ARRAY(float, 1, 0.04048856720328331f),
      STAI_DECLARE_ARRAY(int16_t, 1, -6)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_IN_9_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_IN_9_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_IN_9_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_IN_9_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 4, 4, 6),
      STAI_DECLARE_ARRAY(float, 1, 0.05003214627504349f),
      STAI_DECLARE_ARRAY(int16_t, 1, -3)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_IN_10_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_IN_10_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_IN_10_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_IN_10_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 2, 2, 10),
      STAI_DECLARE_ARRAY(float, 1, 0.03131386637687683f),
      STAI_DECLARE_ARRAY(int16_t, 1, 10)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_IN_11_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_IN_11_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_IN_11_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_IN_11_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 2, 2, 10),
      STAI_DECLARE_ARRAY(float, 1, 0.038860611617565155f),
      STAI_DECLARE_ARRAY(int16_t, 1, 4)),
    },
    .outputs = (stai_tensor[STAI_STUDENT_C1J_PTQ_INT8_OUT_NUM]) {
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_OUT_1_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_1_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_1_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 2, 2, 10),
      STAI_DECLARE_ARRAY(float, 1, 0.03131386637687683f),
      STAI_DECLARE_ARRAY(int16_t, 1, 10)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_OUT_2_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_2_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_2_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_2_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 4, 4, 4),
      STAI_DECLARE_ARRAY(float, 1, 0.05053471401333809f),
      STAI_DECLARE_ARRAY(int16_t, 1, -3)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_OUT_3_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_3_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_3_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_3_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 16, 16, 2),
      STAI_DECLARE_ARRAY(float, 1, 0.04220737889409065f),
      STAI_DECLARE_ARRAY(int16_t, 1, 4)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_OUT_4_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_4_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_4_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_4_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 4, 4, 6),
      STAI_DECLARE_ARRAY(float, 1, 0.04048856720328331f),
      STAI_DECLARE_ARRAY(int16_t, 1, -6)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_OUT_5_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_5_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_5_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_5_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 4, 4, 4),
      STAI_DECLARE_ARRAY(float, 1, 0.06210874021053314f),
      STAI_DECLARE_ARRAY(int16_t, 1, -13)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_OUT_6_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_6_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_6_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_6_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 8, 8, 2),
      STAI_DECLARE_ARRAY(float, 1, 0.06923279166221619f),
      STAI_DECLARE_ARRAY(int16_t, 1, 17)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_OUT_7_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_7_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_7_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_7_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 2, 1, 27),
      STAI_DECLARE_ARRAY(float, 1, 0.18785002827644348f),
      STAI_DECLARE_ARRAY(int16_t, 1, 66)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_OUT_8_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_8_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_8_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_8_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 8, 8, 2),
      STAI_DECLARE_ARRAY(float, 1, 0.04509761556982994f),
      STAI_DECLARE_ARRAY(int16_t, 1, -2)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_OUT_9_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_9_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_9_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_9_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 4, 4, 6),
      STAI_DECLARE_ARRAY(float, 1, 0.05003214627504349f),
      STAI_DECLARE_ARRAY(int16_t, 1, -3)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_OUT_10_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_10_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_10_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_10_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 2, 2, 10),
      STAI_DECLARE_ARRAY(float, 1, 0.038860611617565155f),
      STAI_DECLARE_ARRAY(int16_t, 1, 4)),
    STAI_INIT_TENSOR(
      STAI_STUDENT_C1J_PTQ_INT8_OUT_11_NAME,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_11_FLAGS,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_11_FORMAT,
      STAI_STUDENT_C1J_PTQ_INT8_OUT_11_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 4, 1, 4, 4, 4),
      STAI_DECLARE_ARRAY(float, 1, 0.0738293007016182f),
      STAI_DECLARE_ARRAY(int16_t, 1, -6)),
    },
  .activations = (stai_tensor[STAI_STUDENT_C1J_PTQ_INT8_ACTIVATIONS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_STUDENT_C1J_PTQ_INT8_ACTIVATION_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_STUDENT_C1J_PTQ_INT8_ACTIVATION_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 69776),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },
  .weights = (stai_tensor[STAI_STUDENT_C1J_PTQ_INT8_WEIGHTS_NUM]) {
    STAI_INIT_TENSOR(
      (NULL),
      STAI_STUDENT_C1J_PTQ_INT8_WEIGHT_1_FLAGS,
      STAI_FORMAT_U8,
      STAI_STUDENT_C1J_PTQ_INT8_WEIGHT_1_SIZE_BYTES,
      STAI_DECLARE_ARRAY(int32_t, 1, 740604),
      STAI_EMPTY_ARRAY(),
      STAI_EMPTY_ARRAY()),
    },

  .states = NULL
};
#endif

#define _STAI_CONTEXT_ACQUIRE(_net_ctx, _net_handle) \
  _stai_student_c1j_ptq_int8_context* _net_ctx = (_stai_student_c1j_ptq_int8_context*)(_net_handle); \
  STAI_ASSERT(_net_ctx != NULL) \
  _STAI_SET_ERROR(_net_ctx, _net_ctx->_magic != STAI_MAGIC, \
                  STAI_ERROR_NETWORK_INVALID_CONTEXT_HANDLE, _net_ctx->_return_code)


/*****************************************************************************/
static
void _stai_student_c1j_ptq_int8_check(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  stai_size idx;

// Check activations status
  for (idx=0; idx<STAI_STUDENT_C1J_PTQ_INT8_ACTIVATIONS_NUM; idx++) {
    if (net_ctx->_activations[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_STUDENT_C1J_PTQ_INT8_ACTIVATIONS_NUM) ? STAI_FLAG_ACTIVATIONS : STAI_FLAG_NONE;
// Check inputs status
  for (idx=0; idx<STAI_STUDENT_C1J_PTQ_INT8_IN_NUM; idx++) {
    if (net_ctx->_inputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_STUDENT_C1J_PTQ_INT8_IN_NUM) ? STAI_FLAG_INPUTS : STAI_FLAG_NONE;

  // Check outputs status
  for (idx=0; idx<STAI_STUDENT_C1J_PTQ_INT8_OUT_NUM; idx++) {
    if (net_ctx->_outputs[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_STUDENT_C1J_PTQ_INT8_OUT_NUM) ? STAI_FLAG_OUTPUTS : STAI_FLAG_NONE;

// Check weights status
  for (idx=0; idx<STAI_STUDENT_C1J_PTQ_INT8_WEIGHTS_NUM; idx++) {
    if (net_ctx->_weights[idx] == NULL) break;
  }
  net_ctx->_flags |= (idx == STAI_STUDENT_C1J_PTQ_INT8_WEIGHTS_NUM) ? STAI_FLAG_WEIGHTS : STAI_FLAG_NONE;
STAI_PRINT("  [_stai_network_check] flags: 0x%08x\n", net_ctx->_flags)
}


/*****************************************************************************/
STAI_API_ENTRY
stai_return_code stai_student_c1j_ptq_int8_init(
  stai_network* network)
{
  /* Memory where to store internal context is provided by applications as a raw byte buffer */
  _stai_student_c1j_ptq_int8_context* net_ctx = (_stai_student_c1j_ptq_int8_context*)(network);
  net_ctx->_return_code = STAI_SUCCESS;
  STAI_PRINT("[Entering Network Init] network(%p) context_size(%d)\n", net_ctx, (int32_t)sizeof(_stai_student_c1j_ptq_int8_context))

  _STAI_SET_ERROR(net_ctx, STAI_STUDENT_C1J_PTQ_INT8_CONTEXT_SIZE != sizeof(_stai_student_c1j_ptq_int8_context),
                 STAI_ERROR_NETWORK_INVALID_CONTEXT_SIZE, net_ctx->_return_code)

  {
    const _stai_student_c1j_ptq_int8_context _student_c1j_ptq_int8_context = {
      ._magic = STAI_MAGIC,
      ._signature = STAI_STUDENT_C1J_PTQ_INT8_MODEL_SIGNATURE,
      ._flags = STAI_STUDENT_C1J_PTQ_INT8_FLAGS,
      ._return_code = STAI_SUCCESS,
      ._callback = NULL,
      ._callback_cookie = NULL,
      ._activations = {
      (stai_ptr)g_student_c1j_ptq_int8_activations_1
      },
      ._weights = {
      (stai_ptr)g_student_c1j_ptq_int8_weights_array
      },
      ._inputs = {
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL},
      ._outputs = {
    NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL},
    };

    // Deep copy of internal context to opaque buffer provided by app
    *net_ctx = _student_c1j_ptq_int8_context;

    _stai_student_c1j_ptq_int8_check(net_ctx);
  }

  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_student_c1j_ptq_int8_deinit(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /*  Reset flags to initial state  */
  net_ctx->_flags = STAI_STUDENT_C1J_PTQ_INT8_FLAGS;
  return net_ctx->_return_code;
}

/*****************************************************************************/



/* Int quant #0 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_5_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04220737889409065f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #1 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_7_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04220737889409065f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #2 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_cache_b2_in0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04220737889409065f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #3 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(concat_8_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04220737889409065f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #4 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_11_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07701683789491653f),
    AI_PACK_INTQ_ZP(17)))

/* Int quant #5 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_12_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.07701683789491653f),
    AI_PACK_INTQ_ZP(17)))

/* Int quant #6 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_15_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04509761556982994f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #7 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_17_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04509761556982994f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #8 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_cache_b4_in0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04509761556982994f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #9 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(concat_18_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04509761556982994f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #10 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_21_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06923279166221619f),
    AI_PACK_INTQ_ZP(17)))

/* Int quant #11 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_22_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06923279166221619f),
    AI_PACK_INTQ_ZP(17)))

/* Int quant #12 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_24_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06923279166221619f),
    AI_PACK_INTQ_ZP(17)))

/* Int quant #13 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_cache_b5_in0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06923279166221619f),
    AI_PACK_INTQ_ZP(17)))

/* Int quant #14 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(concat_25_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06923279166221619f),
    AI_PACK_INTQ_ZP(17)))

/* Int quant #15 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_28_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10022054612636566f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #16 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_29_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.10022054612636566f),
    AI_PACK_INTQ_ZP(14)))

/* Int quant #17 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_32_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05053471401333809f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #18 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_34_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05053471401333809f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #19 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_cache_b7_in0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05053471401333809f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #20 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(concat_35_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05053471401333809f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #21 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_38_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06210874021053314f),
    AI_PACK_INTQ_ZP(-13)))

/* Int quant #22 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_39_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06210874021053314f),
    AI_PACK_INTQ_ZP(-13)))

/* Int quant #23 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_41_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06210874021053314f),
    AI_PACK_INTQ_ZP(-13)))

/* Int quant #24 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_cache_b8_in0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06210874021053314f),
    AI_PACK_INTQ_ZP(-13)))

/* Int quant #25 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(concat_42_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06210874021053314f),
    AI_PACK_INTQ_ZP(-13)))

/* Int quant #26 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_45_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0738293007016182f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #27 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_46_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0738293007016182f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #28 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_48_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0738293007016182f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #29 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_cache_b9_in0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0738293007016182f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #30 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(concat_49_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0738293007016182f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #31 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_52_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09495817869901657f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #32 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_53_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.09495817869901657f),
    AI_PACK_INTQ_ZP(8)))

/* Int quant #33 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_56_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04048856720328331f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #34 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_58_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04048856720328331f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #35 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_cache_b11_in0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04048856720328331f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #36 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(concat_59_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04048856720328331f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #37 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_62_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05003214627504349f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #38 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_63_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05003214627504349f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #39 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_65_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05003214627504349f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #40 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_cache_b12_in0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05003214627504349f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #41 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(concat_66_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05003214627504349f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #42 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_69_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05984853208065033f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #43 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_70_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05984853208065033f),
    AI_PACK_INTQ_ZP(-5)))

/* Int quant #44 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_73_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03131386637687683f),
    AI_PACK_INTQ_ZP(10)))

/* Int quant #45 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_75_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03131386637687683f),
    AI_PACK_INTQ_ZP(10)))

/* Int quant #46 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_cache_b14_in0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03131386637687683f),
    AI_PACK_INTQ_ZP(10)))

/* Int quant #47 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(concat_76_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03131386637687683f),
    AI_PACK_INTQ_ZP(10)))

/* Int quant #48 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_79_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.038860611617565155f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #49 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_80_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.038860611617565155f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #50 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_82_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.038860611617565155f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #51 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(serving_default_cache_b15_in0_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.038860611617565155f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #52 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(concat_83_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.038860611617565155f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #53 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_86_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.056112151592969894f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #54 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(eltwise_87_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.056112151592969894f),
    AI_PACK_INTQ_ZP(5)))

/* Int quant #55 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(conv2d_91_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013553140684962273f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #56 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(pool_92_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007074117660522461f),
    AI_PACK_INTQ_ZP(-128)))

/* Int quant #57 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_81_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.038860611617565155f),
    AI_PACK_INTQ_ZP(4)))

/* Int quant #58 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_74_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03131386637687683f),
    AI_PACK_INTQ_ZP(10)))

/* Int quant #59 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_64_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05003214627504349f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #60 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_57_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04048856720328331f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #61 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_47_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0738293007016182f),
    AI_PACK_INTQ_ZP(-6)))

/* Int quant #62 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_40_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06210874021053314f),
    AI_PACK_INTQ_ZP(-13)))

/* Int quant #63 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_33_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.05053471401333809f),
    AI_PACK_INTQ_ZP(-3)))

/* Int quant #64 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_23_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.06923279166221619f),
    AI_PACK_INTQ_ZP(17)))

/* Int quant #65 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_16_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04509761556982994f),
    AI_PACK_INTQ_ZP(-2)))

/* Int quant #66 */
AI_INTQ_INFO_LIST_OBJ_DECLARE(slice_6_output_array_intq, AI_STATIC,
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1,
  AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.04220737889409065f),
    AI_PACK_INTQ_ZP(4)))



/* Array#0 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#1 */
AI_ARRAY_OBJ_DECLARE(
  slice_7_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3584, AI_STATIC)

/* Array#2 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_cache_b2_in0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 512, AI_STATIC)

/* Array#3 */
AI_ARRAY_OBJ_DECLARE(
  concat_8_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#4 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#5 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_12_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 4096, AI_STATIC)

/* Array#6 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#7 */
AI_ARRAY_OBJ_DECLARE(
  slice_17_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 896, AI_STATIC)

/* Array#8 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_cache_b4_in0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 128, AI_STATIC)

/* Array#9 */
AI_ARRAY_OBJ_DECLARE(
  concat_18_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#10 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#11 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_22_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#12 */
AI_ARRAY_OBJ_DECLARE(
  slice_24_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 896, AI_STATIC)

/* Array#13 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_cache_b5_in0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 128, AI_STATIC)

/* Array#14 */
AI_ARRAY_OBJ_DECLARE(
  concat_25_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#15 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_28_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#16 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_29_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024, AI_STATIC)

/* Array#17 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_32_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#18 */
AI_ARRAY_OBJ_DECLARE(
  slice_34_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 448, AI_STATIC)

/* Array#19 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_cache_b7_in0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 64, AI_STATIC)

/* Array#20 */
AI_ARRAY_OBJ_DECLARE(
  concat_35_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#21 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_38_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#22 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_39_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#23 */
AI_ARRAY_OBJ_DECLARE(
  slice_41_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 448, AI_STATIC)

/* Array#24 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_cache_b8_in0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 64, AI_STATIC)

/* Array#25 */
AI_ARRAY_OBJ_DECLARE(
  concat_42_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#26 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_45_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#27 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_46_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#28 */
AI_ARRAY_OBJ_DECLARE(
  slice_48_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 448, AI_STATIC)

/* Array#29 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_cache_b9_in0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 64, AI_STATIC)

/* Array#30 */
AI_ARRAY_OBJ_DECLARE(
  concat_49_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#31 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_52_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#32 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_53_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512, AI_STATIC)

/* Array#33 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_56_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 768, AI_STATIC)

/* Array#34 */
AI_ARRAY_OBJ_DECLARE(
  slice_58_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 672, AI_STATIC)

/* Array#35 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_cache_b11_in0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 96, AI_STATIC)

/* Array#36 */
AI_ARRAY_OBJ_DECLARE(
  concat_59_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 768, AI_STATIC)

/* Array#37 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_62_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 768, AI_STATIC)

/* Array#38 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_63_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 768, AI_STATIC)

/* Array#39 */
AI_ARRAY_OBJ_DECLARE(
  slice_65_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 672, AI_STATIC)

/* Array#40 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_cache_b12_in0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 96, AI_STATIC)

/* Array#41 */
AI_ARRAY_OBJ_DECLARE(
  concat_66_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 768, AI_STATIC)

/* Array#42 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_69_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 768, AI_STATIC)

/* Array#43 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_70_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 768, AI_STATIC)

/* Array#44 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_73_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 320, AI_STATIC)

/* Array#45 */
AI_ARRAY_OBJ_DECLARE(
  slice_75_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 280, AI_STATIC)

/* Array#46 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_cache_b14_in0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 40, AI_STATIC)

/* Array#47 */
AI_ARRAY_OBJ_DECLARE(
  concat_76_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 320, AI_STATIC)

/* Array#48 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_79_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 320, AI_STATIC)

/* Array#49 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_80_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 320, AI_STATIC)

/* Array#50 */
AI_ARRAY_OBJ_DECLARE(
  slice_82_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 280, AI_STATIC)

/* Array#51 */
AI_ARRAY_OBJ_DECLARE(
  serving_default_cache_b15_in0_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 40, AI_STATIC)

/* Array#52 */
AI_ARRAY_OBJ_DECLARE(
  concat_83_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 320, AI_STATIC)

/* Array#53 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_86_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 320, AI_STATIC)

/* Array#54 */
AI_ARRAY_OBJ_DECLARE(
  eltwise_87_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 320, AI_STATIC)

/* Array#55 */
AI_ARRAY_OBJ_DECLARE(
  conv2d_91_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 5120, AI_STATIC)

/* Array#56 */
AI_ARRAY_OBJ_DECLARE(
  pool_92_output_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1280, AI_STATIC)

/* Array#57 */
AI_ARRAY_OBJ_DECLARE(
  slice_81_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 40, AI_STATIC)

/* Array#58 */
AI_ARRAY_OBJ_DECLARE(
  slice_74_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 40, AI_STATIC)

/* Array#59 */
AI_ARRAY_OBJ_DECLARE(
  slice_64_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 96, AI_STATIC)

/* Array#60 */
AI_ARRAY_OBJ_DECLARE(
  slice_57_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 96, AI_STATIC)

/* Array#61 */
AI_ARRAY_OBJ_DECLARE(
  slice_47_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 64, AI_STATIC)

/* Array#62 */
AI_ARRAY_OBJ_DECLARE(
  slice_40_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 64, AI_STATIC)

/* Array#63 */
AI_ARRAY_OBJ_DECLARE(
  slice_33_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 64, AI_STATIC)

/* Array#64 */
AI_ARRAY_OBJ_DECLARE(
  slice_23_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 128, AI_STATIC)

/* Array#65 */
AI_ARRAY_OBJ_DECLARE(
  slice_16_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 128, AI_STATIC)

/* Array#66 */
AI_ARRAY_OBJ_DECLARE(
  slice_6_output_array, AI_ARRAY_FORMAT_S8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 512, AI_STATIC)



/* Tensor #0 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_output, AI_STATIC,
  146, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 16, 16), AI_STRIDE_INIT(4, 1, 1, 16, 256),
  1, &conv2d_5_output_array, &conv2d_5_output_array_intq)

/* Tensor #1 */
AI_TENSOR_OBJ_DECLARE(
  slice_7_output, AI_STATIC,
  278, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 16, 16), AI_STRIDE_INIT(4, 1, 1, 14, 224),
  1, &slice_7_output_array, &slice_7_output_array_intq)

/* Tensor #2 */
AI_TENSOR_OBJ_DECLARE(
  concat_8_output, AI_STATIC,
  9, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 16, 16), AI_STRIDE_INIT(4, 1, 1, 16, 256),
  1, &concat_8_output_array, &concat_8_output_array_intq)

/* Tensor #3 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_cache_b2_in0_output, AI_STATIC,
  254, 0x1,
  AI_SHAPE_INIT(4, 1, 2, 16, 16), AI_STRIDE_INIT(4, 1, 1, 2, 32),
  1, &serving_default_cache_b2_in0_output_array, &serving_default_cache_b2_in0_output_array_intq)

/* Tensor #4 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_output, AI_STATIC,
  20, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 16, 16), AI_STRIDE_INIT(4, 1, 1, 16, 256),
  1, &conv2d_11_output_array, &conv2d_11_output_array_intq)

/* Tensor #5 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_12_output, AI_STATIC,
  235, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 16, 16), AI_STRIDE_INIT(4, 1, 1, 16, 256),
  1, &eltwise_12_output_array, &eltwise_12_output_array_intq)

/* Tensor #6 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_output, AI_STATIC,
  33, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 8, 8), AI_STRIDE_INIT(4, 1, 1, 16, 128),
  1, &conv2d_15_output_array, &conv2d_15_output_array_intq)

/* Tensor #7 */
AI_TENSOR_OBJ_DECLARE(
  slice_17_output, AI_STATIC,
  262, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 8, 8), AI_STRIDE_INIT(4, 1, 1, 14, 112),
  1, &slice_17_output_array, &slice_17_output_array_intq)

/* Tensor #8 */
AI_TENSOR_OBJ_DECLARE(
  concat_18_output, AI_STATIC,
  0, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 8, 8), AI_STRIDE_INIT(4, 1, 1, 16, 128),
  1, &concat_18_output_array, &concat_18_output_array_intq)

/* Tensor #9 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_cache_b4_in0_output, AI_STATIC,
  255, 0x1,
  AI_SHAPE_INIT(4, 1, 2, 8, 8), AI_STRIDE_INIT(4, 1, 1, 2, 16),
  1, &serving_default_cache_b4_in0_output_array, &serving_default_cache_b4_in0_output_array_intq)

/* Tensor #10 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_output, AI_STATIC,
  51, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 8, 8), AI_STRIDE_INIT(4, 1, 1, 16, 128),
  1, &conv2d_21_output_array, &conv2d_21_output_array_intq)

/* Tensor #11 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_22_output, AI_STATIC,
  236, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 8, 8), AI_STRIDE_INIT(4, 1, 1, 16, 128),
  1, &eltwise_22_output_array, &eltwise_22_output_array_intq)

/* Tensor #12 */
AI_TENSOR_OBJ_DECLARE(
  slice_24_output, AI_STATIC,
  264, 0x1,
  AI_SHAPE_INIT(4, 1, 14, 8, 8), AI_STRIDE_INIT(4, 1, 1, 14, 112),
  1, &slice_24_output_array, &slice_24_output_array_intq)

/* Tensor #13 */
AI_TENSOR_OBJ_DECLARE(
  concat_25_output, AI_STATIC,
  1, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 8, 8), AI_STRIDE_INIT(4, 1, 1, 16, 128),
  1, &concat_25_output_array, &concat_25_output_array_intq)

/* Tensor #14 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_cache_b5_in0_output, AI_STATIC,
  256, 0x1,
  AI_SHAPE_INIT(4, 1, 2, 8, 8), AI_STRIDE_INIT(4, 1, 1, 2, 16),
  1, &serving_default_cache_b5_in0_output_array, &serving_default_cache_b5_in0_output_array_intq)

/* Tensor #15 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_28_output, AI_STATIC,
  64, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 8, 8), AI_STRIDE_INIT(4, 1, 1, 16, 128),
  1, &conv2d_28_output_array, &conv2d_28_output_array_intq)

/* Tensor #16 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_29_output, AI_STATIC,
  237, 0x1,
  AI_SHAPE_INIT(4, 1, 16, 8, 8), AI_STRIDE_INIT(4, 1, 1, 16, 128),
  1, &eltwise_29_output_array, &eltwise_29_output_array_intq)

/* Tensor #17 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_32_output, AI_STATIC,
  81, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 4, 4), AI_STRIDE_INIT(4, 1, 1, 32, 128),
  1, &conv2d_32_output_array, &conv2d_32_output_array_intq)

/* Tensor #18 */
AI_TENSOR_OBJ_DECLARE(
  slice_34_output, AI_STATIC,
  266, 0x1,
  AI_SHAPE_INIT(4, 1, 28, 4, 4), AI_STRIDE_INIT(4, 1, 1, 28, 112),
  1, &slice_34_output_array, &slice_34_output_array_intq)

/* Tensor #19 */
AI_TENSOR_OBJ_DECLARE(
  concat_35_output, AI_STATIC,
  2, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 4, 4), AI_STRIDE_INIT(4, 1, 1, 32, 128),
  1, &concat_35_output_array, &concat_35_output_array_intq)

/* Tensor #20 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_cache_b7_in0_output, AI_STATIC,
  257, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 4, 4), AI_STRIDE_INIT(4, 1, 1, 4, 16),
  1, &serving_default_cache_b7_in0_output_array, &serving_default_cache_b7_in0_output_array_intq)

/* Tensor #21 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_38_output, AI_STATIC,
  94, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 4, 4), AI_STRIDE_INIT(4, 1, 1, 32, 128),
  1, &conv2d_38_output_array, &conv2d_38_output_array_intq)

/* Tensor #22 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_39_output, AI_STATIC,
  238, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 4, 4), AI_STRIDE_INIT(4, 1, 1, 32, 128),
  1, &eltwise_39_output_array, &eltwise_39_output_array_intq)

/* Tensor #23 */
AI_TENSOR_OBJ_DECLARE(
  slice_41_output, AI_STATIC,
  268, 0x1,
  AI_SHAPE_INIT(4, 1, 28, 4, 4), AI_STRIDE_INIT(4, 1, 1, 28, 112),
  1, &slice_41_output_array, &slice_41_output_array_intq)

/* Tensor #24 */
AI_TENSOR_OBJ_DECLARE(
  concat_42_output, AI_STATIC,
  3, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 4, 4), AI_STRIDE_INIT(4, 1, 1, 32, 128),
  1, &concat_42_output_array, &concat_42_output_array_intq)

/* Tensor #25 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_cache_b8_in0_output, AI_STATIC,
  258, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 4, 4), AI_STRIDE_INIT(4, 1, 1, 4, 16),
  1, &serving_default_cache_b8_in0_output_array, &serving_default_cache_b8_in0_output_array_intq)

/* Tensor #26 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_45_output, AI_STATIC,
  111, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 4, 4), AI_STRIDE_INIT(4, 1, 1, 32, 128),
  1, &conv2d_45_output_array, &conv2d_45_output_array_intq)

/* Tensor #27 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_46_output, AI_STATIC,
  239, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 4, 4), AI_STRIDE_INIT(4, 1, 1, 32, 128),
  1, &eltwise_46_output_array, &eltwise_46_output_array_intq)

/* Tensor #28 */
AI_TENSOR_OBJ_DECLARE(
  slice_48_output, AI_STATIC,
  270, 0x1,
  AI_SHAPE_INIT(4, 1, 28, 4, 4), AI_STRIDE_INIT(4, 1, 1, 28, 112),
  1, &slice_48_output_array, &slice_48_output_array_intq)

/* Tensor #29 */
AI_TENSOR_OBJ_DECLARE(
  concat_49_output, AI_STATIC,
  4, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 4, 4), AI_STRIDE_INIT(4, 1, 1, 32, 128),
  1, &concat_49_output_array, &concat_49_output_array_intq)

/* Tensor #30 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_cache_b9_in0_output, AI_STATIC,
  259, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 4, 4), AI_STRIDE_INIT(4, 1, 1, 4, 16),
  1, &serving_default_cache_b9_in0_output_array, &serving_default_cache_b9_in0_output_array_intq)

/* Tensor #31 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_52_output, AI_STATIC,
  129, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 4, 4), AI_STRIDE_INIT(4, 1, 1, 32, 128),
  1, &conv2d_52_output_array, &conv2d_52_output_array_intq)

/* Tensor #32 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_53_output, AI_STATIC,
  240, 0x1,
  AI_SHAPE_INIT(4, 1, 32, 4, 4), AI_STRIDE_INIT(4, 1, 1, 32, 128),
  1, &eltwise_53_output_array, &eltwise_53_output_array_intq)

/* Tensor #33 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_56_output, AI_STATIC,
  142, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 4, 4), AI_STRIDE_INIT(4, 1, 1, 48, 192),
  1, &conv2d_56_output_array, &conv2d_56_output_array_intq)

/* Tensor #34 */
AI_TENSOR_OBJ_DECLARE(
  slice_58_output, AI_STATIC,
  272, 0x1,
  AI_SHAPE_INIT(4, 1, 42, 4, 4), AI_STRIDE_INIT(4, 1, 1, 42, 168),
  1, &slice_58_output_array, &slice_58_output_array_intq)

/* Tensor #35 */
AI_TENSOR_OBJ_DECLARE(
  concat_59_output, AI_STATIC,
  5, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 4, 4), AI_STRIDE_INIT(4, 1, 1, 48, 192),
  1, &concat_59_output_array, &concat_59_output_array_intq)

/* Tensor #36 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_cache_b11_in0_output, AI_STATIC,
  250, 0x1,
  AI_SHAPE_INIT(4, 1, 6, 4, 4), AI_STRIDE_INIT(4, 1, 1, 6, 24),
  1, &serving_default_cache_b11_in0_output_array, &serving_default_cache_b11_in0_output_array_intq)

/* Tensor #37 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_62_output, AI_STATIC,
  159, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 4, 4), AI_STRIDE_INIT(4, 1, 1, 48, 192),
  1, &conv2d_62_output_array, &conv2d_62_output_array_intq)

/* Tensor #38 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_63_output, AI_STATIC,
  241, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 4, 4), AI_STRIDE_INIT(4, 1, 1, 48, 192),
  1, &eltwise_63_output_array, &eltwise_63_output_array_intq)

/* Tensor #39 */
AI_TENSOR_OBJ_DECLARE(
  slice_65_output, AI_STATIC,
  274, 0x1,
  AI_SHAPE_INIT(4, 1, 42, 4, 4), AI_STRIDE_INIT(4, 1, 1, 42, 168),
  1, &slice_65_output_array, &slice_65_output_array_intq)

/* Tensor #40 */
AI_TENSOR_OBJ_DECLARE(
  concat_66_output, AI_STATIC,
  6, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 4, 4), AI_STRIDE_INIT(4, 1, 1, 48, 192),
  1, &concat_66_output_array, &concat_66_output_array_intq)

/* Tensor #41 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_cache_b12_in0_output, AI_STATIC,
  251, 0x1,
  AI_SHAPE_INIT(4, 1, 6, 4, 4), AI_STRIDE_INIT(4, 1, 1, 6, 24),
  1, &serving_default_cache_b12_in0_output_array, &serving_default_cache_b12_in0_output_array_intq)

/* Tensor #42 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_69_output, AI_STATIC,
  172, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 4, 4), AI_STRIDE_INIT(4, 1, 1, 48, 192),
  1, &conv2d_69_output_array, &conv2d_69_output_array_intq)

/* Tensor #43 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_70_output, AI_STATIC,
  242, 0x1,
  AI_SHAPE_INIT(4, 1, 48, 4, 4), AI_STRIDE_INIT(4, 1, 1, 48, 192),
  1, &eltwise_70_output_array, &eltwise_70_output_array_intq)

/* Tensor #44 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_73_output, AI_STATIC,
  185, 0x1,
  AI_SHAPE_INIT(4, 1, 80, 2, 2), AI_STRIDE_INIT(4, 1, 1, 80, 160),
  1, &conv2d_73_output_array, &conv2d_73_output_array_intq)

/* Tensor #45 */
AI_TENSOR_OBJ_DECLARE(
  slice_75_output, AI_STATIC,
  277, 0x1,
  AI_SHAPE_INIT(4, 1, 70, 2, 2), AI_STRIDE_INIT(4, 1, 1, 70, 140),
  1, &slice_75_output_array, &slice_75_output_array_intq)

/* Tensor #46 */
AI_TENSOR_OBJ_DECLARE(
  concat_76_output, AI_STATIC,
  7, 0x1,
  AI_SHAPE_INIT(4, 1, 80, 2, 2), AI_STRIDE_INIT(4, 1, 1, 80, 160),
  1, &concat_76_output_array, &concat_76_output_array_intq)

/* Tensor #47 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_cache_b14_in0_output, AI_STATIC,
  252, 0x1,
  AI_SHAPE_INIT(4, 1, 10, 2, 2), AI_STRIDE_INIT(4, 1, 1, 10, 20),
  1, &serving_default_cache_b14_in0_output_array, &serving_default_cache_b14_in0_output_array_intq)

/* Tensor #48 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_79_output, AI_STATIC,
  198, 0x1,
  AI_SHAPE_INIT(4, 1, 80, 2, 2), AI_STRIDE_INIT(4, 1, 1, 80, 160),
  1, &conv2d_79_output_array, &conv2d_79_output_array_intq)

/* Tensor #49 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_80_output, AI_STATIC,
  243, 0x1,
  AI_SHAPE_INIT(4, 1, 80, 2, 2), AI_STRIDE_INIT(4, 1, 1, 80, 160),
  1, &eltwise_80_output_array, &eltwise_80_output_array_intq)

/* Tensor #50 */
AI_TENSOR_OBJ_DECLARE(
  slice_82_output, AI_STATIC,
  280, 0x1,
  AI_SHAPE_INIT(4, 1, 70, 2, 2), AI_STRIDE_INIT(4, 1, 1, 70, 140),
  1, &slice_82_output_array, &slice_82_output_array_intq)

/* Tensor #51 */
AI_TENSOR_OBJ_DECLARE(
  concat_83_output, AI_STATIC,
  8, 0x1,
  AI_SHAPE_INIT(4, 1, 80, 2, 2), AI_STRIDE_INIT(4, 1, 1, 80, 160),
  1, &concat_83_output_array, &concat_83_output_array_intq)

/* Tensor #52 */
AI_TENSOR_OBJ_DECLARE(
  serving_default_cache_b15_in0_output, AI_STATIC,
  253, 0x1,
  AI_SHAPE_INIT(4, 1, 10, 2, 2), AI_STRIDE_INIT(4, 1, 1, 10, 20),
  1, &serving_default_cache_b15_in0_output_array, &serving_default_cache_b15_in0_output_array_intq)

/* Tensor #53 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_86_output, AI_STATIC,
  211, 0x1,
  AI_SHAPE_INIT(4, 1, 80, 2, 2), AI_STRIDE_INIT(4, 1, 1, 80, 160),
  1, &conv2d_86_output_array, &conv2d_86_output_array_intq)

/* Tensor #54 */
AI_TENSOR_OBJ_DECLARE(
  eltwise_87_output, AI_STATIC,
  244, 0x1,
  AI_SHAPE_INIT(4, 1, 80, 2, 2), AI_STRIDE_INIT(4, 1, 1, 80, 160),
  1, &eltwise_87_output_array, &eltwise_87_output_array_intq)

/* Tensor #55 */
AI_TENSOR_OBJ_DECLARE(
  conv2d_91_output, AI_STATIC,
  228, 0x1,
  AI_SHAPE_INIT(4, 1, 1280, 2, 2), AI_STRIDE_INIT(4, 1, 1, 1280, 2560),
  1, &conv2d_91_output_array, &conv2d_91_output_array_intq)

/* Tensor #56 */
AI_TENSOR_OBJ_DECLARE(
  pool_92_output, AI_STATIC,
  249, 0x1,
  AI_SHAPE_INIT(4, 1, 1280, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1280, 1280),
  1, &pool_92_output_array, &pool_92_output_array_intq)

/* Tensor #57 */
AI_TENSOR_OBJ_DECLARE(
  slice_81_output, AI_STATIC,
  279, 0x1,
  AI_SHAPE_INIT(4, 1, 10, 2, 2), AI_STRIDE_INIT(4, 1, 1, 10, 20),
  1, &slice_81_output_array, &slice_81_output_array_intq)

/* Tensor #58 */
AI_TENSOR_OBJ_DECLARE(
  slice_74_output, AI_STATIC,
  276, 0x1,
  AI_SHAPE_INIT(4, 1, 10, 2, 2), AI_STRIDE_INIT(4, 1, 1, 10, 20),
  1, &slice_74_output_array, &slice_74_output_array_intq)

/* Tensor #59 */
AI_TENSOR_OBJ_DECLARE(
  slice_64_output, AI_STATIC,
  273, 0x1,
  AI_SHAPE_INIT(4, 1, 6, 4, 4), AI_STRIDE_INIT(4, 1, 1, 6, 24),
  1, &slice_64_output_array, &slice_64_output_array_intq)

/* Tensor #60 */
AI_TENSOR_OBJ_DECLARE(
  slice_57_output, AI_STATIC,
  271, 0x1,
  AI_SHAPE_INIT(4, 1, 6, 4, 4), AI_STRIDE_INIT(4, 1, 1, 6, 24),
  1, &slice_57_output_array, &slice_57_output_array_intq)

/* Tensor #61 */
AI_TENSOR_OBJ_DECLARE(
  slice_47_output, AI_STATIC,
  269, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 4, 4), AI_STRIDE_INIT(4, 1, 1, 4, 16),
  1, &slice_47_output_array, &slice_47_output_array_intq)

/* Tensor #62 */
AI_TENSOR_OBJ_DECLARE(
  slice_40_output, AI_STATIC,
  267, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 4, 4), AI_STRIDE_INIT(4, 1, 1, 4, 16),
  1, &slice_40_output_array, &slice_40_output_array_intq)

/* Tensor #63 */
AI_TENSOR_OBJ_DECLARE(
  slice_33_output, AI_STATIC,
  265, 0x1,
  AI_SHAPE_INIT(4, 1, 4, 4, 4), AI_STRIDE_INIT(4, 1, 1, 4, 16),
  1, &slice_33_output_array, &slice_33_output_array_intq)

/* Tensor #64 */
AI_TENSOR_OBJ_DECLARE(
  slice_23_output, AI_STATIC,
  263, 0x1,
  AI_SHAPE_INIT(4, 1, 2, 8, 8), AI_STRIDE_INIT(4, 1, 1, 2, 16),
  1, &slice_23_output_array, &slice_23_output_array_intq)

/* Tensor #65 */
AI_TENSOR_OBJ_DECLARE(
  slice_16_output, AI_STATIC,
  261, 0x1,
  AI_SHAPE_INIT(4, 1, 2, 8, 8), AI_STRIDE_INIT(4, 1, 1, 2, 16),
  1, &slice_16_output_array, &slice_16_output_array_intq)

/* Tensor #66 */
AI_TENSOR_OBJ_DECLARE(
  slice_6_output, AI_STATIC,
  275, 0x1,
  AI_SHAPE_INIT(4, 1, 2, 16, 16), AI_STRIDE_INIT(4, 1, 1, 2, 32),
  1, &slice_6_output_array, &slice_6_output_array_intq)



AI_STATIC_CONST ai_u8 slice_7_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_7_axes, AI_ARRAY_FORMAT_U8,
    slice_7_axes_data, slice_7_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_7_starts_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_7_starts, AI_ARRAY_FORMAT_S16,
    slice_7_starts_data, slice_7_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_7_ends_data[] = { 16 };
AI_ARRAY_OBJ_DECLARE(
    slice_7_ends, AI_ARRAY_FORMAT_S16,
    slice_7_ends_data, slice_7_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_7_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_7_layer, 7,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_7_chain,
  NULL, &slice_7_layer, AI_STATIC, 
  .axes = &slice_7_axes, 
  .starts = &slice_7_starts, 
  .ends = &slice_7_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &serving_default_cache_b2_in0_output, &slice_7_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_8_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_8_layer, 8,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_8_chain,
  NULL, &concat_8_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &concat_8_output, &conv2d_11_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_12_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_12_layer, 12,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_12_chain,
  NULL, &eltwise_12_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_u8 slice_17_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_17_axes, AI_ARRAY_FORMAT_U8,
    slice_17_axes_data, slice_17_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_17_starts_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_17_starts, AI_ARRAY_FORMAT_S16,
    slice_17_starts_data, slice_17_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_17_ends_data[] = { 16 };
AI_ARRAY_OBJ_DECLARE(
    slice_17_ends, AI_ARRAY_FORMAT_S16,
    slice_17_ends_data, slice_17_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_17_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_17_layer, 17,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_17_chain,
  NULL, &slice_17_layer, AI_STATIC, 
  .axes = &slice_17_axes, 
  .starts = &slice_17_starts, 
  .ends = &slice_17_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &serving_default_cache_b4_in0_output, &slice_17_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_18_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_18_layer, 18,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_18_chain,
  NULL, &concat_18_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_22_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &concat_18_output, &conv2d_21_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_22_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_22_layer, 22,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_22_chain,
  NULL, &eltwise_22_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_u8 slice_24_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_24_axes, AI_ARRAY_FORMAT_U8,
    slice_24_axes_data, slice_24_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_24_starts_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_24_starts, AI_ARRAY_FORMAT_S16,
    slice_24_starts_data, slice_24_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_24_ends_data[] = { 16 };
AI_ARRAY_OBJ_DECLARE(
    slice_24_ends, AI_ARRAY_FORMAT_S16,
    slice_24_ends_data, slice_24_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_24_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_24_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_24_layer, 24,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_24_chain,
  NULL, &slice_24_layer, AI_STATIC, 
  .axes = &slice_24_axes, 
  .starts = &slice_24_starts, 
  .ends = &slice_24_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_25_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &serving_default_cache_b5_in0_output, &slice_24_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_25_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_25_layer, 25,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_25_chain,
  NULL, &concat_25_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_29_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &concat_25_output, &conv2d_28_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_29_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_29_layer, 29,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_29_chain,
  NULL, &eltwise_29_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_u8 slice_34_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_34_axes, AI_ARRAY_FORMAT_U8,
    slice_34_axes_data, slice_34_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_34_starts_data[] = { 4 };
AI_ARRAY_OBJ_DECLARE(
    slice_34_starts, AI_ARRAY_FORMAT_S16,
    slice_34_starts_data, slice_34_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_34_ends_data[] = { 32 };
AI_ARRAY_OBJ_DECLARE(
    slice_34_ends, AI_ARRAY_FORMAT_S16,
    slice_34_ends_data, slice_34_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_34_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_32_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_34_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_34_layer, 34,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_34_chain,
  NULL, &slice_34_layer, AI_STATIC, 
  .axes = &slice_34_axes, 
  .starts = &slice_34_starts, 
  .ends = &slice_34_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_35_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &serving_default_cache_b7_in0_output, &slice_34_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_35_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_35_layer, 35,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_35_chain,
  NULL, &concat_35_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_39_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &concat_35_output, &conv2d_38_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_39_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_39_layer, 39,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_39_chain,
  NULL, &eltwise_39_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_u8 slice_41_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_41_axes, AI_ARRAY_FORMAT_U8,
    slice_41_axes_data, slice_41_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_41_starts_data[] = { 4 };
AI_ARRAY_OBJ_DECLARE(
    slice_41_starts, AI_ARRAY_FORMAT_S16,
    slice_41_starts_data, slice_41_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_41_ends_data[] = { 32 };
AI_ARRAY_OBJ_DECLARE(
    slice_41_ends, AI_ARRAY_FORMAT_S16,
    slice_41_ends_data, slice_41_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_41_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_39_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_41_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_41_layer, 41,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_41_chain,
  NULL, &slice_41_layer, AI_STATIC, 
  .axes = &slice_41_axes, 
  .starts = &slice_41_starts, 
  .ends = &slice_41_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_42_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &serving_default_cache_b8_in0_output, &slice_41_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_42_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_42_layer, 42,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_42_chain,
  NULL, &concat_42_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_46_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &concat_42_output, &conv2d_45_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_46_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_46_layer, 46,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_46_chain,
  NULL, &eltwise_46_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_u8 slice_48_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_48_axes, AI_ARRAY_FORMAT_U8,
    slice_48_axes_data, slice_48_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_48_starts_data[] = { 4 };
AI_ARRAY_OBJ_DECLARE(
    slice_48_starts, AI_ARRAY_FORMAT_S16,
    slice_48_starts_data, slice_48_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_48_ends_data[] = { 32 };
AI_ARRAY_OBJ_DECLARE(
    slice_48_ends, AI_ARRAY_FORMAT_S16,
    slice_48_ends_data, slice_48_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_48_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_46_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_48_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_48_layer, 48,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_48_chain,
  NULL, &slice_48_layer, AI_STATIC, 
  .axes = &slice_48_axes, 
  .starts = &slice_48_starts, 
  .ends = &slice_48_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_49_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &serving_default_cache_b9_in0_output, &slice_48_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_49_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_49_layer, 49,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_49_chain,
  NULL, &concat_49_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_53_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &concat_49_output, &conv2d_52_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_53_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_53_layer, 53,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_53_chain,
  NULL, &eltwise_53_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_u8 slice_58_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_58_axes, AI_ARRAY_FORMAT_U8,
    slice_58_axes_data, slice_58_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_58_starts_data[] = { 6 };
AI_ARRAY_OBJ_DECLARE(
    slice_58_starts, AI_ARRAY_FORMAT_S16,
    slice_58_starts_data, slice_58_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_58_ends_data[] = { 48 };
AI_ARRAY_OBJ_DECLARE(
    slice_58_ends, AI_ARRAY_FORMAT_S16,
    slice_58_ends_data, slice_58_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_58_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_56_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_58_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_58_layer, 58,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_58_chain,
  NULL, &slice_58_layer, AI_STATIC, 
  .axes = &slice_58_axes, 
  .starts = &slice_58_starts, 
  .ends = &slice_58_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_59_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &serving_default_cache_b11_in0_output, &slice_58_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_59_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_59_layer, 59,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_59_chain,
  NULL, &concat_59_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_63_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &concat_59_output, &conv2d_62_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_63_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_63_layer, 63,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_63_chain,
  NULL, &eltwise_63_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_u8 slice_65_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_65_axes, AI_ARRAY_FORMAT_U8,
    slice_65_axes_data, slice_65_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_65_starts_data[] = { 6 };
AI_ARRAY_OBJ_DECLARE(
    slice_65_starts, AI_ARRAY_FORMAT_S16,
    slice_65_starts_data, slice_65_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_65_ends_data[] = { 48 };
AI_ARRAY_OBJ_DECLARE(
    slice_65_ends, AI_ARRAY_FORMAT_S16,
    slice_65_ends_data, slice_65_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_65_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_63_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_65_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_65_layer, 65,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_65_chain,
  NULL, &slice_65_layer, AI_STATIC, 
  .axes = &slice_65_axes, 
  .starts = &slice_65_starts, 
  .ends = &slice_65_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_66_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &serving_default_cache_b12_in0_output, &slice_65_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_66_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_66_layer, 66,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_66_chain,
  NULL, &concat_66_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_70_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &concat_66_output, &conv2d_69_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_70_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_70_layer, 70,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_70_chain,
  NULL, &eltwise_70_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_u8 slice_75_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_75_axes, AI_ARRAY_FORMAT_U8,
    slice_75_axes_data, slice_75_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_75_starts_data[] = { 10 };
AI_ARRAY_OBJ_DECLARE(
    slice_75_starts, AI_ARRAY_FORMAT_S16,
    slice_75_starts_data, slice_75_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_75_ends_data[] = { 80 };
AI_ARRAY_OBJ_DECLARE(
    slice_75_ends, AI_ARRAY_FORMAT_S16,
    slice_75_ends_data, slice_75_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_75_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_73_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_75_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_75_layer, 75,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_75_chain,
  NULL, &slice_75_layer, AI_STATIC, 
  .axes = &slice_75_axes, 
  .starts = &slice_75_starts, 
  .ends = &slice_75_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_76_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &serving_default_cache_b14_in0_output, &slice_75_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_76_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_76_layer, 76,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_76_chain,
  NULL, &concat_76_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_80_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &concat_76_output, &conv2d_79_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_80_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_80_layer, 80,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_80_chain,
  NULL, &eltwise_80_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)


AI_STATIC_CONST ai_u8 slice_82_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_82_axes, AI_ARRAY_FORMAT_U8,
    slice_82_axes_data, slice_82_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_82_starts_data[] = { 10 };
AI_ARRAY_OBJ_DECLARE(
    slice_82_starts, AI_ARRAY_FORMAT_S16,
    slice_82_starts_data, slice_82_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_82_ends_data[] = { 80 };
AI_ARRAY_OBJ_DECLARE(
    slice_82_ends, AI_ARRAY_FORMAT_S16,
    slice_82_ends_data, slice_82_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_82_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_80_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_82_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_82_layer, 82,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_82_chain,
  NULL, &slice_82_layer, AI_STATIC, 
  .axes = &slice_82_axes, 
  .starts = &slice_82_starts, 
  .ends = &slice_82_ends, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  concat_83_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &serving_default_cache_b15_in0_output, &slice_82_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &concat_83_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  concat_83_layer, 83,
  CONCAT_TYPE, 0x0, NULL,
  concat, forward_concat,
  &concat_83_chain,
  NULL, &concat_83_layer, AI_STATIC, 
  .axis = AI_SHAPE_CHANNEL, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  eltwise_87_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 2, &concat_83_output, &conv2d_86_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_87_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  eltwise_87_layer, 87,
  ELTWISE_INTEGER_TYPE, 0x0, NULL,
  eltwise_integer, forward_eltwise_integer_INT8,
  &eltwise_87_chain,
  NULL, &eltwise_87_layer, AI_STATIC, 
  .operation = ai_sum_f32, 
  .buffer_operation = ai_sum_buffer_INT8, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  pool_92_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_91_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &pool_92_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  pool_92_layer, 92,
  POOL_TYPE, 0x0, NULL,
  pool, forward_ap_integer_INT8,
  &pool_92_chain,
  NULL, &pool_92_layer, AI_STATIC, 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)


AI_STATIC_CONST ai_u8 slice_81_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_81_axes, AI_ARRAY_FORMAT_U8,
    slice_81_axes_data, slice_81_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_81_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_81_starts, AI_ARRAY_FORMAT_S16,
    slice_81_starts_data, slice_81_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_81_ends_data[] = { 10 };
AI_ARRAY_OBJ_DECLARE(
    slice_81_ends, AI_ARRAY_FORMAT_S16,
    slice_81_ends_data, slice_81_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_81_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_80_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_81_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_81_layer, 81,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_81_chain,
  NULL, &slice_81_layer, AI_STATIC, 
  .axes = &slice_81_axes, 
  .starts = &slice_81_starts, 
  .ends = &slice_81_ends, 
)


AI_STATIC_CONST ai_u8 slice_74_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_74_axes, AI_ARRAY_FORMAT_U8,
    slice_74_axes_data, slice_74_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_74_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_74_starts, AI_ARRAY_FORMAT_S16,
    slice_74_starts_data, slice_74_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_74_ends_data[] = { 10 };
AI_ARRAY_OBJ_DECLARE(
    slice_74_ends, AI_ARRAY_FORMAT_S16,
    slice_74_ends_data, slice_74_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_74_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_73_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_74_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_74_layer, 74,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_74_chain,
  NULL, &slice_74_layer, AI_STATIC, 
  .axes = &slice_74_axes, 
  .starts = &slice_74_starts, 
  .ends = &slice_74_ends, 
)


AI_STATIC_CONST ai_u8 slice_64_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_64_axes, AI_ARRAY_FORMAT_U8,
    slice_64_axes_data, slice_64_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_64_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_64_starts, AI_ARRAY_FORMAT_S16,
    slice_64_starts_data, slice_64_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_64_ends_data[] = { 6 };
AI_ARRAY_OBJ_DECLARE(
    slice_64_ends, AI_ARRAY_FORMAT_S16,
    slice_64_ends_data, slice_64_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_64_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_63_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_64_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_64_layer, 64,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_64_chain,
  NULL, &slice_64_layer, AI_STATIC, 
  .axes = &slice_64_axes, 
  .starts = &slice_64_starts, 
  .ends = &slice_64_ends, 
)


AI_STATIC_CONST ai_u8 slice_57_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_57_axes, AI_ARRAY_FORMAT_U8,
    slice_57_axes_data, slice_57_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_57_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_57_starts, AI_ARRAY_FORMAT_S16,
    slice_57_starts_data, slice_57_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_57_ends_data[] = { 6 };
AI_ARRAY_OBJ_DECLARE(
    slice_57_ends, AI_ARRAY_FORMAT_S16,
    slice_57_ends_data, slice_57_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_57_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_56_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_57_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_57_layer, 57,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_57_chain,
  NULL, &slice_57_layer, AI_STATIC, 
  .axes = &slice_57_axes, 
  .starts = &slice_57_starts, 
  .ends = &slice_57_ends, 
)


AI_STATIC_CONST ai_u8 slice_47_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_47_axes, AI_ARRAY_FORMAT_U8,
    slice_47_axes_data, slice_47_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_47_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_47_starts, AI_ARRAY_FORMAT_S16,
    slice_47_starts_data, slice_47_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_47_ends_data[] = { 4 };
AI_ARRAY_OBJ_DECLARE(
    slice_47_ends, AI_ARRAY_FORMAT_S16,
    slice_47_ends_data, slice_47_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_47_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_46_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_47_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_47_layer, 47,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_47_chain,
  NULL, &slice_47_layer, AI_STATIC, 
  .axes = &slice_47_axes, 
  .starts = &slice_47_starts, 
  .ends = &slice_47_ends, 
)


AI_STATIC_CONST ai_u8 slice_40_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_40_axes, AI_ARRAY_FORMAT_U8,
    slice_40_axes_data, slice_40_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_40_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_40_starts, AI_ARRAY_FORMAT_S16,
    slice_40_starts_data, slice_40_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_40_ends_data[] = { 4 };
AI_ARRAY_OBJ_DECLARE(
    slice_40_ends, AI_ARRAY_FORMAT_S16,
    slice_40_ends_data, slice_40_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_40_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_39_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_40_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_40_layer, 40,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_40_chain,
  NULL, &slice_40_layer, AI_STATIC, 
  .axes = &slice_40_axes, 
  .starts = &slice_40_starts, 
  .ends = &slice_40_ends, 
)


AI_STATIC_CONST ai_u8 slice_33_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_33_axes, AI_ARRAY_FORMAT_U8,
    slice_33_axes_data, slice_33_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_33_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_33_starts, AI_ARRAY_FORMAT_S16,
    slice_33_starts_data, slice_33_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_33_ends_data[] = { 4 };
AI_ARRAY_OBJ_DECLARE(
    slice_33_ends, AI_ARRAY_FORMAT_S16,
    slice_33_ends_data, slice_33_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_33_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_32_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_33_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_33_layer, 33,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_33_chain,
  NULL, &slice_33_layer, AI_STATIC, 
  .axes = &slice_33_axes, 
  .starts = &slice_33_starts, 
  .ends = &slice_33_ends, 
)


AI_STATIC_CONST ai_u8 slice_23_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_23_axes, AI_ARRAY_FORMAT_U8,
    slice_23_axes_data, slice_23_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_23_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_23_starts, AI_ARRAY_FORMAT_S16,
    slice_23_starts_data, slice_23_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_23_ends_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_23_ends, AI_ARRAY_FORMAT_S16,
    slice_23_ends_data, slice_23_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_23_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &eltwise_22_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_23_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_23_layer, 23,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_23_chain,
  NULL, &slice_23_layer, AI_STATIC, 
  .axes = &slice_23_axes, 
  .starts = &slice_23_starts, 
  .ends = &slice_23_ends, 
)


AI_STATIC_CONST ai_u8 slice_16_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_16_axes, AI_ARRAY_FORMAT_U8,
    slice_16_axes_data, slice_16_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_16_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_16_starts, AI_ARRAY_FORMAT_S16,
    slice_16_starts_data, slice_16_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_16_ends_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_16_ends, AI_ARRAY_FORMAT_S16,
    slice_16_ends_data, slice_16_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_15_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_16_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_16_layer, 16,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_16_chain,
  NULL, &slice_16_layer, AI_STATIC, 
  .axes = &slice_16_axes, 
  .starts = &slice_16_starts, 
  .ends = &slice_16_ends, 
)


AI_STATIC_CONST ai_u8 slice_6_axes_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_6_axes, AI_ARRAY_FORMAT_U8,
    slice_6_axes_data, slice_6_axes_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_6_starts_data[] = { 0 };
AI_ARRAY_OBJ_DECLARE(
    slice_6_starts, AI_ARRAY_FORMAT_S16,
    slice_6_starts_data, slice_6_starts_data, 1, AI_STATIC_CONST)

AI_STATIC_CONST ai_i16 slice_6_ends_data[] = { 2 };
AI_ARRAY_OBJ_DECLARE(
    slice_6_ends, AI_ARRAY_FORMAT_S16,
    slice_6_ends_data, slice_6_ends_data, 1, AI_STATIC_CONST)
AI_TENSOR_CHAIN_OBJ_DECLARE(
  slice_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &conv2d_5_output),
  AI_TENSOR_LIST_OBJ_INIT(AI_FLAG_NONE, 1, &slice_6_output),
  AI_TENSOR_LIST_OBJ_EMPTY,
  AI_TENSOR_LIST_OBJ_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  slice_6_layer, 6,
  SLICE_TYPE, 0x0, NULL,
  slice, forward_slice,
  &slice_6_chain,
  NULL, &slice_6_layer, AI_STATIC, 
  .axes = &slice_6_axes, 
  .starts = &slice_6_starts, 
  .ends = &slice_6_ends, 
)
/**  Hybrid layers declarations section  *************************************/
void forward_lite_slice_slice_7(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  conv2d_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 12640);
  conv2d_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 12640);
  slice_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  slice_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(7, 1, { conv2d_5_output.data->data});
  forward_slice(&slice_7_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(7, 1, { slice_7_output.data->data});
}
void forward_lite_concat_concat_8(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  serving_default_cache_b2_in0_output_array.data = AI_PTR(net_ctx->_inputs[1] + 0);
  serving_default_cache_b2_in0_output_array.data_start = AI_PTR(net_ctx->_inputs[1] + 0);
  slice_7_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  slice_7_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  concat_8_output_array.data = AI_PTR(net_ctx->_activations[0] + 3584);
  concat_8_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3584);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(8, 2, { serving_default_cache_b2_in0_output.data->data,slice_7_output.data->data});
  forward_concat(&concat_8_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(8, 1, { concat_8_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_12(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  concat_8_output_array.data = AI_PTR(net_ctx->_activations[0] + 3584);
  concat_8_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3584);
  conv2d_11_output_array.data = AI_PTR(net_ctx->_activations[0] + 7680);
  conv2d_11_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 7680);
  eltwise_12_output_array.data = AI_PTR(net_ctx->_activations[0] + 16736);
  eltwise_12_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 16736);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(12, 2, { concat_8_output.data->data,conv2d_11_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_12_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(12, 1, { eltwise_12_output.data->data});
}
void forward_lite_slice_slice_17(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  conv2d_15_output_array.data = AI_PTR(net_ctx->_activations[0] + 544);
  conv2d_15_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 544);
  slice_17_output_array.data = AI_PTR(net_ctx->_activations[0] + 1568);
  slice_17_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1568);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(17, 1, { conv2d_15_output.data->data});
  forward_slice(&slice_17_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(17, 1, { slice_17_output.data->data});
}
void forward_lite_concat_concat_18(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  serving_default_cache_b4_in0_output_array.data = AI_PTR(net_ctx->_inputs[2] + 0);
  serving_default_cache_b4_in0_output_array.data_start = AI_PTR(net_ctx->_inputs[2] + 0);
  slice_17_output_array.data = AI_PTR(net_ctx->_activations[0] + 1568);
  slice_17_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1568);
  concat_18_output_array.data = AI_PTR(net_ctx->_activations[0] + 2464);
  concat_18_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2464);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(18, 2, { serving_default_cache_b4_in0_output.data->data,slice_17_output.data->data});
  forward_concat(&concat_18_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(18, 1, { concat_18_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_22(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  concat_18_output_array.data = AI_PTR(net_ctx->_activations[0] + 2464);
  concat_18_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2464);
  conv2d_21_output_array.data = AI_PTR(net_ctx->_activations[0] + 3488);
  conv2d_21_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3488);
  eltwise_22_output_array.data = AI_PTR(net_ctx->_activations[0] + 4512);
  eltwise_22_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4512);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(22, 2, { concat_18_output.data->data,conv2d_21_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_22_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(22, 1, { eltwise_22_output.data->data});
}
void forward_lite_slice_slice_24(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  eltwise_22_output_array.data = AI_PTR(net_ctx->_activations[0] + 4512);
  eltwise_22_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4512);
  slice_24_output_array.data = AI_PTR(net_ctx->_activations[0] + 1568);
  slice_24_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1568);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(24, 1, { eltwise_22_output.data->data});
  forward_slice(&slice_24_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(24, 1, { slice_24_output.data->data});
}
void forward_lite_concat_concat_25(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  serving_default_cache_b5_in0_output_array.data = AI_PTR(net_ctx->_inputs[3] + 0);
  serving_default_cache_b5_in0_output_array.data_start = AI_PTR(net_ctx->_inputs[3] + 0);
  slice_24_output_array.data = AI_PTR(net_ctx->_activations[0] + 1568);
  slice_24_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1568);
  concat_25_output_array.data = AI_PTR(net_ctx->_activations[0] + 2464);
  concat_25_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2464);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(25, 2, { serving_default_cache_b5_in0_output.data->data,slice_24_output.data->data});
  forward_concat(&concat_25_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(25, 1, { concat_25_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_29(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  concat_25_output_array.data = AI_PTR(net_ctx->_activations[0] + 2464);
  concat_25_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2464);
  conv2d_28_output_array.data = AI_PTR(net_ctx->_activations[0] + 3488);
  conv2d_28_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3488);
  eltwise_29_output_array.data = AI_PTR(net_ctx->_activations[0] + 5536);
  eltwise_29_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 5536);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(29, 2, { concat_25_output.data->data,conv2d_28_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_29_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(29, 1, { eltwise_29_output.data->data});
}
void forward_lite_slice_slice_34(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  conv2d_32_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_32_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  slice_34_output_array.data = AI_PTR(net_ctx->_activations[0] + 1568);
  slice_34_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1568);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(34, 1, { conv2d_32_output.data->data});
  forward_slice(&slice_34_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(34, 1, { slice_34_output.data->data});
}
void forward_lite_concat_concat_35(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  serving_default_cache_b7_in0_output_array.data = AI_PTR(net_ctx->_inputs[4] + 0);
  serving_default_cache_b7_in0_output_array.data_start = AI_PTR(net_ctx->_inputs[4] + 0);
  slice_34_output_array.data = AI_PTR(net_ctx->_activations[0] + 1568);
  slice_34_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1568);
  concat_35_output_array.data = AI_PTR(net_ctx->_activations[0] + 2016);
  concat_35_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2016);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(35, 2, { serving_default_cache_b7_in0_output.data->data,slice_34_output.data->data});
  forward_concat(&concat_35_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(35, 1, { concat_35_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_39(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  concat_35_output_array.data = AI_PTR(net_ctx->_activations[0] + 2016);
  concat_35_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2016);
  conv2d_38_output_array.data = AI_PTR(net_ctx->_activations[0] + 3616);
  conv2d_38_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3616);
  eltwise_39_output_array.data = AI_PTR(net_ctx->_activations[0] + 2528);
  eltwise_39_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2528);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(39, 2, { concat_35_output.data->data,conv2d_38_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_39_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(39, 1, { eltwise_39_output.data->data});
}
void forward_lite_slice_slice_41(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  eltwise_39_output_array.data = AI_PTR(net_ctx->_activations[0] + 2528);
  eltwise_39_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2528);
  slice_41_output_array.data = AI_PTR(net_ctx->_activations[0] + 1568);
  slice_41_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1568);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(41, 1, { eltwise_39_output.data->data});
  forward_slice(&slice_41_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(41, 1, { slice_41_output.data->data});
}
void forward_lite_concat_concat_42(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  serving_default_cache_b8_in0_output_array.data = AI_PTR(net_ctx->_inputs[5] + 0);
  serving_default_cache_b8_in0_output_array.data_start = AI_PTR(net_ctx->_inputs[5] + 0);
  slice_41_output_array.data = AI_PTR(net_ctx->_activations[0] + 1568);
  slice_41_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1568);
  concat_42_output_array.data = AI_PTR(net_ctx->_activations[0] + 2016);
  concat_42_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2016);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(42, 2, { serving_default_cache_b8_in0_output.data->data,slice_41_output.data->data});
  forward_concat(&concat_42_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(42, 1, { concat_42_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_46(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  concat_42_output_array.data = AI_PTR(net_ctx->_activations[0] + 2016);
  concat_42_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2016);
  conv2d_45_output_array.data = AI_PTR(net_ctx->_activations[0] + 8608);
  conv2d_45_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 8608);
  eltwise_46_output_array.data = AI_PTR(net_ctx->_activations[0] + 3040);
  eltwise_46_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3040);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(46, 2, { concat_42_output.data->data,conv2d_45_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_46_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(46, 1, { eltwise_46_output.data->data});
}
void forward_lite_slice_slice_48(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  eltwise_46_output_array.data = AI_PTR(net_ctx->_activations[0] + 3040);
  eltwise_46_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3040);
  slice_48_output_array.data = AI_PTR(net_ctx->_activations[0] + 1568);
  slice_48_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1568);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(48, 1, { eltwise_46_output.data->data});
  forward_slice(&slice_48_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(48, 1, { slice_48_output.data->data});
}
void forward_lite_concat_concat_49(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  serving_default_cache_b9_in0_output_array.data = AI_PTR(net_ctx->_inputs[6] + 0);
  serving_default_cache_b9_in0_output_array.data_start = AI_PTR(net_ctx->_inputs[6] + 0);
  slice_48_output_array.data = AI_PTR(net_ctx->_activations[0] + 1568);
  slice_48_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1568);
  concat_49_output_array.data = AI_PTR(net_ctx->_activations[0] + 2016);
  concat_49_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2016);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(49, 2, { serving_default_cache_b9_in0_output.data->data,slice_48_output.data->data});
  forward_concat(&concat_49_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(49, 1, { concat_49_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_53(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  concat_49_output_array.data = AI_PTR(net_ctx->_activations[0] + 2016);
  concat_49_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2016);
  conv2d_52_output_array.data = AI_PTR(net_ctx->_activations[0] + 3552);
  conv2d_52_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3552);
  eltwise_53_output_array.data = AI_PTR(net_ctx->_activations[0] + 5536);
  eltwise_53_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 5536);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(53, 2, { concat_49_output.data->data,conv2d_52_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_53_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(53, 1, { eltwise_53_output.data->data});
}
void forward_lite_slice_slice_58(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  conv2d_56_output_array.data = AI_PTR(net_ctx->_activations[0] + 1568);
  conv2d_56_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1568);
  slice_58_output_array.data = AI_PTR(net_ctx->_activations[0] + 3552);
  slice_58_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3552);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(58, 1, { conv2d_56_output.data->data});
  forward_slice(&slice_58_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(58, 1, { slice_58_output.data->data});
}
void forward_lite_concat_concat_59(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  serving_default_cache_b11_in0_output_array.data = AI_PTR(net_ctx->_inputs[7] + 0);
  serving_default_cache_b11_in0_output_array.data_start = AI_PTR(net_ctx->_inputs[7] + 0);
  slice_58_output_array.data = AI_PTR(net_ctx->_activations[0] + 3552);
  slice_58_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3552);
  concat_59_output_array.data = AI_PTR(net_ctx->_activations[0] + 5536);
  concat_59_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 5536);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(59, 2, { serving_default_cache_b11_in0_output.data->data,slice_58_output.data->data});
  forward_concat(&concat_59_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(59, 1, { concat_59_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_63(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  concat_59_output_array.data = AI_PTR(net_ctx->_activations[0] + 5536);
  concat_59_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 5536);
  conv2d_62_output_array.data = AI_PTR(net_ctx->_activations[0] + 3552);
  conv2d_62_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3552);
  eltwise_63_output_array.data = AI_PTR(net_ctx->_activations[0] + 6304);
  eltwise_63_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 6304);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(63, 2, { concat_59_output.data->data,conv2d_62_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_63_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(63, 1, { eltwise_63_output.data->data});
}
void forward_lite_slice_slice_65(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  eltwise_63_output_array.data = AI_PTR(net_ctx->_activations[0] + 6304);
  eltwise_63_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 6304);
  slice_65_output_array.data = AI_PTR(net_ctx->_activations[0] + 3552);
  slice_65_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3552);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(65, 1, { eltwise_63_output.data->data});
  forward_slice(&slice_65_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(65, 1, { slice_65_output.data->data});
}
void forward_lite_concat_concat_66(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  serving_default_cache_b12_in0_output_array.data = AI_PTR(net_ctx->_inputs[8] + 0);
  serving_default_cache_b12_in0_output_array.data_start = AI_PTR(net_ctx->_inputs[8] + 0);
  slice_65_output_array.data = AI_PTR(net_ctx->_activations[0] + 3552);
  slice_65_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3552);
  concat_66_output_array.data = AI_PTR(net_ctx->_activations[0] + 5536);
  concat_66_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 5536);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(66, 2, { serving_default_cache_b12_in0_output.data->data,slice_65_output.data->data});
  forward_concat(&concat_66_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(66, 1, { concat_66_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_70(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  concat_66_output_array.data = AI_PTR(net_ctx->_activations[0] + 5536);
  concat_66_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 5536);
  conv2d_69_output_array.data = AI_PTR(net_ctx->_activations[0] + 3552);
  conv2d_69_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3552);
  eltwise_70_output_array.data = AI_PTR(net_ctx->_activations[0] + 7072);
  eltwise_70_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 7072);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(70, 2, { concat_66_output.data->data,conv2d_69_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_70_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(70, 1, { eltwise_70_output.data->data});
}
void forward_lite_slice_slice_75(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  conv2d_73_output_array.data = AI_PTR(net_ctx->_activations[0] + 3552);
  conv2d_73_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3552);
  slice_75_output_array.data = AI_PTR(net_ctx->_activations[0] + 3872);
  slice_75_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3872);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(75, 1, { conv2d_73_output.data->data});
  forward_slice(&slice_75_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(75, 1, { slice_75_output.data->data});
}
void forward_lite_concat_concat_76(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  serving_default_cache_b14_in0_output_array.data = AI_PTR(net_ctx->_inputs[9] + 0);
  serving_default_cache_b14_in0_output_array.data_start = AI_PTR(net_ctx->_inputs[9] + 0);
  slice_75_output_array.data = AI_PTR(net_ctx->_activations[0] + 3872);
  slice_75_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3872);
  concat_76_output_array.data = AI_PTR(net_ctx->_activations[0] + 4152);
  concat_76_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4152);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(76, 2, { serving_default_cache_b14_in0_output.data->data,slice_75_output.data->data});
  forward_concat(&concat_76_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(76, 1, { concat_76_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_80(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  concat_76_output_array.data = AI_PTR(net_ctx->_activations[0] + 4152);
  concat_76_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4152);
  conv2d_79_output_array.data = AI_PTR(net_ctx->_activations[0] + 5536);
  conv2d_79_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 5536);
  eltwise_80_output_array.data = AI_PTR(net_ctx->_activations[0] + 5856);
  eltwise_80_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 5856);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(80, 2, { concat_76_output.data->data,conv2d_79_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_80_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(80, 1, { eltwise_80_output.data->data});
}
void forward_lite_slice_slice_82(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  eltwise_80_output_array.data = AI_PTR(net_ctx->_activations[0] + 5856);
  eltwise_80_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 5856);
  slice_82_output_array.data = AI_PTR(net_ctx->_activations[0] + 3872);
  slice_82_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3872);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(82, 1, { eltwise_80_output.data->data});
  forward_slice(&slice_82_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(82, 1, { slice_82_output.data->data});
}
void forward_lite_concat_concat_83(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  serving_default_cache_b15_in0_output_array.data = AI_PTR(net_ctx->_inputs[10] + 0);
  serving_default_cache_b15_in0_output_array.data_start = AI_PTR(net_ctx->_inputs[10] + 0);
  slice_82_output_array.data = AI_PTR(net_ctx->_activations[0] + 3872);
  slice_82_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3872);
  concat_83_output_array.data = AI_PTR(net_ctx->_activations[0] + 4152);
  concat_83_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4152);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(83, 2, { serving_default_cache_b15_in0_output.data->data,slice_82_output.data->data});
  forward_concat(&concat_83_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(83, 1, { concat_83_output.data->data});
}
void forward_lite_eltwise_integer_INT8_eltwise_87(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  concat_83_output_array.data = AI_PTR(net_ctx->_activations[0] + 4152);
  concat_83_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4152);
  conv2d_86_output_array.data = AI_PTR(net_ctx->_activations[0] + 5536);
  conv2d_86_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 5536);
  eltwise_87_output_array.data = AI_PTR(net_ctx->_activations[0] + 7072);
  eltwise_87_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 7072);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(87, 2, { concat_83_output.data->data,conv2d_86_output.data->data});
  forward_eltwise_integer_INT8(&eltwise_87_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(87, 1, { eltwise_87_output.data->data});
}
void forward_lite_ap_integer_INT8_pool_92(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  conv2d_91_output_array.data = AI_PTR(net_ctx->_activations[0] + 7072);
  conv2d_91_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 7072);
  pool_92_output_array.data = AI_PTR(net_ctx->_activations[0] + 16736);
  pool_92_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 16736);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(92, 1, { conv2d_91_output.data->data});
  forward_ap_integer_INT8(&pool_92_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(92, 1, { pool_92_output.data->data});
}
void forward_lite_slice_slice_81(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  eltwise_80_output_array.data = AI_PTR(net_ctx->_activations[0] + 5856);
  eltwise_80_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 5856);
  slice_81_output_array.data = AI_PTR(net_ctx->_outputs[9] + 0);
  slice_81_output_array.data_start = AI_PTR(net_ctx->_outputs[9] + 0);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(81, 1, { eltwise_80_output.data->data});
  forward_slice(&slice_81_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(81, 1, { slice_81_output.data->data});
}
void forward_lite_slice_slice_74(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  conv2d_73_output_array.data = AI_PTR(net_ctx->_activations[0] + 3552);
  conv2d_73_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3552);
  slice_74_output_array.data = AI_PTR(net_ctx->_outputs[0] + 0);
  slice_74_output_array.data_start = AI_PTR(net_ctx->_outputs[0] + 0);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(74, 1, { conv2d_73_output.data->data});
  forward_slice(&slice_74_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(74, 1, { slice_74_output.data->data});
}
void forward_lite_slice_slice_64(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  eltwise_63_output_array.data = AI_PTR(net_ctx->_activations[0] + 6304);
  eltwise_63_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 6304);
  slice_64_output_array.data = AI_PTR(net_ctx->_outputs[8] + 0);
  slice_64_output_array.data_start = AI_PTR(net_ctx->_outputs[8] + 0);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(64, 1, { eltwise_63_output.data->data});
  forward_slice(&slice_64_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(64, 1, { slice_64_output.data->data});
}
void forward_lite_slice_slice_57(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  conv2d_56_output_array.data = AI_PTR(net_ctx->_activations[0] + 1568);
  conv2d_56_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 1568);
  slice_57_output_array.data = AI_PTR(net_ctx->_outputs[3] + 0);
  slice_57_output_array.data_start = AI_PTR(net_ctx->_outputs[3] + 0);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(57, 1, { conv2d_56_output.data->data});
  forward_slice(&slice_57_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(57, 1, { slice_57_output.data->data});
}
void forward_lite_slice_slice_47(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  eltwise_46_output_array.data = AI_PTR(net_ctx->_activations[0] + 3040);
  eltwise_46_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 3040);
  slice_47_output_array.data = AI_PTR(net_ctx->_outputs[10] + 0);
  slice_47_output_array.data_start = AI_PTR(net_ctx->_outputs[10] + 0);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(47, 1, { eltwise_46_output.data->data});
  forward_slice(&slice_47_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(47, 1, { slice_47_output.data->data});
}
void forward_lite_slice_slice_40(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  eltwise_39_output_array.data = AI_PTR(net_ctx->_activations[0] + 2528);
  eltwise_39_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 2528);
  slice_40_output_array.data = AI_PTR(net_ctx->_outputs[4] + 0);
  slice_40_output_array.data_start = AI_PTR(net_ctx->_outputs[4] + 0);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(40, 1, { eltwise_39_output.data->data});
  forward_slice(&slice_40_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(40, 1, { slice_40_output.data->data});
}
void forward_lite_slice_slice_33(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  conv2d_32_output_array.data = AI_PTR(net_ctx->_activations[0] + 0);
  conv2d_32_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 0);
  slice_33_output_array.data = AI_PTR(net_ctx->_outputs[1] + 0);
  slice_33_output_array.data_start = AI_PTR(net_ctx->_outputs[1] + 0);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(33, 1, { conv2d_32_output.data->data});
  forward_slice(&slice_33_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(33, 1, { slice_33_output.data->data});
}
void forward_lite_slice_slice_23(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  eltwise_22_output_array.data = AI_PTR(net_ctx->_activations[0] + 4512);
  eltwise_22_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 4512);
  slice_23_output_array.data = AI_PTR(net_ctx->_outputs[5] + 0);
  slice_23_output_array.data_start = AI_PTR(net_ctx->_outputs[5] + 0);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(23, 1, { eltwise_22_output.data->data});
  forward_slice(&slice_23_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(23, 1, { slice_23_output.data->data});
}
void forward_lite_slice_slice_16(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  conv2d_15_output_array.data = AI_PTR(net_ctx->_activations[0] + 544);
  conv2d_15_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 544);
  slice_16_output_array.data = AI_PTR(net_ctx->_outputs[7] + 0);
  slice_16_output_array.data_start = AI_PTR(net_ctx->_outputs[7] + 0);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(16, 1, { conv2d_15_output.data->data});
  forward_slice(&slice_16_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(16, 1, { slice_16_output.data->data});
}
void forward_lite_slice_slice_6(_stai_student_c1j_ptq_int8_context* net_ctx)
{
  conv2d_5_output_array.data = AI_PTR(net_ctx->_activations[0] + 12640);
  conv2d_5_output_array.data_start = AI_PTR(net_ctx->_activations[0] + 12640);
  slice_6_output_array.data = AI_PTR(net_ctx->_outputs[2] + 0);
  slice_6_output_array.data_start = AI_PTR(net_ctx->_outputs[2] + 0);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(6, 1, { conv2d_5_output.data->data});
  forward_slice(&slice_6_layer);
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(6, 1, { slice_6_output.data->data});
}

/*****************************************************************************/


static const ai_u16 conv2d_0_t_in_0_shape_w_const_u16 = 64;
static const ai_u16 conv2d_0_t_in_0_shape_h_const_u16 = 64;
static const ai_u16 conv2d_0_t_in_0_shape_ch_const_u16 = 1;
static const ai_u16 conv2d_0_t_out_0_shape_ch_const_u16 = 16;
static const ai_u16 conv2d_0_t_weight_0_shape_w_const_u16 = 3;
static const ai_u16 conv2d_0_t_weight_0_shape_h_const_u16 = 3;
static const ai_u16 conv2d_0_l_stride_1_const_u16 = 2;
static const ai_u16 conv2d_0_l_stride_0_const_u16 = 2;
static const ai_i32 conv2d_0_l_pad_W_0_const_s32 = 0;
static const ai_i32 conv2d_0_l_pad_H_0_const_s32 = 0;
static const ai_i8 conv2d_0_t_in_0_fmt_zero_const_s8 = -1;
static const ai_i8 conv2d_0_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_0_t_in_0_fmt_scale_const_f32 = 0.007843137718737125f;
static const ai_float conv2d_0_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_0_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.006761576049029827f, 0.011534281075000763f, 0.0034789713099598885f, 0.012590471655130386f, 0.010440178215503693f, 0.016896123066544533f, 0.006261862814426422f, 0.03312729299068451f, 0.03538038209080696f, 0.0069756824523210526f, 0.012190569192171097f, 0.026535727083683014f, 0.004900740459561348f, 0.02631262317299843f, 0.011309395544230938f, 0.004628679249435663f);
static const ai_layer_format_type conv2d_0_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;
static const ai_u16 conv2d_0_t_out_0_shape_w_const_u16 = 32;
static const ai_u16 conv2d_0_t_out_0_shape_h_const_u16 = 32;

static const ai_i8 conv2d_1_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_1_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_1_pad_before_t_in_0_shape_h_const_u32 = 32;

static const ai_u16 conv2d_1_t_in_0_shape_w_const_u16 = 34;
static const ai_u16 conv2d_1_t_in_0_shape_h_const_u16 = 34;
static const ai_u16 conv2d_1_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 conv2d_1_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_1_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_1_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_1_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_1_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_1_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_1_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.02393142133951187f, 0.023615919053554535f, 0.011949454434216022f, 0.014425625093281269f, 0.010155863128602505f, 0.011748173274099827f, 0.012809250503778458f, 0.01541885081678629f, 0.004166842438280582f, 0.014952543191611767f, 0.00834878534078598f, 0.016843918710947037f, 0.03124147653579712f, 0.006655149627476931f, 0.009947271086275578f, 0.01266010943800211f);
static const ai_u16 conv2d_1_t_out_0_shape_w_const_u16 = 32;
static const ai_u16 conv2d_1_t_out_0_shape_h_const_u16 = 32;

static const ai_u16 conv2d_2_t_in_0_shape_w_const_u16 = 32;
static const ai_u16 conv2d_2_t_in_0_shape_h_const_u16 = 32;
static const ai_u16 conv2d_2_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_2_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_2_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 conv2d_2_t_out_0_shape_ch_const_u16 = 8;
static const ai_i8 conv2d_2_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_2_t_out_0_fmt_zero_const_s8 = -11;
static const ai_float conv2d_2_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_2_t_out_0_fmt_scale_const_f32 = 0.0713423639535904f;
static const ai_float conv2d_2_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00466803228482604f, 0.005065049044787884f, 0.004528655670583248f, 0.005709264427423477f, 0.006044481415301561f, 0.0056885345838963985f, 0.00772236380726099f, 0.0061760940589010715f);
static const ai_layer_format_type conv2d_2_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 conv2d_3_t_in_0_shape_w_const_u16 = 32;
static const ai_u16 conv2d_3_t_in_0_shape_h_const_u16 = 32;
static const ai_u16 conv2d_3_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_3_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_3_t_in_0_shape_ch_const_u16 = 8;
static const ai_u16 conv2d_3_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 conv2d_3_t_in_0_fmt_zero_const_s8 = -11;
static const ai_i8 conv2d_3_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_3_t_in_0_fmt_scale_const_f32 = 0.0713423639535904f;
static const ai_float conv2d_3_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_3_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0026994217187166214f, 0.002357573015615344f, 0.005911525338888168f, 0.00464578066021204f, 0.003344458295032382f, 0.0021761488169431686f, 0.0018057781271636486f, 0.0016574820037931204f, 0.0027176167350262403f, 0.0034158029593527317f, 0.0017258485313504934f, 0.003017151728272438f, 0.005267251282930374f, 0.001365272793918848f, 0.00429965415969491f, 0.006892150733619928f, 0.004525985103100538f, 0.005419317167252302f, 0.002469284925609827f, 0.004229744430631399f, 0.0018307397840544581f, 0.0029129385948181152f, 0.008133824914693832f, 0.003057811176404357f, 0.005176371894776821f, 0.005697317887097597f, 0.0032444617245346308f, 0.0025200245436280966f, 0.002997103612869978f, 0.002039529848843813f, 0.004789944738149643f, 0.003919507376849651f, 0.005281395744532347f, 0.0050551751628518105f, 0.006213285028934479f, 0.0016061156056821346f, 0.00491518247872591f, 0.004811898339539766f, 0.005805822089314461f, 0.0031995170284062624f, 0.006625858601182699f, 0.003594364272430539f, 0.007864805869758129f, 0.007945729419589043f, 0.0030829415190964937f, 0.003805505344644189f, 0.0022400813177227974f, 0.0038787960074841976f);
static const ai_layer_format_type conv2d_3_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_4_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_4_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_4_pad_before_t_in_0_shape_h_const_u32 = 32;

static const ai_u16 conv2d_4_t_in_0_shape_w_const_u16 = 34;
static const ai_u16 conv2d_4_t_in_0_shape_h_const_u16 = 34;
static const ai_u16 conv2d_4_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 conv2d_4_l_stride_1_const_u16 = 2;
static const ai_u16 conv2d_4_l_stride_0_const_u16 = 2;
static const ai_i8 conv2d_4_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_4_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_4_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_4_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_4_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.008903607726097107f, 0.010072068311274052f, 0.004511274863034487f, 0.00448981486260891f, 0.008067346177995205f, 0.015822000801563263f, 0.01089331042021513f, 0.005292527377605438f, 0.006579529494047165f, 0.005658882670104504f, 0.009270559065043926f, 0.009377826005220413f, 0.0035270564258098602f, 0.021103350445628166f, 0.007706661242991686f, 0.006114266812801361f, 0.004881192464381456f, 0.004355785436928272f, 0.006030843127518892f, 0.0053071919828653336f, 0.006483231205493212f, 0.006908502895385027f, 0.003720939625054598f, 0.010929186828434467f, 0.0070812455378472805f, 0.004708468448370695f, 0.01724868267774582f, 0.009059811942279339f, 0.006641875021159649f, 0.013383861631155014f, 0.0052848388440907f, 0.007842939347028732f, 0.0047970921732485294f, 0.0045748502016067505f, 0.004466010257601738f, 0.008400150574743748f, 0.005734215024858713f, 0.004592692945152521f, 0.004136538133025169f, 0.00930397491902113f, 0.005868073552846909f, 0.010604599490761757f, 0.004039953928440809f, 0.0024608478415757418f, 0.003655484179034829f, 0.016855178400874138f, 0.01557059958577156f, 0.004734071437269449f);
static const ai_u16 conv2d_4_t_out_0_shape_w_const_u16 = 16;
static const ai_u16 conv2d_4_t_out_0_shape_h_const_u16 = 16;

static const ai_u16 conv2d_5_t_in_0_shape_w_const_u16 = 16;
static const ai_u16 conv2d_5_t_in_0_shape_h_const_u16 = 16;
static const ai_u16 conv2d_5_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_5_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_5_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 conv2d_5_t_out_0_shape_ch_const_u16 = 16;
static const ai_i8 conv2d_5_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_5_t_out_0_fmt_zero_const_s8 = 4;
static const ai_float conv2d_5_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_5_t_out_0_fmt_scale_const_f32 = 0.04220737889409065f;
static const ai_float conv2d_5_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.002326929708942771f, 0.0025676870718598366f, 0.002196359680965543f, 0.0021793427877128124f, 0.003337613306939602f, 0.0033214690629392862f, 0.0023192616645246744f, 0.0023791203275322914f, 0.0029041713569313288f, 0.001845446415245533f, 0.002079655881971121f, 0.0022723907604813576f, 0.0020572314970195293f, 0.002897164085879922f, 0.0024037195835262537f, 0.0021369631867855787f);
static const ai_layer_format_type conv2d_5_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;



static const ai_u16 conv2d_9_t_in_0_shape_w_const_u16 = 16;
static const ai_u16 conv2d_9_t_in_0_shape_h_const_u16 = 16;
static const ai_u16 conv2d_9_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_9_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_9_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 conv2d_9_t_out_0_shape_ch_const_u16 = 96;
static const ai_i8 conv2d_9_t_in_0_fmt_zero_const_s8 = 4;
static const ai_i8 conv2d_9_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_9_t_in_0_fmt_scale_const_f32 = 0.04220737889409065f;
static const ai_float conv2d_9_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_9_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0037262672558426857f, 0.005465312395244837f, 0.0038577388040721416f, 0.006783691234886646f, 0.003986210562288761f, 0.0030176397413015366f, 0.003150637960061431f, 0.0057654851116240025f, 0.0029619550332427025f, 0.006278947461396456f, 0.0034593904856592417f, 0.0029281461611390114f, 0.0015973950503394008f, 0.006988436449319124f, 0.004138481803238392f, 0.008795949630439281f, 0.00501266960054636f, 0.003249179804697633f, 0.0026132618077099323f, 0.007762981113046408f, 0.006854515988379717f, 0.0021404502913355827f, 0.009083086624741554f, 0.004987961612641811f, 0.00201911642216146f, 0.0013356378767639399f, 0.004406921099871397f, 0.003939616959542036f, 0.003663616022095084f, 0.003095762338489294f, 0.005324997939169407f, 0.004523258190602064f, 0.0043030837550759315f, 0.004493299406021833f, 0.005296383984386921f, 0.0036613743286579847f, 0.004887047223746777f, 0.0017636000411584973f, 0.005635005887597799f, 0.0034840721637010574f, 0.0030622505582869053f, 0.01705143414437771f, 0.007284584455192089f, 0.0057813022285699844f, 0.009527000598609447f, 0.0033417444210499525f, 0.006503538694232702f, 0.0045664445497095585f, 0.0033962770830839872f, 0.007011099718511105f, 0.00509521784260869f, 0.00243961694650352f, 0.0011794567108154297f, 0.002825810108333826f, 0.024097107350826263f, 0.0021930444054305553f, 0.006658207159489393f, 0.0033622621558606625f, 0.009999791160225868f, 0.003538824850693345f, 0.004259511362761259f, 0.0049296231009066105f, 0.003282167250290513f, 0.004665100481361151f, 0.014662585221230984f, 0.002879090839996934f, 0.006650437135249376f, 0.006174186710268259f, 0.004912247881293297f, 0.005864779464900494f, 0.003521229838952422f, 0.003556156298145652f, 0.006784157361835241f, 0.004884884227067232f, 0.0032227281481027603f, 0.0033549678046256304f, 0.004053650423884392f, 0.002716476796194911f, 0.0051727634854614735f, 0.0015706210397183895f, 0.004010259173810482f, 0.020894968882203102f, 0.004220927599817514f, 0.0030374692287296057f, 0.004204532131552696f, 0.004413316026329994f, 0.0027058918494731188f, 0.005146383307874203f, 0.0023021951783448458f, 0.021264880895614624f, 0.007321569137275219f, 0.003674554405733943f, 0.009990268386900425f, 0.00414187740534544f, 0.0030868013855069876f, 0.0030189489480108023f);
static const ai_layer_format_type conv2d_9_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_10_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_10_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_10_pad_before_t_in_0_shape_h_const_u32 = 16;

static const ai_u16 conv2d_10_t_in_0_shape_w_const_u16 = 18;
static const ai_u16 conv2d_10_t_in_0_shape_h_const_u16 = 18;
static const ai_u16 conv2d_10_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_10_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_10_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_10_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_10_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_10_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_10_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_10_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.007470021024346352f, 0.00850577000528574f, 0.011766756884753704f, 0.014793981797993183f, 0.010586516000330448f, 0.009996812790632248f, 0.011803572997450829f, 0.007344729732722044f, 0.01658320240676403f, 0.010276158340275288f, 0.011645253747701645f, 0.010162289254367352f, 0.01473055500537157f, 0.006492977030575275f, 0.013581700623035431f, 0.011624625883996487f, 0.009567626751959324f, 0.007729408331215382f, 0.008529423736035824f, 0.011492319405078888f, 0.0073955850675702095f, 0.02217751182615757f, 0.008590498007833958f, 0.006278206128627062f, 0.014931970275938511f, 0.013198691420257092f, 0.007647414691746235f, 0.028191788122057915f, 0.008249654434621334f, 0.01355807390064001f, 0.0052464986220002174f, 0.0038017628248780966f, 0.013959084637463093f, 0.005174336023628712f, 0.010271443985402584f, 0.008231929503381252f, 0.013431400060653687f, 0.02009948156774044f, 0.005257603246718645f, 0.010760621167719364f, 0.010681447573006153f, 0.007015405688434839f, 0.013479233719408512f, 0.013091979548335075f, 0.010303701274096966f, 0.011151694692671299f, 0.008558020927011967f, 0.005923907738178968f, 0.007427167613059282f, 0.01224743202328682f, 0.011490067467093468f, 0.011595332063734531f, 0.016807064414024353f, 0.01407240703701973f, 0.00880863331258297f, 0.03958674147725105f, 0.010052226483821869f, 0.004070460796356201f, 0.008742542937397957f, 0.005917679984122515f, 0.005235467106103897f, 0.012037165462970734f, 0.01306435838341713f, 0.00897838082164526f, 0.012148830108344555f, 0.018733177334070206f, 0.013453537598252296f, 0.013717493042349815f, 0.01020499225705862f, 0.0042845215648412704f, 0.01232712622731924f, 0.0039504519663751125f, 0.01485474407672882f, 0.011803762055933475f, 0.011970012448728085f, 0.01342841424047947f, 0.0201621912419796f, 0.008445958606898785f, 0.010111112147569656f, 0.014568500220775604f, 0.009762848727405071f, 0.011689509265124798f, 0.006586555391550064f, 0.009408609941601753f, 0.010101496241986752f, 0.007488824427127838f, 0.009459909051656723f, 0.011094111949205399f, 0.014154830016195774f, 0.005753270350396633f, 0.006382897961884737f, 0.007976125925779343f, 0.009135502390563488f, 0.0218387208878994f, 0.00996449775993824f, 0.007260718382894993f);
static const ai_u16 conv2d_10_t_out_0_shape_w_const_u16 = 16;
static const ai_u16 conv2d_10_t_out_0_shape_h_const_u16 = 16;

static const ai_u16 conv2d_11_t_in_0_shape_w_const_u16 = 16;
static const ai_u16 conv2d_11_t_in_0_shape_h_const_u16 = 16;
static const ai_u16 conv2d_11_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_11_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_11_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_11_t_out_0_shape_ch_const_u16 = 16;
static const ai_i8 conv2d_11_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_11_t_out_0_fmt_zero_const_s8 = 17;
static const ai_float conv2d_11_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_11_t_out_0_fmt_scale_const_f32 = 0.07701683789491653f;
static const ai_float conv2d_11_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0033553221728652716f, 0.003155861981213093f, 0.002720209304243326f, 0.002080869162455201f, 0.002797222463414073f, 0.0025472568813711405f, 0.0033971669618040323f, 0.002537075662985444f, 0.0021416530944406986f, 0.0034063481725752354f, 0.0025386156048625708f, 0.003494824282824993f, 0.003351657884195447f, 0.00428667850792408f, 0.0024355086497962475f, 0.002294079400599003f);
static const ai_layer_format_type conv2d_11_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_13_t_in_0_shape_w_const_u16 = 16;
static const ai_u16 conv2d_13_t_in_0_shape_h_const_u16 = 16;
static const ai_u16 conv2d_13_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_13_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_13_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 conv2d_13_t_out_0_shape_ch_const_u16 = 96;
static const ai_i8 conv2d_13_t_in_0_fmt_zero_const_s8 = 17;
static const ai_i8 conv2d_13_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_13_t_in_0_fmt_scale_const_f32 = 0.07701683789491653f;
static const ai_float conv2d_13_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_13_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0008399361395277083f, 0.0020909947343170643f, 0.0015335297212004662f, 0.0010974111501127481f, 0.0020865690894424915f, 0.0013860457111150026f, 0.002419940894469619f, 0.003437979379668832f, 0.0023497389629483223f, 0.0024391792248934507f, 0.0033478836994618177f, 0.002436394104734063f, 0.004701570142060518f, 0.0011553820222616196f, 0.0011489586904644966f, 0.0022556043695658445f, 0.0017409808933734894f, 0.0015197799075394869f, 0.0060894605703651905f, 0.001470665680244565f, 0.0020603144075721502f, 0.0019366176566109061f, 0.0019285643938928843f, 0.001564537757076323f, 0.0017176568508148193f, 0.00350624299608171f, 0.0023312976118177176f, 0.001774378353729844f, 0.004163979087024927f, 0.0029193891678005457f, 0.0012756792129948735f, 0.002195863053202629f, 0.003075946355238557f, 0.0017680081073194742f, 0.0020498642697930336f, 0.0022513712756335735f, 0.0031043672934174538f, 0.002208999590948224f, 0.0018442089203745127f, 0.0025437360163778067f, 0.0016056153690442443f, 0.001232267008163035f, 0.000696364848408848f, 0.0023515592329204082f, 0.0022665380965918303f, 0.002010864205658436f, 0.0012867862824350595f, 0.0018832564819604158f, 0.002537492895498872f, 0.001413271646015346f, 0.0030602829065173864f, 0.004960220772773027f, 0.002241261303424835f, 0.0019368260400369763f, 0.002485819859430194f, 0.0022833961993455887f, 0.0016671775374561548f, 0.0009880687575787306f, 0.003236931748688221f, 0.003380147973075509f, 0.001833967282436788f, 0.002143226331099868f, 0.0013795733684673905f, 0.002004180336371064f, 0.0027916680555790663f, 0.0010943540837615728f, 0.0026027446147054434f, 0.002263848204165697f, 0.0019075707532465458f, 0.001837768591940403f, 0.0018666748655959964f, 0.003939406014978886f, 0.0007635708316229284f, 0.001741735264658928f, 0.0005191947566345334f, 0.001934568746946752f, 0.002903289394453168f, 0.001634864485822618f, 0.0024099608417600393f, 0.0010044187074527144f, 0.002879720414057374f, 0.0007120930822566152f, 0.0019887115340679884f, 0.0020041170064359903f, 0.003086067968979478f, 0.0030934535898268223f, 0.0018945630872622132f, 0.0025283435825258493f, 0.0023430134169757366f, 0.0019226587610319257f, 0.002110670553520322f, 0.0030564479529857635f, 0.0022601846139878035f, 0.001506103202700615f, 0.002804642543196678f, 0.0016870496328920126f);
static const ai_layer_format_type conv2d_13_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_14_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_14_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_14_pad_before_t_in_0_shape_h_const_u32 = 16;

static const ai_u16 conv2d_14_t_in_0_shape_w_const_u16 = 18;
static const ai_u16 conv2d_14_t_in_0_shape_h_const_u16 = 18;
static const ai_u16 conv2d_14_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_14_l_stride_1_const_u16 = 2;
static const ai_u16 conv2d_14_l_stride_0_const_u16 = 2;
static const ai_i8 conv2d_14_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_14_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_14_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_14_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_14_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.013590601272881031f, 0.006315925624221563f, 0.009187034331262112f, 0.005481945350766182f, 0.0037696058861911297f, 0.0064432029612362385f, 0.004689622670412064f, 0.003510805545374751f, 0.0035507751163095236f, 0.005406776908785105f, 0.0027996303979307413f, 0.004866526462137699f, 0.001761933439411223f, 0.006764352321624756f, 0.006661312188953161f, 0.0063724578358232975f, 0.007921362295746803f, 0.004050487652420998f, 0.002406390616670251f, 0.014512463472783566f, 0.0049314652569592f, 0.006371945608407259f, 0.005125518888235092f, 0.0031369978096336126f, 0.007985686883330345f, 0.0019780287984758615f, 0.0032043568789958954f, 0.006548795849084854f, 0.003880738513544202f, 0.003194455523043871f, 0.007260842714458704f, 0.0039436956867575645f, 0.004292352125048637f, 0.005432582460343838f, 0.007997682318091393f, 0.0034681090619415045f, 0.004291214514523745f, 0.008127499371767044f, 0.010040552355349064f, 0.007705803494900465f, 0.017561398446559906f, 0.015551775693893433f, 0.007878283970057964f, 0.007290076930075884f, 0.002581304404884577f, 0.0036327484995126724f, 0.010833708569407463f, 0.007450649049133062f, 0.004115933086723089f, 0.00882040522992611f, 0.006975235417485237f, 0.0024108225479722023f, 0.003677048720419407f, 0.0026440382935106754f, 0.01502691674977541f, 0.0060582892037928104f, 0.0047380756586790085f, 0.0067098187282681465f, 0.002374589443206787f, 0.003633514977991581f, 0.003590440144762397f, 0.004456691909581423f, 0.006524406373500824f, 0.006738720927387476f, 0.003984882961958647f, 0.01622001826763153f, 0.002382376231253147f, 0.00510883005335927f, 0.0036610434763133526f, 0.004629242233932018f, 0.007365888450294733f, 0.003597262082621455f, 0.007059300784021616f, 0.0059306807816028595f, 0.0075791277922689915f, 0.007518677040934563f, 0.0036970926448702812f, 0.002797494176775217f, 0.005123715847730637f, 0.036128658801317215f, 0.00309440353885293f, 0.009525555185973644f, 0.008478464558720589f, 0.009284399449825287f, 0.006239394657313824f, 0.0027340089436620474f, 0.0023157205432653427f, 0.008737483993172646f, 0.004963046871125698f, 0.00576495286077261f, 0.00372694106772542f, 0.004078161437064409f, 0.004151869565248489f, 0.004994433373212814f, 0.0037727784365415573f, 0.011941282078623772f);
static const ai_u16 conv2d_14_t_out_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_14_t_out_0_shape_h_const_u16 = 8;

static const ai_u16 conv2d_15_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_15_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_15_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_15_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_15_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_15_t_out_0_shape_ch_const_u16 = 16;
static const ai_i8 conv2d_15_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_15_t_out_0_fmt_zero_const_s8 = -2;
static const ai_float conv2d_15_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_15_t_out_0_fmt_scale_const_f32 = 0.04509761556982994f;
static const ai_float conv2d_15_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.001917639165185392f, 0.0015109353698790073f, 0.0025649734307080507f, 0.002880812855437398f, 0.003244965337216854f, 0.0029652107041329145f, 0.0021709820721298456f, 0.0021531544625759125f, 0.0027988862711936235f, 0.0022191505413502455f, 0.0019310838542878628f, 0.002894252771511674f, 0.0022422117181122303f, 0.002507301513105631f, 0.002462332369759679f, 0.0037913001142442226f);
static const ai_layer_format_type conv2d_15_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;



static const ai_u16 conv2d_19_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_19_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_19_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_19_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_19_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 conv2d_19_t_out_0_shape_ch_const_u16 = 96;
static const ai_i8 conv2d_19_t_in_0_fmt_zero_const_s8 = -2;
static const ai_i8 conv2d_19_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_19_t_in_0_fmt_scale_const_f32 = 0.04509761556982994f;
static const ai_float conv2d_19_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_19_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.003087040036916733f, 0.002326167654246092f, 0.0035402823705226183f, 0.003208479844033718f, 0.0032224447932094336f, 0.001716451020911336f, 0.002846392337232828f, 0.0022044251672923565f, 0.004453692585229874f, 0.0029234993271529675f, 0.005108667071908712f, 0.0021034670062363148f, 0.005390428937971592f, 0.0010754899121820927f, 0.0017712305998429656f, 0.0024218223989009857f, 0.0035865011159330606f, 0.002223343588411808f, 0.0033862364944070578f, 0.0022762310691177845f, 0.0028768477495759726f, 0.003976738080382347f, 0.004563035909086466f, 0.002588588511571288f, 0.004610614851117134f, 0.004613817669451237f, 0.004139287862926722f, 0.003005425212904811f, 0.001244488637894392f, 0.006840815301984549f, 0.0024232955183833838f, 0.00231547630392015f, 0.005585487466305494f, 0.0020099091343581676f, 0.0015471368096768856f, 0.003869181964546442f, 0.011027862317860126f, 0.004078048747032881f, 0.005518914666026831f, 0.0033269308041781187f, 0.0057787587866187096f, 0.004381478298455477f, 0.0028709513135254383f, 0.0012044989271089435f, 0.0013222573325037956f, 0.003908142447471619f, 0.003446937073022127f, 0.002914718585088849f, 0.0037124163936823606f, 0.004118822049349546f, 0.0012845878954976797f, 0.004210508894175291f, 0.004782321397215128f, 0.004337586462497711f, 0.0034483338240534067f, 0.004313475918024778f, 0.00393285695463419f, 0.004461710341274738f, 0.00076407624874264f, 0.004318734165281057f, 0.004427370615303516f, 0.0029772683046758175f, 0.003913101274520159f, 0.012478587217628956f, 0.0031878286972641945f, 0.002733559813350439f, 0.00899385754019022f, 0.0026174646336585283f, 0.0014492931077256799f, 0.0025070265401154757f, 0.0030270088464021683f, 0.003171728691086173f, 0.0032388230320066214f, 0.0033038691617548466f, 0.0027483240701258183f, 0.0025140137877315283f, 0.0033871610648930073f, 0.0022682882845401764f, 0.002801955910399556f, 0.0029752899426966906f, 0.005062615033239126f, 0.00315265916287899f, 0.0034431887324899435f, 0.0034117470495402813f, 0.0034939826000481844f, 0.0032383007928729057f, 0.0024747010320425034f, 0.018891852349042892f, 0.0019929460249841213f, 0.004449015483260155f, 0.0019407387590035796f, 0.0038481804076582193f, 0.01030763890594244f, 0.0029465656261891127f, 0.0024501781444996595f, 0.0028078292962163687f);
static const ai_layer_format_type conv2d_19_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_20_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_20_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_20_pad_before_t_in_0_shape_h_const_u32 = 8;

static const ai_u16 conv2d_20_t_in_0_shape_w_const_u16 = 10;
static const ai_u16 conv2d_20_t_in_0_shape_h_const_u16 = 10;
static const ai_u16 conv2d_20_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_20_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_20_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_20_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_20_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_20_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_20_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_20_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.015681108459830284f, 0.00915464200079441f, 0.00715752923861146f, 0.01724252849817276f, 0.005269884597510099f, 0.011450772173702717f, 0.010860656388103962f, 0.011312637478113174f, 0.00751526840031147f, 0.013162962161004543f, 0.007243120577186346f, 0.010898571461439133f, 0.012707462534308434f, 0.012427505105733871f, 0.01158548891544342f, 0.008230771869421005f, 0.015805555507540703f, 0.009321708232164383f, 0.008678793907165527f, 0.009771126322448254f, 0.015682602301239967f, 0.009439138695597649f, 0.005267549306154251f, 0.009430762380361557f, 0.0045209466479718685f, 0.010495754890143871f, 0.005450108088552952f, 0.005407807417213917f, 0.009678786620497704f, 0.004397285170853138f, 0.008724160492420197f, 0.0070500499568879604f, 0.005930057726800442f, 0.008101927116513252f, 0.022773930802941322f, 0.005709811579436064f, 0.01592102088034153f, 0.007596505340188742f, 0.009494824334979057f, 0.010004861280322075f, 0.008466471917927265f, 0.006148113403469324f, 0.011483356356620789f, 0.025768008083105087f, 0.013025370426476002f, 0.009438185952603817f, 0.006661601364612579f, 0.014516383409500122f, 0.00781605951488018f, 0.016771666705608368f, 0.012597472406923771f, 0.005974171683192253f, 0.004372422583401203f, 0.010247311554849148f, 0.009835151955485344f, 0.01696203649044037f, 0.012763223610818386f, 0.0072723072953522205f, 0.012080395594239235f, 0.007160521578043699f, 0.00754100875928998f, 0.012791912071406841f, 0.0068069095723330975f, 0.013615960255265236f, 0.01456380169838667f, 0.010669100098311901f, 0.010064986534416676f, 0.011982003226876259f, 0.016804315149784088f, 0.006430639419704676f, 0.013485023751854897f, 0.007837716490030289f, 0.013572187162935734f, 0.012365437112748623f, 0.008501735515892506f, 0.009344742633402348f, 0.012705590575933456f, 0.018282314762473106f, 0.011601448059082031f, 0.014119498431682587f, 0.009415731765329838f, 0.007923787459731102f, 0.009625187143683434f, 0.009130625054240227f, 0.006913086399435997f, 0.011772477999329567f, 0.005091973580420017f, 0.006663504056632519f, 0.010252829641103745f, 0.01944403536617756f, 0.011328384280204773f, 0.008186965249478817f, 0.004281203728169203f, 0.009964313358068466f, 0.015464122407138348f, 0.014303695410490036f);
static const ai_u16 conv2d_20_t_out_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_20_t_out_0_shape_h_const_u16 = 8;

static const ai_u16 conv2d_21_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_21_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_21_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_21_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_21_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_21_t_out_0_shape_ch_const_u16 = 16;
static const ai_i8 conv2d_21_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_21_t_out_0_fmt_zero_const_s8 = 17;
static const ai_float conv2d_21_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_21_t_out_0_fmt_scale_const_f32 = 0.06923279166221619f;
static const ai_float conv2d_21_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0024898711126297712f, 0.002522875787690282f, 0.002712649293243885f, 0.00183517683763057f, 0.0025834175758063793f, 0.003828096901997924f, 0.0023526265285909176f, 0.003030326683074236f, 0.002011439995840192f, 0.0020709235686808825f, 0.0027938461862504482f, 0.0025389897637069225f, 0.002563095185905695f, 0.002597072860226035f, 0.0026351939886808395f, 0.0029094559140503407f);
static const ai_layer_format_type conv2d_21_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;




static const ai_u16 conv2d_26_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_26_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_26_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_26_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_26_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 conv2d_26_t_out_0_shape_ch_const_u16 = 96;
static const ai_i8 conv2d_26_t_in_0_fmt_zero_const_s8 = 17;
static const ai_i8 conv2d_26_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_26_t_in_0_fmt_scale_const_f32 = 0.06923279166221619f;
static const ai_float conv2d_26_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_26_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.002430869732052088f, 0.0035438365302979946f, 0.002226774813607335f, 0.0024592983536422253f, 0.003950477112084627f, 0.0029678677674382925f, 0.002541402354836464f, 0.002278354251757264f, 0.002841687062755227f, 0.002713199006393552f, 0.0020315467845648527f, 0.00272461399435997f, 0.002562175039201975f, 0.0033621734473854303f, 0.0024959035217761993f, 0.00341172912158072f, 0.0023141466081142426f, 0.003308233106508851f, 0.004413335584104061f, 0.0025367841590195894f, 0.0026230027433484793f, 0.0021163313649594784f, 0.0034974943846464157f, 0.0034785461612045765f, 0.0028526238165795803f, 0.0023398895282298326f, 0.005034787114709616f, 0.003983970265835524f, 0.0027304566465318203f, 0.005306018982082605f, 0.0021786317229270935f, 0.003632407635450363f, 0.0030488045886158943f, 0.0029089702293276787f, 0.0024368963204324245f, 0.002438242780044675f, 0.003580370917916298f, 0.0024012511130422354f, 0.0023543578572571278f, 0.003139815991744399f, 0.002740312134847045f, 0.002430787542834878f, 0.0059970286674797535f, 0.002553712110966444f, 0.003204799722880125f, 0.0014004168333485723f, 0.0028569400310516357f, 0.001344709424301982f, 0.0032730665989220142f, 0.002753952983766794f, 0.0024819045793265104f, 0.0038566216826438904f, 0.0023593620862811804f, 0.002204860094934702f, 0.003002006094902754f, 0.004394338931888342f, 0.002174842869862914f, 0.0015859517734497786f, 0.0018878435948863626f, 0.0033056980464607477f, 0.002564125694334507f, 0.0015582646010443568f, 0.0019474533619359136f, 0.006278446409851313f, 0.0035821667406708f, 0.003273610258474946f, 0.0017644319450482726f, 0.00146260776091367f, 0.002183484146371484f, 0.002419850090518594f, 0.00163445679936558f, 0.0019259696127846837f, 0.0015780264511704445f, 0.0018741810927167535f, 0.0023864677641540766f, 0.0022681760601699352f, 0.004887233953922987f, 0.0021963375620543957f, 0.0045002177357673645f, 0.005203427746891975f, 0.001714977785013616f, 0.002440058859065175f, 0.0028680807445198298f, 0.0027635949663817883f, 0.00254049152135849f, 0.0017543202266097069f, 0.003466847585514188f, 0.0021859777625650167f, 0.0014152531512081623f, 0.0025500627234578133f, 0.001278031850233674f, 0.0021459690760821104f, 0.003183473367244005f, 0.002886627335101366f, 0.0013099906500428915f, 0.004984147381037474f);
static const ai_layer_format_type conv2d_26_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_27_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_27_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_27_pad_before_t_in_0_shape_h_const_u32 = 8;

static const ai_u16 conv2d_27_t_in_0_shape_w_const_u16 = 10;
static const ai_u16 conv2d_27_t_in_0_shape_h_const_u16 = 10;
static const ai_u16 conv2d_27_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_27_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_27_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_27_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_27_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_27_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_27_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_27_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0187609251588583f, 0.009324982762336731f, 0.007623013108968735f, 0.009291329421103f, 0.008662845939397812f, 0.00632141949608922f, 0.006130694877356291f, 0.01625126600265503f, 0.00921420194208622f, 0.013945668935775757f, 0.00719142472371459f, 0.010729889385402203f, 0.010373048484325409f, 0.007249808870255947f, 0.00558947678655386f, 0.010711915791034698f, 0.0072728474624454975f, 0.008971180766820908f, 0.007599083241075277f, 0.005009555257856846f, 0.009406892582774162f, 0.005538391415029764f, 0.006795073859393597f, 0.009358138777315617f, 0.00896179024130106f, 0.006889566779136658f, 0.009181338362395763f, 0.004698037635535002f, 0.015336853452026844f, 0.00495988130569458f, 0.008622216992080212f, 0.009913209825754166f, 0.015204205177724361f, 0.0116855064406991f, 0.007634964305907488f, 0.007531138136982918f, 0.005549069959670305f, 0.0128357307985425f, 0.011459671892225742f, 0.006576291751116514f, 0.008557134307920933f, 0.01651477999985218f, 0.0031902859918773174f, 0.01592146046459675f, 0.007589679677039385f, 0.008273652754724026f, 0.005555390380322933f, 0.010616270825266838f, 0.005833262577652931f, 0.007429690100252628f, 0.01362040638923645f, 0.011450264602899551f, 0.006350788753479719f, 0.00787996407598257f, 0.019344115629792213f, 0.005424091592431068f, 0.0071594747714698315f, 0.012447364628314972f, 0.013093897141516209f, 0.006033340003341436f, 0.006633318495005369f, 0.011339598335325718f, 0.017368920147418976f, 0.007044883444905281f, 0.004190434236079454f, 0.012029066681861877f, 0.012789863161742687f, 0.014752390794456005f, 0.017435265704989433f, 0.007220870349556208f, 0.007914007641375065f, 0.01396709494292736f, 0.0116815697401762f, 0.010150264948606491f, 0.011138219386339188f, 0.004957942757755518f, 0.008314348757266998f, 0.008890856057405472f, 0.005264872685074806f, 0.0031381575390696526f, 0.010194225236773491f, 0.007581981364637613f, 0.0069078607484698296f, 0.017007609829306602f, 0.005425150040537119f, 0.00452808104455471f, 0.006030677817761898f, 0.00923354271799326f, 0.00923926942050457f, 0.008913510479032993f, 0.006260547786951065f, 0.005656936205923557f, 0.005163109395653009f, 0.006461399141699076f, 0.012050466611981392f, 0.008378597907721996f);
static const ai_u16 conv2d_27_t_out_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_27_t_out_0_shape_h_const_u16 = 8;

static const ai_u16 conv2d_28_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_28_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_28_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_28_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_28_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_28_t_out_0_shape_ch_const_u16 = 16;
static const ai_i8 conv2d_28_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_28_t_out_0_fmt_zero_const_s8 = 14;
static const ai_float conv2d_28_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_28_t_out_0_fmt_scale_const_f32 = 0.10022054612636566f;
static const ai_float conv2d_28_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0035034362226724625f, 0.0044860863126814365f, 0.0029773004353046417f, 0.0036449420731514692f, 0.004867255687713623f, 0.0036817174404859543f, 0.005201718769967556f, 0.0035229602362960577f, 0.0030425144359469414f, 0.004101619124412537f, 0.0021374959032982588f, 0.0045971921645104885f, 0.0035684651229530573f, 0.004623107612133026f, 0.0028310157358646393f, 0.002722406992688775f);
static const ai_layer_format_type conv2d_28_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_30_t_in_0_shape_w_const_u16 = 8;
static const ai_u16 conv2d_30_t_in_0_shape_h_const_u16 = 8;
static const ai_u16 conv2d_30_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_30_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_30_t_in_0_shape_ch_const_u16 = 16;
static const ai_u16 conv2d_30_t_out_0_shape_ch_const_u16 = 96;
static const ai_i8 conv2d_30_t_in_0_fmt_zero_const_s8 = 14;
static const ai_i8 conv2d_30_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_30_t_in_0_fmt_scale_const_f32 = 0.10022054612636566f;
static const ai_float conv2d_30_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_30_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0017237093998119235f, 0.0018604503711685538f, 0.001699859625659883f, 0.0011629638029262424f, 0.001932751853018999f, 0.0017604570602998137f, 0.0024146358482539654f, 0.0016422263579443097f, 0.002291428856551647f, 0.000994721194729209f, 0.0016649559838697314f, 0.0011125595774501562f, 0.0014302858617156744f, 0.002037291182205081f, 0.0014516266528517008f, 0.0023737873416393995f, 0.0008891152101568878f, 0.00106794573366642f, 0.0011576218530535698f, 0.0019508474506437778f, 0.0011525941081345081f, 0.0010271474020555615f, 0.0014273924753069878f, 0.0025116391479969025f, 0.001191485789604485f, 0.0020323540084064007f, 0.0014586071483790874f, 0.0010836039436981082f, 0.001484253560192883f, 0.0017188327619805932f, 0.0021726912818849087f, 0.0010438695317134261f, 0.0015495106345042586f, 0.0016176652861759067f, 0.0011442232644185424f, 0.0034346224274486303f, 0.0020425806287676096f, 0.0012897347332909703f, 0.0025829672813415527f, 0.0010640317341312766f, 0.0022113537415862083f, 0.002336545381695032f, 0.0016606706194579601f, 0.0010413089767098427f, 0.001822068472392857f, 0.002222128678113222f, 0.0009723500697873533f, 0.0021716998890042305f, 0.0013958566123619676f, 0.0023723733611404896f, 0.0010043946094810963f, 0.002269404474645853f, 0.0011038564844056964f, 0.0017660472076386213f, 0.0019921250641345978f, 0.0012330238241702318f, 0.0020602415315806866f, 0.0017207510536536574f, 0.0009006298496387899f, 0.0013997171772643924f, 0.0014029128942638636f, 0.0015636885073035955f, 0.0008603547466918826f, 0.0008712129783816636f, 0.0022302621509879827f, 0.0014813585439696908f, 0.00257147871889174f, 0.003091719700023532f, 0.001231112633831799f, 0.0020403829403221607f, 0.002757462440058589f, 0.0021738505456596613f, 0.00104503333568573f, 0.0018306805286556482f, 0.0016359981382265687f, 0.0016040981281548738f, 0.00131103559397161f, 0.0017120325937867165f, 0.0012615497689694166f, 0.002258467022329569f, 0.0014200714649632573f, 0.0006618829793296754f, 0.0014056330546736717f, 0.0022693753708153963f, 0.001089655328541994f, 0.0020988821052014828f, 0.002267187926918268f, 0.0016186342108994722f, 0.0007715080282650888f, 0.0009108875528909266f, 0.0024967645294964314f, 0.0016371544916182756f, 0.003408609889447689f, 0.0008594895480200648f, 0.0011610629735514522f, 0.001573986024595797f);
static const ai_layer_format_type conv2d_30_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_31_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_31_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_31_pad_before_t_in_0_shape_h_const_u32 = 8;

static const ai_u16 conv2d_31_t_in_0_shape_w_const_u16 = 10;
static const ai_u16 conv2d_31_t_in_0_shape_h_const_u16 = 10;
static const ai_u16 conv2d_31_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_31_l_stride_1_const_u16 = 2;
static const ai_u16 conv2d_31_l_stride_0_const_u16 = 2;
static const ai_i8 conv2d_31_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_31_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_31_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_31_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_31_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.006259993650019169f, 0.005755294114351273f, 0.002885549096390605f, 0.004718618001788855f, 0.007896572351455688f, 0.0042496840469539165f, 0.007260012440383434f, 0.005094069987535477f, 0.0037029471714049578f, 0.008240514434874058f, 0.004031756892800331f, 0.004647849593311548f, 0.004303034394979477f, 0.00372047396376729f, 0.005326184444129467f, 0.003312924411147833f, 0.010177180171012878f, 0.01526741124689579f, 0.005827257875353098f, 0.003195623867213726f, 0.008390807546675205f, 0.005212742369621992f, 0.006487375125288963f, 0.0025749222841113806f, 0.014756760559976101f, 0.004883136600255966f, 0.004233462270349264f, 0.004818742163479328f, 0.009816402569413185f, 0.0058832187205553055f, 0.006705873645842075f, 0.00804942473769188f, 0.003987685311585665f, 0.010447603650391102f, 0.008577384054660797f, 0.0029841619543731213f, 0.004831313621252775f, 0.008736379444599152f, 0.00538500165566802f, 0.017851393669843674f, 0.004226537887006998f, 0.005731902550905943f, 0.005317918490618467f, 0.006120649632066488f, 0.006140628829598427f, 0.003264006460085511f, 0.007909092120826244f, 0.0032106004655361176f, 0.006504965014755726f, 0.00530202267691493f, 0.006213739048689604f, 0.004668892826884985f, 0.007196249905973673f, 0.003527434077113867f, 0.004091996233910322f, 0.003879840951412916f, 0.003838023403659463f, 0.004996195901185274f, 0.01451534777879715f, 0.005084827542304993f, 0.006039994768798351f, 0.01331839244812727f, 0.006040374748408794f, 0.006587564013898373f, 0.005717990919947624f, 0.0045516230165958405f, 0.002767308847978711f, 0.002648718887940049f, 0.004050925374031067f, 0.004324633162468672f, 0.004054458346217871f, 0.004395093768835068f, 0.02367200329899788f, 0.008838788606226444f, 0.0045356834307312965f, 0.005616558250039816f, 0.009096330963075161f, 0.00655226968228817f, 0.004732703790068626f, 0.002720329910516739f, 0.008610407821834087f, 0.01001936849206686f, 0.008713492192327976f, 0.006471849977970123f, 0.006967075634747744f, 0.0032557565718889236f, 0.004573196638375521f, 0.004882839508354664f, 0.007658894639462233f, 0.013492953963577747f, 0.003303401404991746f, 0.0046229311265051365f, 0.004529255907982588f, 0.0068175033666193485f, 0.00966707430779934f, 0.008926674723625183f);
static const ai_u16 conv2d_31_t_out_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_31_t_out_0_shape_h_const_u16 = 4;

static const ai_u16 conv2d_32_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_32_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_32_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_32_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_32_t_in_0_shape_ch_const_u16 = 96;
static const ai_u16 conv2d_32_t_out_0_shape_ch_const_u16 = 32;
static const ai_i8 conv2d_32_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_32_t_out_0_fmt_zero_const_s8 = -3;
static const ai_float conv2d_32_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_32_t_out_0_fmt_scale_const_f32 = 0.05053471401333809f;
static const ai_float conv2d_32_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0014893084298819304f, 0.002736829686909914f, 0.0019984275568276644f, 0.0020908848382532597f, 0.0015054758405312896f, 0.0028015929274260998f, 0.0024072732776403427f, 0.0025516399182379246f, 0.002760416828095913f, 0.0020242659375071526f, 0.002886482747271657f, 0.0021114598494023085f, 0.0021014821249991655f, 0.0025564865209162235f, 0.0021944919135421515f, 0.002086715307086706f, 0.0022939920891076326f, 0.0017725761281326413f, 0.0015100467717275023f, 0.002096788724884391f, 0.002350135240703821f, 0.0024216969031840563f, 0.002340268576517701f, 0.0019760322757065296f, 0.001938582630828023f, 0.002355399774387479f, 0.002785168122500181f, 0.0016552143497392535f, 0.002527073724195361f, 0.0035643484443426132f, 0.0024745340924710035f, 0.0017638400895521045f);
static const ai_layer_format_type conv2d_32_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;



static const ai_u16 conv2d_36_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_36_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_36_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_36_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_36_t_in_0_shape_ch_const_u16 = 32;
static const ai_u16 conv2d_36_t_out_0_shape_ch_const_u16 = 192;
static const ai_i8 conv2d_36_t_in_0_fmt_zero_const_s8 = -3;
static const ai_i8 conv2d_36_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_36_t_in_0_fmt_scale_const_f32 = 0.05053471401333809f;
static const ai_float conv2d_36_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_36_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00175359973218292f, 0.002345404587686062f, 0.0028165988624095917f, 0.00273721176199615f, 0.0021284418180584908f, 0.0035474481992423534f, 0.002076228614896536f, 0.0017995268572121859f, 0.0030613811686635017f, 0.001120575936511159f, 0.0028077340684831142f, 0.0020705917850136757f, 0.005953438580036163f, 0.0011624020989984274f, 0.0026380158960819244f, 0.0019687102176249027f, 0.0033762885723263025f, 0.003825620748102665f, 0.003450353629887104f, 0.002189415041357279f, 0.002620060695335269f, 0.002031804295256734f, 0.0017715776339173317f, 0.0018623078940436244f, 0.0028739783447235823f, 0.0032183837611228228f, 0.0025585112161934376f, 0.00256023439578712f, 0.0066275447607040405f, 0.001592510030604899f, 0.004584008827805519f, 0.001937453867867589f, 0.002807952929288149f, 0.0024216067977249622f, 0.0020966297015547752f, 0.0011553006479516625f, 0.003417583415284753f, 0.0018375779036432505f, 0.002637686673551798f, 0.001529513276182115f, 0.0032764060888439417f, 0.0027689130511134863f, 0.002117999130859971f, 0.003064498072490096f, 0.001780258258804679f, 0.003373826155439019f, 0.0018865858437493443f, 0.003690781071782112f, 0.0027581024914979935f, 0.0009783050045371056f, 0.004117813427001238f, 0.001775215147063136f, 0.0020916566718369722f, 0.0024910348001867533f, 0.0022817514836788177f, 0.002479121321812272f, 0.0026359029579907656f, 0.002634554635733366f, 0.0028701485134661198f, 0.002257291693240404f, 0.0037094017025083303f, 0.0022427872754633427f, 0.0018446752801537514f, 0.002755390014499426f, 0.003473990596830845f, 0.00252149673178792f, 0.0038880575448274612f, 0.0030894686933606863f, 0.0026267743669450283f, 0.003172788070514798f, 0.0018848063191398978f, 0.0023011742159724236f, 0.002361779101192951f, 0.003458037506788969f, 0.002406190149486065f, 0.0028748696204274893f, 0.0025673597119748592f, 0.003193598473444581f, 0.003531642258167267f, 0.0014583026058971882f, 0.0016125915572047234f, 0.0030005830340087414f, 0.002519126981496811f, 0.0011335773160681129f, 0.0014392398297786713f, 0.002079934813082218f, 0.00505228154361248f, 0.002196623245254159f, 0.003357658861204982f, 0.0019679854158312082f, 0.00541050685569644f, 0.0016497215256094933f, 0.002460428047925234f, 0.0025895482394844294f, 0.0030512127559632063f, 0.003353029489517212f, 0.0018260303186252713f, 0.0022357397247105837f, 0.001296296832151711f, 0.002882277127355337f, 0.0026981846895068884f, 0.0023233825340867043f, 0.003112213686108589f, 0.004655452910810709f, 0.0019190572202205658f, 0.0026910032611340284f, 0.001714185462333262f, 0.00190233055036515f, 0.0017653757240623236f, 0.0019474647706374526f, 0.004825027659535408f, 0.00146964518353343f, 0.0021095455158501863f, 0.0028697815723717213f, 0.0023018810898065567f, 0.0020635579712688923f, 0.0016937273321673274f, 0.002131655113771558f, 0.0023810435086488724f, 0.0026694429107010365f, 0.003659443464130163f, 0.003083925461396575f, 0.0022440508473664522f, 0.0021419995464384556f, 0.003561289282515645f, 0.0026633013039827347f, 0.0038672813680022955f, 0.0016269572079181671f, 0.004060431849211454f, 0.0026911795139312744f, 0.0023324384819716215f, 0.0038147938903421164f, 0.0018691068980842829f, 0.002111709676682949f, 0.003772004507482052f, 0.0028127373661845922f, 0.0013989844592288136f, 0.0018279743380844593f, 0.003425567876547575f, 0.0026632603257894516f, 0.000936598633415997f, 0.00191302050370723f, 0.002128036692738533f, 0.002688673324882984f, 0.006407829467207193f, 0.002117677591741085f, 0.0029180925339460373f, 0.0019753542728722095f, 0.0009224277455359697f, 0.0028470694087445736f, 0.004003845155239105f, 0.0023274500854313374f, 0.007136445492506027f, 0.002750084735453129f, 0.0015787630109116435f, 0.0029635345563292503f, 0.004988919012248516f, 0.002163353143259883f, 0.001605656580068171f, 0.002966807223856449f, 0.0016588837606832385f, 0.002184855518862605f, 0.0018112953985109925f, 0.0030023420695215464f, 0.0036662283819168806f, 0.004331377800554037f, 0.0025138817727565765f, 0.0028933484572917223f, 0.002050603274255991f, 0.00238797883503139f, 0.001287085353396833f, 0.0027350925374776125f, 0.0024449550546705723f, 0.0014556744135916233f, 0.0026865671388804913f, 0.0016172213945537806f, 0.0025869393721222878f, 0.0060945069417357445f, 0.001829015091061592f, 0.0014443417312577367f, 0.003177057020366192f, 0.008510025218129158f, 0.0021764927078038454f, 0.0018041001167148352f, 0.0017133221263065934f, 0.0026981323026120663f, 0.003224613144993782f, 0.002723624464124441f, 0.001949466997757554f, 0.003127087838947773f, 0.0024469958152621984f, 0.0028525632806122303f);
static const ai_layer_format_type conv2d_36_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_37_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_37_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_37_pad_before_t_in_0_shape_h_const_u32 = 4;

static const ai_u16 conv2d_37_t_in_0_shape_w_const_u16 = 6;
static const ai_u16 conv2d_37_t_in_0_shape_h_const_u16 = 6;
static const ai_u16 conv2d_37_t_in_0_shape_ch_const_u16 = 192;
static const ai_u16 conv2d_37_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_37_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_37_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_37_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_37_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_37_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_37_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.009887357242405415f, 0.007304647006094456f, 0.008112492971122265f, 0.007558537181466818f, 0.00776736531406641f, 0.005497157573699951f, 0.006741110235452652f, 0.005224332679063082f, 0.009509561583399773f, 0.010754464194178581f, 0.009451943449676037f, 0.019143790006637573f, 0.004147150553762913f, 0.06148802489042282f, 0.009080693125724792f, 0.01086147129535675f, 0.0072883181273937225f, 0.007125383242964745f, 0.006458179093897343f, 0.011485750786960125f, 0.005732537247240543f, 0.008292227052152157f, 0.009206850081682205f, 0.013247798196971416f, 0.027357827872037888f, 0.005961280316114426f, 0.005075826309621334f, 0.009607131592929363f, 0.011659449897706509f, 0.015619253739714622f, 0.006645967718213797f, 0.007973900064826012f, 0.0073658679611980915f, 0.007966026663780212f, 0.00771408062428236f, 0.01314457505941391f, 0.009839785285294056f, 0.008085477165877819f, 0.0056956857442855835f, 0.00649307994171977f, 0.0072956630028784275f, 0.010213298723101616f, 0.00798887386918068f, 0.008746825158596039f, 0.007615070324391127f, 0.008847229182720184f, 0.010082872584462166f, 0.012076952494680882f, 0.014481553807854652f, 0.015067832544445992f, 0.003979469649493694f, 0.016562825068831444f, 0.013504870235919952f, 0.012212651781737804f, 0.014458227902650833f, 0.006078420672565699f, 0.0065033226273953915f, 0.00706782191991806f, 0.004271678626537323f, 0.006691720802336931f, 0.004417724907398224f, 0.007355350535362959f, 0.005716483574360609f, 0.005917665548622608f, 0.008830921724438667f, 0.012694107368588448f, 0.007935648784041405f, 0.008460323326289654f, 0.004428309388458729f, 0.008264824748039246f, 0.012086671777069569f, 0.010909316129982471f, 0.008070417679846287f, 0.008466877974569798f, 0.008037333376705647f, 0.009582141414284706f, 0.007457999512553215f, 0.011817196384072304f, 0.007015427108854055f, 0.011536162346601486f, 0.011394853703677654f, 0.007833569310605526f, 0.008122113533318043f, 0.013684157282114029f, 0.007980392314493656f, 0.013552622869610786f, 0.005160304252058268f, 0.006600719410926104f, 0.009851526468992233f, 0.006178795360028744f, 0.013235547579824924f, 0.008884314447641373f, 0.006585342343896627f, 0.008532785810530186f, 0.00909794308245182f, 0.005552198272198439f, 0.015896126627922058f, 0.007623114623129368f, 0.009345555678009987f, 0.011682030744850636f, 0.00976155512034893f, 0.007065506186336279f, 0.0105277793481946f, 0.008773613721132278f, 0.005306803621351719f, 0.007202827371656895f, 0.0068322839215397835f, 0.013096660375595093f, 0.017563002184033394f, 0.009883602149784565f, 0.009644054807722569f, 0.012404815293848515f, 0.01467710267752409f, 0.01082694437354803f, 0.007954582571983337f, 0.008049963042140007f, 0.006788372527807951f, 0.015262864530086517f, 0.009092090651392937f, 0.007316711824387312f, 0.011028105393052101f, 0.004191266372799873f, 0.00555494986474514f, 0.022529922425746918f, 0.00977539736777544f, 0.006955181248486042f, 0.006958129815757275f, 0.011144601739943027f, 0.006122085265815258f, 0.0065133958123624325f, 0.0149068059399724f, 0.0056664119474589825f, 0.008424216881394386f, 0.010590167716145515f, 0.0036100337747484446f, 0.00616504717618227f, 0.04236679524183273f, 0.014451577328145504f, 0.008420995436608791f, 0.004456070717424154f, 0.013775997795164585f, 0.009484679438173771f, 0.015055754221975803f, 0.005200641229748726f, 0.008284319192171097f, 0.0071133156307041645f, 0.00678233290091157f, 0.013942888006567955f, 0.013890211470425129f, 0.00783923827111721f, 0.007398148998618126f, 0.00883102510124445f, 0.007489989977329969f, 0.008086652494966984f, 0.008693048730492592f, 0.010840097442269325f, 0.007304275408387184f, 0.012496167793869972f, 0.018790021538734436f, 0.019184652715921402f, 0.008243686519563198f, 0.011339736171066761f, 0.013353696092963219f, 0.00823064986616373f, 0.0069228666834533215f, 0.006054403260350227f, 0.01044737920165062f, 0.008437162265181541f, 0.04726267233490944f, 0.008665380999445915f, 0.010360544547438622f, 0.006761420052498579f, 0.0093656275421381f, 0.011881607584655285f, 0.008230566047132015f, 0.009588015265762806f, 0.015818579122424126f, 0.010812120512127876f, 0.00932522863149643f, 0.01603771559894085f, 0.005370594561100006f, 0.0107183288782835f, 0.010953519493341446f, 0.014420750550925732f, 0.00801834650337696f, 0.009221655316650867f, 0.007970123551785946f, 0.010077296756207943f, 0.006581613328307867f, 0.006468338891863823f, 0.009048850275576115f, 0.009756580926477909f);
static const ai_u16 conv2d_37_t_out_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_37_t_out_0_shape_h_const_u16 = 4;

static const ai_u16 conv2d_38_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_38_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_38_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_38_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_38_t_in_0_shape_ch_const_u16 = 192;
static const ai_u16 conv2d_38_t_out_0_shape_ch_const_u16 = 32;
static const ai_i8 conv2d_38_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_38_t_out_0_fmt_zero_const_s8 = -13;
static const ai_float conv2d_38_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_38_t_out_0_fmt_scale_const_f32 = 0.06210874021053314f;
static const ai_float conv2d_38_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00324101815931499f, 0.002159043913707137f, 0.0026706478092819452f, 0.002736081602051854f, 0.001681399648077786f, 0.0015304210828617215f, 0.0022039294708520174f, 0.0016428657108917832f, 0.0014114760560914874f, 0.0025372998788952827f, 0.001937499619089067f, 0.0016139866784214973f, 0.0021605801302939653f, 0.0019139323849231005f, 0.0016256950329989195f, 0.0016923127695918083f, 0.0018552046967670321f, 0.002088378183543682f, 0.0018123877234756947f, 0.001739394385367632f, 0.001642501214519143f, 0.0018130273092538118f, 0.0017856062622740865f, 0.0013315246906131506f, 0.002110297791659832f, 0.002255806466564536f, 0.0017369046108797193f, 0.0012705376138910651f, 0.0011989377671852708f, 0.0016744122840464115f, 0.001935879117809236f, 0.0014609451172873378f);
static const ai_layer_format_type conv2d_38_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;




static const ai_u16 conv2d_43_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_43_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_43_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_43_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_43_t_in_0_shape_ch_const_u16 = 32;
static const ai_u16 conv2d_43_t_out_0_shape_ch_const_u16 = 192;
static const ai_i8 conv2d_43_t_in_0_fmt_zero_const_s8 = -13;
static const ai_i8 conv2d_43_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_43_t_in_0_fmt_scale_const_f32 = 0.06210874021053314f;
static const ai_float conv2d_43_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_43_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0021562804467976093f, 0.0015466787153854966f, 0.0018842295976355672f, 0.0022854108829051256f, 0.0014880348462611437f, 0.0013301257276907563f, 0.0017809186829254031f, 0.0010238786926493049f, 0.0020032813772559166f, 0.001578560215421021f, 0.0023053153418004513f, 0.0032408537808805704f, 0.002122730715200305f, 0.0015650482382625341f, 0.0018010269850492477f, 0.0016928132390603423f, 0.0015299732331186533f, 0.0016910978592932224f, 0.002189441118389368f, 0.0014017008943483233f, 0.0021487458143383265f, 0.0017692336114123464f, 0.0019142302917316556f, 0.0018907678313553333f, 0.0012486828491091728f, 0.001781389000825584f, 0.0018914417596533895f, 0.002106227446347475f, 0.001026553101837635f, 0.0013718048576265574f, 0.00267236796207726f, 0.0016938108019530773f, 0.002175723435357213f, 0.002406942192465067f, 0.0019538316410034895f, 0.002340310253202915f, 0.0009515565470792353f, 0.0016191084869205952f, 0.001545812003314495f, 0.0016063718358054757f, 0.0017397403717041016f, 0.002842445159330964f, 0.0015096621355041862f, 0.0020404690876603127f, 0.0022961082868278027f, 0.0027399978134781122f, 0.0026702010072767735f, 0.001483346102759242f, 0.002047410700470209f, 0.0011740517802536488f, 0.002394067356362939f, 0.002206226345151663f, 0.00078516238136217f, 0.0017762959469109774f, 0.0021595172584056854f, 0.00313386507332325f, 0.0015996004221960902f, 0.0023056184872984886f, 0.00172721641138196f, 0.0029389571864157915f, 0.002428258303552866f, 0.002413996960967779f, 0.0015855778474360704f, 0.002116106217727065f, 0.00320997997187078f, 0.0014838207280263305f, 0.0018473919481039047f, 0.0016124791000038385f, 0.001811924739740789f, 0.0015874197706580162f, 0.0023839285131543875f, 0.001836591400206089f, 0.0021806044969707727f, 0.0013489191187545657f, 0.0016398484585806727f, 0.0015163938514888287f, 0.0022100075148046017f, 0.0019530568970367312f, 0.0019130721921101213f, 0.002568138064816594f, 0.0020391608122736216f, 0.001697542960755527f, 0.001532515394501388f, 0.0022044023498892784f, 0.0019884782377630472f, 0.001531035639345646f, 0.0021196510642766953f, 0.0035252608358860016f, 0.0020472281612455845f, 0.0018250498687848449f, 0.0029152678325772285f, 0.002576539758592844f, 0.0013900873018428683f, 0.001928161596879363f, 0.0017370195128023624f, 0.0017807753756642342f, 0.0013837272999808192f, 0.0011563223088160157f, 0.002387163694947958f, 0.0021956833079457283f, 0.0021912811789661646f, 0.0019333952805027366f, 0.0012603725772351027f, 0.0031809022184461355f, 0.002835371531546116f, 0.0015644683735445142f, 0.0016963341040536761f, 0.002243761671707034f, 0.002209728118032217f, 0.0014329897239804268f, 0.0018464757595211267f, 0.0014814576134085655f, 0.0014443612890318036f, 0.001983494032174349f, 0.0014043912524357438f, 0.0024807543959468603f, 0.0015241897199302912f, 0.0017019262304529548f, 0.0013718638801947236f, 0.001666917116381228f, 0.0032800317276269197f, 0.002401817124336958f, 0.0012796020600944757f, 0.002834416227415204f, 0.0023859902285039425f, 0.002418598160147667f, 0.0017553453799337149f, 0.001605661935172975f, 0.0014075664803385735f, 0.00255720061250031f, 0.002448519691824913f, 0.001929307822138071f, 0.0007412178674712777f, 0.0019922165665775537f, 0.0016548968851566315f, 0.003038556780666113f, 0.00331253744661808f, 0.0022354729007929564f, 0.0019618121441453695f, 0.0015152540290728211f, 0.0019014060962945223f, 0.0024899179115891457f, 0.0016582653624936938f, 0.0019908875692635775f, 0.0018623300129547715f, 0.002212202176451683f, 0.0008123930892907083f, 0.0016007453668862581f, 0.002273539314046502f, 0.0023868130519986153f, 0.0014934204518795013f, 0.0026064557023346424f, 0.00232368940487504f, 0.0014787983382120728f, 0.0016357220010831952f, 0.00200864695943892f, 0.0015653228620067239f, 0.0011411984451115131f, 0.0025662826374173164f, 0.0015033689560368657f, 0.002366746310144663f, 0.001679114648140967f, 0.0017911411123350263f, 0.004046011716127396f, 0.0027630901895463467f, 0.0015659532509744167f, 0.0019394403789192438f, 0.0021808757446706295f, 0.0021417972166091204f, 0.0020236473064869642f, 0.0016497106989845634f, 0.0018645364325493574f, 0.001429698895663023f, 0.002122406614944339f, 0.001301292097195983f, 0.0025790922809392214f, 0.0019956612959504128f, 0.001897093141451478f, 0.0015477812848985195f, 0.0024433094076812267f, 0.002065044827759266f, 0.0014710512477904558f, 0.002459141192957759f, 0.0020474623888731003f, 0.0014090950135141611f, 0.0024751543533056974f, 0.0012319838861003518f, 0.002075689844787121f, 0.0019155105110257864f, 0.0015157905872911215f, 0.0015966438222676516f, 0.0015951566165313125f);
static const ai_layer_format_type conv2d_43_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_44_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_44_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_44_pad_before_t_in_0_shape_h_const_u32 = 4;

static const ai_u16 conv2d_44_t_in_0_shape_w_const_u16 = 6;
static const ai_u16 conv2d_44_t_in_0_shape_h_const_u16 = 6;
static const ai_u16 conv2d_44_t_in_0_shape_ch_const_u16 = 192;
static const ai_u16 conv2d_44_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_44_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_44_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_44_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_44_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_44_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_44_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0064058429561555386f, 0.006320850923657417f, 0.013905413448810577f, 0.008908318355679512f, 0.012290657497942448f, 0.010550132021307945f, 0.011809892021119595f, 0.023587064817547798f, 0.009290521033108234f, 0.012879274785518646f, 0.00802408903837204f, 0.010056080296635628f, 0.006971926894038916f, 0.014186559244990349f, 0.00902685895562172f, 0.014782926067709923f, 0.015095562674105167f, 0.008280697278678417f, 0.010101989842951298f, 0.0048973956145346165f, 0.007379630114883184f, 0.011017539538443089f, 0.014318530447781086f, 0.012285154312849045f, 0.0054709757678210735f, 0.00584813579916954f, 0.016129406169056892f, 0.009063316509127617f, 0.020498113706707954f, 0.010825073346495628f, 0.007663636934012175f, 0.009411675855517387f, 0.008382663130760193f, 0.009121464565396309f, 0.011648095212876797f, 0.011038044467568398f, 0.03836090862751007f, 0.009973594918847084f, 0.00982974749058485f, 0.008314683102071285f, 0.010044622235000134f, 0.008872848935425282f, 0.01309360284358263f, 0.004538903944194317f, 0.016392242163419724f, 0.006420198827981949f, 0.007018121890723705f, 0.01839848980307579f, 0.00524534285068512f, 0.010077418759465218f, 0.012140789069235325f, 0.0053247371688485146f, 0.015257841907441616f, 0.012788784690201283f, 0.007290937472134829f, 0.007980035617947578f, 0.011334353126585484f, 0.010388555005192757f, 0.016217485070228577f, 0.009877138771116734f, 0.007149906828999519f, 0.0080338129773736f, 0.011414862237870693f, 0.006405884400010109f, 0.010727048851549625f, 0.011400300078094006f, 0.009386049583554268f, 0.00915516633540392f, 0.017494307830929756f, 0.009624537080526352f, 0.008747980929911137f, 0.0073465267196297646f, 0.014442195184528828f, 0.01459443662315607f, 0.017339972779154778f, 0.005498482380062342f, 0.006710848305374384f, 0.007990061305463314f, 0.004428686574101448f, 0.0163437407463789f, 0.010310203768312931f, 0.007767088245600462f, 0.008684735745191574f, 0.009097617119550705f, 0.006376837845891714f, 0.006272565107792616f, 0.00997192319482565f, 0.005238774232566357f, 0.0043354216031730175f, 0.005774225108325481f, 0.007405292708426714f, 0.010525614023208618f, 0.01159333810210228f, 0.006671312265098095f, 0.009427618235349655f, 0.007055962458252907f, 0.011386271566152573f, 0.007867483422160149f, 0.020042283460497856f, 0.009734471328556538f, 0.0088658994063735f, 0.0056771584786474705f, 0.010873131453990936f, 0.007580520585179329f, 0.006279465276747942f, 0.013386948965489864f, 0.0076018390245735645f, 0.006858060136437416f, 0.009278383105993271f, 0.0101017439737916f, 0.006064805667847395f, 0.00816337764263153f, 0.0063569918274879456f, 0.025073828175663948f, 0.016463221982121468f, 0.010317277163267136f, 0.009507891722023487f, 0.005345433484762907f, 0.0198068767786026f, 0.008857877925038338f, 0.005922622513025999f, 0.012614349834620953f, 0.0076210093684494495f, 0.007387970574200153f, 0.006495277397334576f, 0.009338734671473503f, 0.011578974314033985f, 0.005662011913955212f, 0.015200063586235046f, 0.006549820303916931f, 0.010051988996565342f, 0.005207674577832222f, 0.010631972923874855f, 0.015280898660421371f, 0.014537793584167957f, 0.011634843423962593f, 0.0063212597742676735f, 0.006365763023495674f, 0.005342491902410984f, 0.011383118107914925f, 0.009408388286828995f, 0.011080170050263405f, 0.008661789819598198f, 0.006911404896527529f, 0.018105875700712204f, 0.007273291237652302f, 0.013635019771754742f, 0.012838232330977917f, 0.006068238988518715f, 0.007411078084260225f, 0.011789276264607906f, 0.011598601937294006f, 0.00998106598854065f, 0.0129152312874794f, 0.01297072134912014f, 0.007120446301996708f, 0.010048109106719494f, 0.015224002301692963f, 0.008111751638352871f, 0.012066378258168697f, 0.0034719170071184635f, 0.014790153130888939f, 0.005102633032947779f, 0.006454938091337681f, 0.007254249881953001f, 0.012976741418242455f, 0.01486993208527565f, 0.005954768508672714f, 0.015536279417574406f, 0.009114961139857769f, 0.009189551696181297f, 0.022627880796790123f, 0.016446752473711967f, 0.00438261404633522f, 0.006057101301848888f, 0.007467735558748245f, 0.010821210220456123f, 0.006576954387128353f, 0.005033050663769245f, 0.010441765189170837f, 0.010630705393850803f, 0.01795024424791336f, 0.006257500033825636f, 0.013069843873381615f, 0.013845395296812057f, 0.009796124882996082f, 0.0058183977380394936f, 0.008904121816158295f, 0.008770131506025791f, 0.013514097779989243f, 0.007409959565848112f, 0.0076172188855707645f);
static const ai_u16 conv2d_44_t_out_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_44_t_out_0_shape_h_const_u16 = 4;

static const ai_u16 conv2d_45_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_45_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_45_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_45_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_45_t_in_0_shape_ch_const_u16 = 192;
static const ai_u16 conv2d_45_t_out_0_shape_ch_const_u16 = 32;
static const ai_i8 conv2d_45_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_45_t_out_0_fmt_zero_const_s8 = -6;
static const ai_float conv2d_45_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_45_t_out_0_fmt_scale_const_f32 = 0.0738293007016182f;
static const ai_float conv2d_45_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00290391081944108f, 0.0028458500746637583f, 0.0027553194668143988f, 0.0026186571922153234f, 0.0021124344784766436f, 0.0021543276961892843f, 0.0018908422207459807f, 0.001943154027685523f, 0.0016897267196327448f, 0.0021422815043479204f, 0.0025582981761544943f, 0.0021262774243950844f, 0.0020617053378373384f, 0.002012675628066063f, 0.0033953795209527016f, 0.0019349672365933657f, 0.002406628802418709f, 0.0020043777767568827f, 0.0027978913858532906f, 0.0017406245460733771f, 0.002573989098891616f, 0.0014903999399393797f, 0.0020730351097881794f, 0.002141474513337016f, 0.0015262860106304288f, 0.0019787338096648455f, 0.0022361513692885637f, 0.0023359188344329596f, 0.0016979409847408533f, 0.0019436165457591414f, 0.002069193869829178f, 0.0013056033058091998f);
static const ai_layer_format_type conv2d_45_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;




static const ai_u16 conv2d_50_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_50_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_50_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_50_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_50_t_in_0_shape_ch_const_u16 = 32;
static const ai_u16 conv2d_50_t_out_0_shape_ch_const_u16 = 192;
static const ai_i8 conv2d_50_t_in_0_fmt_zero_const_s8 = -6;
static const ai_i8 conv2d_50_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_50_t_in_0_fmt_scale_const_f32 = 0.0738293007016182f;
static const ai_float conv2d_50_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_50_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.001717155915684998f, 0.0021867849864065647f, 0.0011557029793038964f, 0.0013143125688657165f, 0.0012394473887979984f, 0.0015208115801215172f, 0.0012670893920585513f, 0.0014043764676898718f, 0.0018245340324938297f, 0.0015447997720912099f, 0.0015759014058858156f, 0.0010586861753836274f, 0.002006473485380411f, 0.0020147988107055426f, 0.002021975815296173f, 0.002025581430643797f, 0.0025224662385880947f, 0.0024464651942253113f, 0.0016860743053257465f, 0.002155856229364872f, 0.0016819299198687077f, 0.0009977533482015133f, 0.001064663752913475f, 0.0015908489003777504f, 0.0018310991581529379f, 0.0013654744252562523f, 0.0017554981168359518f, 0.0022051059640944004f, 0.002373010851442814f, 0.0020126947201788425f, 0.002161912154406309f, 0.0016355984844267368f, 0.0010353197576478124f, 0.0017202278831973672f, 0.002167115919291973f, 0.0010867678793147206f, 0.0012528834631666541f, 0.0026246230117976665f, 0.0033625271171331406f, 0.001796378055587411f, 0.002049028407782316f, 0.000927860732190311f, 0.0017095647053793073f, 0.0013501659268513322f, 0.0024453378282487392f, 0.00123379856813699f, 0.0017509674653410912f, 0.0016524131642654538f, 0.0012536029098555446f, 0.0005835822666995227f, 0.001819969736970961f, 0.0024839455727487803f, 0.0021415026858448982f, 0.0017890784656628966f, 0.002150020794942975f, 0.00223868852481246f, 0.0014963395660743117f, 0.002438996685668826f, 0.0016141539672389627f, 0.0010701800929382443f, 0.0016373938415199518f, 0.001432408345863223f, 0.0016012303531169891f, 0.0025764971505850554f, 0.0020781683269888163f, 0.0016940630739554763f, 0.0019242586567997932f, 0.0015704893739894032f, 0.0007909224950708449f, 0.0011940927943214774f, 0.0012886535841971636f, 0.0020308406092226505f, 0.0024836177472025156f, 0.001779986429028213f, 0.0015494519611820579f, 0.0020723578054457903f, 0.0016381342429667711f, 0.001556968316435814f, 0.0027256058529019356f, 0.0013032767456024885f, 0.0011426924029365182f, 0.0017142848810181022f, 0.0014534967485815287f, 0.00115868984721601f, 0.0013892636634409428f, 0.0018051284132525325f, 0.0012018259149044752f, 0.001856376649811864f, 0.0017709707608446479f, 0.000572859775274992f, 0.0013214405626058578f, 0.0014442357933148742f, 0.002778131514787674f, 0.00219710823148489f, 0.0013796082930639386f, 0.0014539072290062904f, 0.001490637892857194f, 0.0026353078428655863f, 0.0012893551029264927f, 0.0014381257351487875f, 0.0023251587990671396f, 0.0012843729928135872f, 0.0010485759703442454f, 0.0014192871749401093f, 0.001548880129121244f, 0.0008027528529055417f, 0.0020872773602604866f, 0.0021086381748318672f, 0.0013603107072412968f, 0.0014498640084639192f, 0.0011607662308961153f, 0.0016237820964306593f, 0.001549362437799573f, 0.002127666026353836f, 0.001599836745299399f, 0.002625193214043975f, 0.002401885809376836f, 0.0018587304512038827f, 0.0017801137873902917f, 0.0011546805035322905f, 0.0015433556400239468f, 0.0012739303056150675f, 0.0016006563091650605f, 0.0015052336966618896f, 0.0014797428157180548f, 0.0015277457423508167f, 0.001801730366423726f, 0.0014802718069404364f, 0.0014031000901013613f, 0.002212594263255596f, 0.0007134107290767133f, 0.0010547938290983438f, 0.0009889854118227959f, 0.002211905550211668f, 0.001432600780390203f, 0.0015110780950635672f, 0.001046686782501638f, 0.0014433151809498668f, 0.00179318489972502f, 0.001675735809840262f, 0.0010460198391228914f, 0.0008593366947025061f, 0.0015334963100031018f, 0.00157270731870085f, 0.0015119650634005666f, 0.001621863222680986f, 0.001793296542018652f, 0.0021628225222229958f, 0.0011598676210269332f, 0.0018426199676468968f, 0.00213515548966825f, 0.0014618977438658476f, 0.001966326730325818f, 0.001389912678860128f, 0.0015672689769417048f, 0.0019478128524497151f, 0.0019756087567657232f, 0.0009275121265091002f, 0.002528285840526223f, 0.002695586532354355f, 0.0018594044959172606f, 0.0020137906540185213f, 0.0020913025364279747f, 0.001356043154373765f, 0.001689509372226894f, 0.0011785579845309258f, 0.0015835530357435346f, 0.0007997470092959702f, 0.0011133843800053f, 0.0013078657211735845f, 0.0016192212933674455f, 0.0020763431675732136f, 0.0022578276693820953f, 0.001277376664802432f, 0.0011686801444739103f, 0.0016664931317791343f, 0.0022253310307860374f, 0.0021535020787268877f, 0.0014094564830884337f, 0.00238803937099874f, 0.002016379265114665f, 0.0021734475158154964f, 0.002745198318734765f, 0.0024051887448877096f, 0.0022324766032397747f, 0.00198186794295907f, 0.0016083690570667386f, 0.0014310062397271395f, 0.0016909762052819133f, 0.0010887846583500504f, 0.0011457729851827025f, 0.0016520910430699587f);
static const ai_layer_format_type conv2d_50_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_51_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_51_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_51_pad_before_t_in_0_shape_h_const_u32 = 4;

static const ai_u16 conv2d_51_t_in_0_shape_w_const_u16 = 6;
static const ai_u16 conv2d_51_t_in_0_shape_h_const_u16 = 6;
static const ai_u16 conv2d_51_t_in_0_shape_ch_const_u16 = 192;
static const ai_u16 conv2d_51_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_51_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_51_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_51_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_51_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_51_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_51_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.006056672893464565f, 0.008839478716254234f, 0.00860364455729723f, 0.010002645663917065f, 0.013214210979640484f, 0.008050378412008286f, 0.008666212670505047f, 0.015307511202991009f, 0.006741333287209272f, 0.008376901969313622f, 0.005848554894328117f, 0.019272606819868088f, 0.005240736063569784f, 0.005759652704000473f, 0.007238051388412714f, 0.0073838126845657825f, 0.00501085352152586f, 0.005444529000669718f, 0.012449757196009159f, 0.010289705358445644f, 0.011152496561408043f, 0.008554793894290924f, 0.007547978311777115f, 0.005721664987504482f, 0.008027372881770134f, 0.008318005129694939f, 0.009204275906085968f, 0.005936675239354372f, 0.008648986928164959f, 0.0042487154714763165f, 0.015256302431225777f, 0.008124020881950855f, 0.008378096856176853f, 0.010372384451329708f, 0.0068783508613705635f, 0.006072844844311476f, 0.010898410342633724f, 0.006245271768420935f, 0.005261403042823076f, 0.0163415540009737f, 0.007629439700394869f, 0.016407489776611328f, 0.01274411752820015f, 0.0075797573663294315f, 0.012749035842716694f, 0.01792152412235737f, 0.013621116988360882f, 0.009057440795004368f, 0.007280342280864716f, 0.01560908742249012f, 0.007429888471961021f, 0.0058187153190374374f, 0.005052688531577587f, 0.011709105223417282f, 0.008415461517870426f, 0.007364760618656874f, 0.006874886807054281f, 0.005034248810261488f, 0.009792945347726345f, 0.010780198499560356f, 0.0060157920233905315f, 0.012149767018854618f, 0.012225319631397724f, 0.008338619954884052f, 0.009548284113407135f, 0.00513039343059063f, 0.008704178035259247f, 0.018066363409161568f, 0.012025834061205387f, 0.016643641516566277f, 0.01568002812564373f, 0.005541826598346233f, 0.005772114265710115f, 0.0059317611157894135f, 0.012927563861012459f, 0.008551284670829773f, 0.008308996446430683f, 0.00632643885910511f, 0.01026217546314001f, 0.0050664194859564304f, 0.010531512089073658f, 0.01091422326862812f, 0.012985282577574253f, 0.009458386339247227f, 0.007515843026340008f, 0.006532940082252026f, 0.008563341572880745f, 0.007799284532666206f, 0.015279017388820648f, 0.009926971048116684f, 0.01216958463191986f, 0.008812151849269867f, 0.008918831124901772f, 0.006311171688139439f, 0.011657290160655975f, 0.005962858442217112f, 0.014536387287080288f, 0.005365665070712566f, 0.007557974196970463f, 0.009279989637434483f, 0.004075409844517708f, 0.020115336403250694f, 0.015967341139912605f, 0.010104252956807613f, 0.009483092464506626f, 0.02415880188345909f, 0.00839168205857277f, 0.010909496806561947f, 0.004118886776268482f, 0.011709602549672127f, 0.006814883556216955f, 0.01124269887804985f, 0.00726888794451952f, 0.005635442212224007f, 0.007855450734496117f, 0.005788876675069332f, 0.010140746831893921f, 0.007164544425904751f, 0.0033103825990110636f, 0.010043847374618053f, 0.006094036158174276f, 0.008396483026444912f, 0.003836707677692175f, 0.008984286338090897f, 0.005359047558158636f, 0.0069753434509038925f, 0.004828990437090397f, 0.01793954148888588f, 0.008722389116883278f, 0.006513134576380253f, 0.019933203235268593f, 0.01485312171280384f, 0.008463367819786072f, 0.006430268753319979f, 0.01266707107424736f, 0.010925661772489548f, 0.015204089693725109f, 0.00778098264709115f, 0.008139883168041706f, 0.006407917942851782f, 0.011182109825313091f, 0.009386783465743065f, 0.00716759916394949f, 0.008639123290777206f, 0.008568602614104748f, 0.011322710663080215f, 0.005556646268814802f, 0.0075767757371068f, 0.01153427641838789f, 0.006761046126484871f, 0.007117504719644785f, 0.010375406593084335f, 0.006983906961977482f, 0.0120160561054945f, 0.009123602882027626f, 0.006239835172891617f, 0.006403408944606781f, 0.022677432745695114f, 0.008136245422065258f, 0.00709488382562995f, 0.00641850009560585f, 0.009574241936206818f, 0.00731933768838644f, 0.014731324277818203f, 0.006246855016797781f, 0.011216382496058941f, 0.0184771865606308f, 0.009753508493304253f, 0.011249353177845478f, 0.017054304480552673f, 0.007228625938296318f, 0.0094937514513731f, 0.005959377158433199f, 0.016811853274703026f, 0.013684317469596863f, 0.008262358605861664f, 0.006377468351274729f, 0.005378589034080505f, 0.009208614006638527f, 0.005887795239686966f, 0.006512760184705257f, 0.010880046524107456f, 0.0077847568318247795f, 0.015332674607634544f, 0.006382382940500975f, 0.010126426815986633f, 0.008174173533916473f, 0.006409661378711462f, 0.012386911548674107f, 0.006462401244789362f, 0.013362805359065533f, 0.007014947943389416f);
static const ai_u16 conv2d_51_t_out_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_51_t_out_0_shape_h_const_u16 = 4;

static const ai_u16 conv2d_52_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_52_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_52_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_52_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_52_t_in_0_shape_ch_const_u16 = 192;
static const ai_u16 conv2d_52_t_out_0_shape_ch_const_u16 = 32;
static const ai_i8 conv2d_52_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_52_t_out_0_fmt_zero_const_s8 = 8;
static const ai_float conv2d_52_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_52_t_out_0_fmt_scale_const_f32 = 0.09495817869901657f;
static const ai_float conv2d_52_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0022588663268834352f, 0.003011684864759445f, 0.0019086392130702734f, 0.0015430045314133167f, 0.001918684458360076f, 0.002083113417029381f, 0.0033226830419152975f, 0.0022376978304237127f, 0.0022573405876755714f, 0.0032693257089704275f, 0.0019227333832532167f, 0.0020781198982149363f, 0.002846386516466737f, 0.002298529725521803f, 0.002956666285172105f, 0.003756848629564047f, 0.0028335945680737495f, 0.002384222811087966f, 0.002183819655328989f, 0.001631156075745821f, 0.0022337441332638264f, 0.00225075869821012f, 0.0019909325055778027f, 0.004118745680898428f, 0.0030661949422210455f, 0.002138259122148156f, 0.0019524861127138138f, 0.0019598675426095724f, 0.0016611734172329307f, 0.002152165863662958f, 0.0028826340567320585f, 0.001774346805177629f);
static const ai_layer_format_type conv2d_52_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_54_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_54_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_54_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_54_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_54_t_in_0_shape_ch_const_u16 = 32;
static const ai_u16 conv2d_54_t_out_0_shape_ch_const_u16 = 192;
static const ai_i8 conv2d_54_t_in_0_fmt_zero_const_s8 = 8;
static const ai_i8 conv2d_54_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_54_t_in_0_fmt_scale_const_f32 = 0.09495817869901657f;
static const ai_float conv2d_54_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_54_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0014885038835927844f, 0.0015728889266029f, 0.0006506554200313985f, 0.0008419066434726119f, 0.0013942902442067862f, 0.001730117597617209f, 0.0018021368887275457f, 0.0019693905487656593f, 0.0014550318010151386f, 0.0017899031518027186f, 0.0017111693741753697f, 0.0017649622168391943f, 0.0012040288420394063f, 0.0010356734273955226f, 0.0008446372230537236f, 0.0017429607687518f, 0.0022322814911603928f, 0.0013442190829664469f, 0.0006501974421553314f, 0.0011448588920757174f, 0.001357142929919064f, 0.0009657348273321986f, 0.0018447330221533775f, 0.0009526588255539536f, 0.001225725864060223f, 0.0011760812485590577f, 0.0009198338375426829f, 0.0007269073976203799f, 0.0015049566281959414f, 0.0012535882415249944f, 0.002076679840683937f, 0.0019803736358880997f, 0.0011085843434557319f, 0.0009047519415616989f, 0.0011766721727326512f, 0.0009837409015744925f, 0.001553419861011207f, 0.0016016740119084716f, 0.001207338529638946f, 0.0018413860816508532f, 0.0011163641465827823f, 0.001325862598605454f, 0.001682662172242999f, 0.0019170072628185153f, 0.0008784202509559691f, 0.001010232255794108f, 0.0015239448985084891f, 0.0010900930501520634f, 0.0013647277373820543f, 0.0011934706708416343f, 0.0015799712855368853f, 0.0013478525215759873f, 0.0015180229675024748f, 0.0015447235200554132f, 0.0009575978619977832f, 0.0012165458174422383f, 0.0018560751341283321f, 0.0013616244541481137f, 0.0022524220403283834f, 0.0012253017630428076f, 0.0020657326094806194f, 0.0014171225484460592f, 0.001715754740871489f, 0.0015640691854059696f, 0.0019100928911939263f, 0.0016159192891791463f, 0.0013097648043185472f, 0.0016359094297513366f, 0.0008844974800013006f, 0.0010988990543410182f, 0.0017301914049312472f, 0.0024054660461843014f, 0.0023814779706299305f, 0.0015910568181425333f, 0.0013268576003611088f, 0.0011735602747648954f, 0.0010303454473614693f, 0.0017734160646796227f, 0.0019077134784311056f, 0.0010205418802797794f, 0.0017961956327781081f, 0.0014632872771471739f, 0.0009575755102559924f, 0.0012297596549615264f, 0.0022973765153437853f, 0.001206926186569035f, 0.0012127377558499575f, 0.0014506956795230508f, 0.0016253958456218243f, 0.0014739077305421233f, 0.0008548401528969407f, 0.00045180495362728834f, 0.0014750405680388212f, 0.0015528578078374267f, 0.0014050465542823076f, 0.0014840456424281001f, 0.0011234787525609136f, 0.0011505252914503217f, 0.0016505136154592037f, 0.0010341024026274681f, 0.001168836490251124f, 0.0016915063606575131f, 0.0014365063980221748f, 0.0011673850240185857f, 0.001389133045449853f, 0.0013453986030071974f, 0.0013871998526155949f, 0.0012724158586934209f, 0.0009802434360608459f, 0.0011795572936534882f, 0.0018076705746352673f, 0.0009632398141548038f, 0.0012991582043468952f, 0.0011845249682664871f, 0.00179267767816782f, 0.0011411586310714483f, 0.0014680941822007298f, 0.001288343220949173f, 0.0010353818070143461f, 0.0010493263835087419f, 0.0017643222818151116f, 0.001225845655426383f, 0.0013180627720430493f, 0.0013551275478675961f, 0.001187906484119594f, 0.0015827062306925654f, 0.001232293201610446f, 0.0013997406931594014f, 0.0007559082005172968f, 0.0012548071099445224f, 0.0021884841844439507f, 0.0014359598280861974f, 0.001645104493945837f, 0.0012388700852170587f, 0.0014121255371719599f, 0.0007862185593694448f, 0.002185139339417219f, 0.0018627047538757324f, 0.0010844393400475383f, 0.0017454889602959156f, 0.0013141947565600276f, 0.001414358732290566f, 0.0008832152234390378f, 0.0009774091886356473f, 0.0017025605775415897f, 0.0018994893180206418f, 0.0005491701886057854f, 0.0013961407821625471f, 0.0017839978681877255f, 0.001053109299391508f, 0.001073826802894473f, 0.000656777760013938f, 0.0026340584736317396f, 0.0017273387638852f, 0.0013328123604878783f, 0.0011821665102615952f, 0.0016445644432678819f, 0.00112571578938514f, 0.0017322731437161565f, 0.0011382834054529667f, 0.0011971878120675683f, 0.0016716705868020654f, 0.0009405094315297902f, 0.0014410704607143998f, 0.0012020898284390569f, 0.001188861089758575f, 0.0016929747071117163f, 0.0010443776845932007f, 0.0015001975698396564f, 0.0011101642157882452f, 0.001245160703547299f, 0.0017645107582211494f, 0.001251708366908133f, 0.0011186938500031829f, 0.0015905477339401841f, 0.0011921667028218508f, 0.0014603374293074012f, 0.0035241444129496813f, 0.0009889273205772042f, 0.0014824059326201677f, 0.0009333408670499921f, 0.0012721158564090729f, 0.0016415042337030172f, 0.001236107898876071f, 0.001119299209676683f, 0.0017249600496143103f, 0.0015359933022409678f, 0.0007126557175070047f, 0.00166514259763062f, 0.001088678021915257f, 0.001146386843174696f, 0.0011110747000202537f);
static const ai_layer_format_type conv2d_54_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_55_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_55_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_55_pad_before_t_in_0_shape_h_const_u32 = 4;

static const ai_u16 conv2d_55_t_in_0_shape_w_const_u16 = 6;
static const ai_u16 conv2d_55_t_in_0_shape_h_const_u16 = 6;
static const ai_u16 conv2d_55_t_in_0_shape_ch_const_u16 = 192;
static const ai_u16 conv2d_55_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_55_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_55_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_55_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_55_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_55_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_55_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.007622335106134415f, 0.007519375998526812f, 0.006629342213273048f, 0.015722157433629036f, 0.010540831834077835f, 0.004874242004007101f, 0.008578035049140453f, 0.005947005935013294f, 0.005632992833852768f, 0.006065181456506252f, 0.007429075893014669f, 0.010492957197129726f, 0.0068553537130355835f, 0.00589723838493228f, 0.012645810842514038f, 0.006116167642176151f, 0.007528217975050211f, 0.005875417497009039f, 0.014567575417459011f, 0.008115959353744984f, 0.004670195747166872f, 0.01480915304273367f, 0.006759470794349909f, 0.015040997415781021f, 0.00791905727237463f, 0.008663791231811047f, 0.014654603786766529f, 0.00752973509952426f, 0.0062361652962863445f, 0.008659209124743938f, 0.009233531542122364f, 0.004533933009952307f, 0.008963624015450478f, 0.012086634524166584f, 0.0070808143354952335f, 0.009816338308155537f, 0.007046690210700035f, 0.012094956822693348f, 0.00982683151960373f, 0.009796297177672386f, 0.010599445551633835f, 0.0079102274030447f, 0.007841169834136963f, 0.010966929607093334f, 0.012358050793409348f, 0.01560888346284628f, 0.01691557839512825f, 0.009730233810842037f, 0.01277926191687584f, 0.012157663702964783f, 0.004847677890211344f, 0.011451356112957f, 0.009113630279898643f, 0.00515622366219759f, 0.007382620591670275f, 0.005509966518729925f, 0.006163872312754393f, 0.007167876698076725f, 0.007159321568906307f, 0.006908481940627098f, 0.00830801296979189f, 0.006486530881375074f, 0.008045834489166737f, 0.011974194087088108f, 0.004853744525462389f, 0.004379935562610626f, 0.005370213184505701f, 0.004603497684001923f, 0.007365941535681486f, 0.017984021455049515f, 0.009427754208445549f, 0.008436569944024086f, 0.011216321960091591f, 0.005787839647382498f, 0.010063344612717628f, 0.00913531519472599f, 0.00822441652417183f, 0.008883173577487469f, 0.006395547650754452f, 0.007778954692184925f, 0.006776636466383934f, 0.005443486385047436f, 0.007905337028205395f, 0.011289136484265327f, 0.00605618953704834f, 0.007245234213769436f, 0.006980776321142912f, 0.0049279434606432915f, 0.007881381548941135f, 0.007222043816000223f, 0.01012231595814228f, 0.009283149614930153f, 0.010018324479460716f, 0.008700007572770119f, 0.00735436612740159f, 0.007318488322198391f, 0.011084460653364658f, 0.008093751035630703f, 0.006619696039706469f, 0.011307978071272373f, 0.007861537858843803f, 0.011217128485441208f, 0.00899689830839634f, 0.010737065225839615f, 0.005955792497843504f, 0.008004064671695232f, 0.006485361140221357f, 0.01486460492014885f, 0.012989485636353493f, 0.006974737159907818f, 0.0036788834258913994f, 0.009912623092532158f, 0.006449948530644178f, 0.00674816919490695f, 0.008646052330732346f, 0.008369148708879948f, 0.005595765542238951f, 0.012610708363354206f, 0.008888699114322662f, 0.014616108499467373f, 0.007246280089020729f, 0.0070853568613529205f, 0.00774121331050992f, 0.006547768600285053f, 0.005221405532211065f, 0.004445724654942751f, 0.007358099799603224f, 0.004768539685755968f, 0.0072297933511435986f, 0.004718162585049868f, 0.005540360230952501f, 0.007790371775627136f, 0.00788566842675209f, 0.006063347682356834f, 0.004863487556576729f, 0.007597262505441904f, 0.0062272436916828156f, 0.009209639392793179f, 0.009491976350545883f, 0.00626404769718647f, 0.005188531242311001f, 0.010110052302479744f, 0.008323432877659798f, 0.008627128787338734f, 0.006480053067207336f, 0.005690083373337984f, 0.006902630440890789f, 0.009067407809197903f, 0.005874225869774818f, 0.015586882829666138f, 0.008502606302499771f, 0.009453988634049892f, 0.012547687627375126f, 0.007612130604684353f, 0.00782166887074709f, 0.013310681097209454f, 0.008993099443614483f, 0.010789602994918823f, 0.00451535452157259f, 0.007048213388770819f, 0.005540250334888697f, 0.008186661638319492f, 0.009222225286066532f, 0.017615700140595436f, 0.009781977161765099f, 0.009237811900675297f, 0.004789772443473339f, 0.009909436106681824f, 0.010163616389036179f, 0.00816357508301735f, 0.007668222766369581f, 0.005163359455764294f, 0.008189824409782887f, 0.013784689828753471f, 0.006970708258450031f, 0.010533774271607399f, 0.008187055587768555f, 0.007237807381898165f, 0.010324041359126568f, 0.006258659530431032f, 0.014206837862730026f, 0.009040839038789272f, 0.008491193875670433f, 0.012836792506277561f, 0.00859754253178835f, 0.005139006767421961f, 0.0047231935895979404f, 0.00826898030936718f, 0.0034806448966264725f, 0.007996580563485622f, 0.007468817755579948f, 0.0056949094869196415f);
static const ai_u16 conv2d_55_t_out_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_55_t_out_0_shape_h_const_u16 = 4;

static const ai_u16 conv2d_56_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_56_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_56_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_56_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_56_t_in_0_shape_ch_const_u16 = 192;
static const ai_u16 conv2d_56_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 conv2d_56_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_56_t_out_0_fmt_zero_const_s8 = -6;
static const ai_float conv2d_56_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_56_t_out_0_fmt_scale_const_f32 = 0.04048856720328331f;
static const ai_float conv2d_56_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.002916534896939993f, 0.002232600701972842f, 0.0024571730755269527f, 0.0027203212957829237f, 0.0024747555144131184f, 0.0020022958051413298f, 0.0014310304541140795f, 0.001861044205725193f, 0.0014902731636539102f, 0.0014844818506389856f, 0.0016579808434471488f, 0.0017024959670379758f, 0.0013831929536536336f, 0.0024290261790156364f, 0.0015358516247943044f, 0.0015526770148426294f, 0.0011815775651484728f, 0.001277961884625256f, 0.0014831805601716042f, 0.001809768145903945f, 0.001105150324292481f, 0.0016984461108222604f, 0.0016801602905616164f, 0.001691407524049282f, 0.0014980084961280227f, 0.001480072271078825f, 0.0019444250501692295f, 0.0011836796766147017f, 0.0016952469013631344f, 0.001570196240209043f, 0.0014965968439355493f, 0.0017570268828421831f, 0.0015968111110851169f, 0.0018574902787804604f, 0.0015987433725968003f, 0.0017801980720832944f, 0.001615344313904643f, 0.0014255550922825933f, 0.0018143522320315242f, 0.0017495601205155253f, 0.002409678651019931f, 0.0021655254531651735f, 0.002040889812633395f, 0.0012528309598565102f, 0.0015907292254269123f, 0.0014251606771722436f, 0.0015509544173255563f, 0.0016785124316811562f);
static const ai_layer_format_type conv2d_56_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;



static const ai_u16 conv2d_60_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_60_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_60_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_60_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_60_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 conv2d_60_t_out_0_shape_ch_const_u16 = 288;
static const ai_i8 conv2d_60_t_in_0_fmt_zero_const_s8 = -6;
static const ai_i8 conv2d_60_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_60_t_in_0_fmt_scale_const_f32 = 0.04048856720328331f;
static const ai_float conv2d_60_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_60_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.002003439934924245f, 0.004592029843479395f, 0.002530902624130249f, 0.0020840393844991922f, 0.001694746082648635f, 0.0020049167796969414f, 0.0025325664319097996f, 0.0027698350604623556f, 0.0026355909649282694f, 0.0022192457690835f, 0.003482747357338667f, 0.001997358864173293f, 0.001737675629556179f, 0.0020685538183897734f, 0.0027558377478271723f, 0.003081750823184848f, 0.0012754304334521294f, 0.0016766528133302927f, 0.0025971103459596634f, 0.002770398510619998f, 0.0033189267851412296f, 0.001611285493709147f, 0.003932026214897633f, 0.0018283688696101308f, 0.0018907507183030248f, 0.0022908179089426994f, 0.0034777068067342043f, 0.0024539348669350147f, 0.00348341534845531f, 0.002655829070135951f, 0.0013709503691643476f, 0.0020878519862890244f, 0.002247110242024064f, 0.003170913318172097f, 0.002237545559182763f, 0.00219001155346632f, 0.0027972508687525988f, 0.0021930986549705267f, 0.0037289566826075315f, 0.002322141546756029f, 0.0031247269362211227f, 0.0029540089890360832f, 0.003587127896025777f, 0.003234119853004813f, 0.0035261414013803005f, 0.0027280934154987335f, 0.0017523624701425433f, 0.0029108531307429075f, 0.0021400447003543377f, 0.0035143736749887466f, 0.0017342426581308246f, 0.0028479257598519325f, 0.0008828670834191144f, 0.0022529882844537497f, 0.00233746157027781f, 0.0030207023955881596f, 0.0023313311394304037f, 0.0036478694528341293f, 0.0025238243397325277f, 0.0023751696571707726f, 0.002738956129178405f, 0.0018732377793639898f, 0.0021741457749158144f, 0.0023878489155322313f, 0.003157914150506258f, 0.0020766074303537607f, 0.0022689690813422203f, 0.002251159865409136f, 0.003444817615672946f, 0.0019061360508203506f, 0.003216719953343272f, 0.0025702668353915215f, 0.0027187003288418055f, 0.0030955253168940544f, 0.003866614541038871f, 0.002521076239645481f, 0.0024327177088707685f, 0.001368529861792922f, 0.001661455724388361f, 0.002299721585586667f, 0.0024295032490044832f, 0.001633969834074378f, 0.004017188213765621f, 0.0020807133987545967f, 0.0024136914871633053f, 0.0024873889051377773f, 0.001633626176044345f, 0.0015952300745993853f, 0.0016835130518302321f, 0.0021182952914386988f, 0.0016917421016842127f, 0.002487121382728219f, 0.003228901419788599f, 0.0034636210184544325f, 0.0019272627541795373f, 0.0017510990146547556f, 0.0023809594567865133f, 0.0033713141456246376f, 0.00221754377707839f, 0.003308146493509412f, 0.005416522733867168f, 0.0024585670325905085f, 0.002862379187718034f, 0.002280475338920951f, 0.0031137263868004084f, 0.004203341901302338f, 0.001511839684098959f, 0.0025936660822480917f, 0.002514526480808854f, 0.0029834213200956583f, 0.003529477631673217f, 0.0018886260222643614f, 0.004585614427924156f, 0.0030485258903354406f, 0.0037037732545286417f, 0.0024257893674075603f, 0.0022439854219555855f, 0.001689453492872417f, 0.0026610265485942364f, 0.0022544036619365215f, 0.001995961181819439f, 0.00226981402374804f, 0.0024360406678169966f, 0.0021961950697004795f, 0.00342592503875494f, 0.0022633050102740526f, 0.002451273612678051f, 0.001961573725566268f, 0.0019236367661505938f, 0.002280318411067128f, 0.002106262370944023f, 0.003519456833600998f, 0.0026249217335134745f, 0.002268563723191619f, 0.0013396177673712373f, 0.0015157980378717184f, 0.0015689192805439234f, 0.002594550373032689f, 0.0028315477538853884f, 0.0029278385918587446f, 0.0022938253823667765f, 0.0024363165721297264f, 0.0023492826148867607f, 0.0019392733229324222f, 0.0017498654779046774f, 0.0027570396196097136f, 0.004498242400586605f, 0.002833000151440501f, 0.0017636610427871346f, 0.0027489052154123783f, 0.0014768801629543304f, 0.0029386498499661684f, 0.0035737957805395126f, 0.0018070935038849711f, 0.0026503191329538822f, 0.0024529569782316685f, 0.002196833724156022f, 0.001976810162886977f, 0.0021416014060378075f, 0.0024081969168037176f, 0.0023484069388359785f, 0.0014576379908248782f, 0.0019803261384367943f, 0.0018520081648603082f, 0.002945592859759927f, 0.0025547631084918976f, 0.004634668584913015f, 0.0024985852651298046f, 0.002481906209141016f, 0.0040759011171758175f, 0.0031032052356749773f, 0.004298909101635218f, 0.0017332957359030843f, 0.0015109048690646887f, 0.002135099610313773f, 0.001750766416080296f, 0.00330700003542006f, 0.0025194406043738127f, 0.0025044267531484365f, 0.0016192991752177477f, 0.001953710336238146f, 0.0022481910418719053f, 0.0032534440979361534f, 0.0035175662487745285f, 0.0023753575515002012f, 0.0027587031945586205f, 0.0034575946629047394f, 0.0029777325689792633f, 0.003894105786457658f, 0.00213794456794858f, 0.0017389800632372499f, 0.002214728854596615f, 0.0026652889791876078f, 0.0018678598571568727f, 0.0023690834641456604f, 0.0017297202721238136f, 0.002313978038728237f, 0.00352215557359159f, 0.003135184058919549f, 0.00190539110917598f, 0.0018438247498124838f, 0.0028813390526920557f, 0.0021775916684418917f, 0.0031419158913195133f, 0.0024782083928585052f, 0.002523240400478244f, 0.001203856780193746f, 0.0040720487013459206f, 0.0018740376690402627f, 0.0030992315150797367f, 0.002307033631950617f, 0.002816525287926197f, 0.002627880312502384f, 0.0016587937716394663f, 0.0022855724673718214f, 0.0018457670230418444f, 0.0019982794765383005f, 0.0012913133250549436f, 0.0018328068545088172f, 0.002264757640659809f, 0.002410542918369174f, 0.0029325245413929224f, 0.0021607549861073494f, 0.003423466579988599f, 0.002053050324320793f, 0.003639448434114456f, 0.002895235549658537f, 0.0028895363211631775f, 0.0033403567504137754f, 0.00272111757658422f, 0.002923453925177455f, 0.0016781412996351719f, 0.0028039461467415094f, 0.0027541853487491608f, 0.0013938990887254477f, 0.003118726424872875f, 0.003243005136027932f, 0.0030637364834547043f, 0.002026417525485158f, 0.002117146272212267f, 0.0019570274744182825f, 0.003294723341241479f, 0.001594293280504644f, 0.004331557080149651f, 0.0026491968892514706f, 0.002814294770359993f, 0.002606511814519763f, 0.0022588486317545176f, 0.001122796442359686f, 0.00259679788723588f, 0.00221966952085495f, 0.0031717929523438215f, 0.0021266944240778685f, 0.002454617293551564f, 0.00255891727283597f, 0.0025570914149284363f, 0.003800454083830118f, 0.0030889667104929686f, 0.002806989476084709f, 0.002406683284789324f, 0.002474068198353052f, 0.0019043482607230544f, 0.0013972908491268754f, 0.0027449754998087883f, 0.0011809917632490396f, 0.001980770844966173f, 0.0019830025266855955f, 0.002064055297523737f, 0.003184688976034522f, 0.003433837555348873f, 0.0025573240127414465f, 0.001944134826771915f, 0.002144309924915433f, 0.003080161986872554f, 0.003183322725817561f, 0.001383745577186346f, 0.001937940251082182f, 0.0026152017526328564f, 0.0025405592750757933f, 0.0033626919612288475f, 0.0025884739588946104f, 0.002304019406437874f, 0.0023888573050498962f, 0.0025776010006666183f, 0.002523789880797267f, 0.002251111203804612f, 0.002346278168261051f, 0.00225435639731586f);
static const ai_layer_format_type conv2d_60_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_61_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_61_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_61_pad_before_t_in_0_shape_h_const_u32 = 4;

static const ai_u16 conv2d_61_t_in_0_shape_w_const_u16 = 6;
static const ai_u16 conv2d_61_t_in_0_shape_h_const_u16 = 6;
static const ai_u16 conv2d_61_t_in_0_shape_ch_const_u16 = 288;
static const ai_u16 conv2d_61_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_61_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_61_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_61_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_61_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_61_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_61_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.013145986944437027f, 0.00485064135864377f, 0.0077256555669009686f, 0.007223422639071941f, 0.014679772779345512f, 0.009195336140692234f, 0.008132906630635262f, 0.006158880423754454f, 0.005424460396170616f, 0.007492667995393276f, 0.008255093358457088f, 0.00904793106019497f, 0.007844255305826664f, 0.008557177148759365f, 0.005857957061380148f, 0.0094341104850173f, 0.00868226308375597f, 0.008829043246805668f, 0.009064486250281334f, 0.00522629963234067f, 0.005531626287847757f, 0.00709763215854764f, 0.007223966531455517f, 0.008711506612598896f, 0.005747530609369278f, 0.00660735322162509f, 0.008932848460972309f, 0.007485657464712858f, 0.005732502788305283f, 0.01247384399175644f, 0.00721147283911705f, 0.02088231034576893f, 0.007146855350583792f, 0.004048801027238369f, 0.0171064306050539f, 0.014887620694935322f, 0.009930581785738468f, 0.022888047620654106f, 0.0056356522254645824f, 0.0073116314597427845f, 0.019025065004825592f, 0.00470218388363719f, 0.014403519220650196f, 0.006773815955966711f, 0.006118718534708023f, 0.007263259496539831f, 0.009834685362875462f, 0.006537752225995064f, 0.00489843962714076f, 0.0070588793605566025f, 0.0087618762627244f, 0.005138119217008352f, 0.02354762703180313f, 0.006665698252618313f, 0.006969986949115992f, 0.015474585816264153f, 0.0061147259548306465f, 0.00527117308229208f, 0.006721516139805317f, 0.008576120249927044f, 0.005428737495094538f, 0.010165463201701641f, 0.00546999229118228f, 0.009625451639294624f, 0.014438784681260586f, 0.00809991080313921f, 0.007392951287329197f, 0.010041531175374985f, 0.0033796357456594706f, 0.008197441697120667f, 0.012984639964997768f, 0.005982907488942146f, 0.012819860130548477f, 0.006529329810291529f, 0.0040085348300635815f, 0.007728063967078924f, 0.007385579869151115f, 0.0063009667210280895f, 0.016297785565257072f, 0.006725947372615337f, 0.007335025817155838f, 0.010173026472330093f, 0.00885601993650198f, 0.010351434350013733f, 0.00641148816794157f, 0.0067807710729539394f, 0.011196460574865341f, 0.01880020648241043f, 0.01043386198580265f, 0.010870643891394138f, 0.021206172183156013f, 0.010729867964982986f, 0.008387518115341663f, 0.00789633672684431f, 0.02462777867913246f, 0.011155491694808006f, 0.0044603305868804455f, 0.0032905908301472664f, 0.008056318387389183f, 0.008816484361886978f, 0.004761240910738707f, 0.006594989914447069f, 0.00959963258355856f, 0.005716837011277676f, 0.00465390644967556f, 0.005303284619003534f, 0.011031866073608398f, 0.006568026728928089f, 0.006157154217362404f, 0.007320656906813383f, 0.011147582903504372f, 0.00970664992928505f, 0.007632126100361347f, 0.006836383603513241f, 0.0053670573979616165f, 0.007234437856823206f, 0.005218710750341415f, 0.00694442680105567f, 0.007563129533082247f, 0.022260205820202827f, 0.009359986521303654f, 0.006977927405387163f, 0.011652196757495403f, 0.010148168541491032f, 0.006578496657311916f, 0.007896683178842068f, 0.006450686138123274f, 0.011069645173847675f, 0.008512177504599094f, 0.010446107015013695f, 0.011314325965940952f, 0.004144885111600161f, 0.009448889642953873f, 0.007726539857685566f, 0.027966903522610664f, 0.005670714657753706f, 0.01860656775534153f, 0.0066617270931601524f, 0.0061719841323792934f, 0.0062408423982560635f, 0.0051705678924918175f, 0.008431694470345974f, 0.008641956374049187f, 0.006709238048642874f, 0.01718830317258835f, 0.004215386230498552f, 0.0037409623619168997f, 0.011653619818389416f, 0.020494814962148666f, 0.005911672953516245f, 0.013367630541324615f, 0.007919742725789547f, 0.005086563993245363f, 0.007309891749173403f, 0.006841405760496855f, 0.01172195840626955f, 0.009141878224909306f, 0.012574692256748676f, 0.006107100751250982f, 0.008601143956184387f, 0.008528563193976879f, 0.011957596987485886f, 0.008988544344902039f, 0.00768861873075366f, 0.007151268422603607f, 0.009616782888770103f, 0.004442230798304081f, 0.014646104536950588f, 0.013967958278954029f, 0.0036934001836925745f, 0.005285755731165409f, 0.0038181173149496317f, 0.01776520162820816f, 0.009148691780865192f, 0.0126319769769907f, 0.014678476378321648f, 0.006096790079027414f, 0.010292922146618366f, 0.006630721036344767f, 0.01494022086262703f, 0.00728821475058794f, 0.012663263827562332f, 0.007123947609215975f, 0.005570549517869949f, 0.00752546451985836f, 0.0073049357160925865f, 0.010521827265620232f, 0.005450200289487839f, 0.006375965662300587f, 0.00788047257810831f, 0.014134160242974758f, 0.009634138084948063f, 0.007293473929166794f, 0.007647416554391384f, 0.013031630776822567f, 0.005909019149839878f, 0.008550739847123623f, 0.005194955505430698f, 0.011800006031990051f, 0.01127931009978056f, 0.005438636057078838f, 0.004137967713177204f, 0.008272870443761349f, 0.010545863769948483f, 0.010381870903074741f, 0.009330024011433125f, 0.029676567763090134f, 0.004566326271742582f, 0.008413128554821014f, 0.010686232708394527f, 0.012901926413178444f, 0.00623604841530323f, 0.008487030863761902f, 0.016756204888224602f, 0.008039946667850018f, 0.023344600573182106f, 0.014079750515520573f, 0.008077039383351803f, 0.010785031132400036f, 0.011091252788901329f, 0.006084624212235212f, 0.008163364604115486f, 0.0063065048307180405f, 0.007416837848722935f, 0.015200738795101643f, 0.004969369620084763f, 0.008317484520375729f, 0.019402822479605675f, 0.005405799485743046f, 0.008470248430967331f, 0.00776680139824748f, 0.00844528991729021f, 0.007583434693515301f, 0.004502074792981148f, 0.01804264634847641f, 0.010906814597547054f, 0.006396010052412748f, 0.004448466468602419f, 0.016802510246634483f, 0.010575382970273495f, 0.00514542730525136f, 0.008505169302225113f, 0.013353990390896797f, 0.00348862586542964f, 0.011146580800414085f, 0.009160705842077732f, 0.008125300519168377f, 0.016384433954954147f, 0.030200207605957985f, 0.010172363370656967f, 0.0167829692363739f, 0.006217408925294876f, 0.006553152110427618f, 0.005332236178219318f, 0.006450925953686237f, 0.007317142561078072f, 0.005122512113302946f, 0.011744001880288124f, 0.010358868166804314f, 0.00871958676725626f, 0.006108218338340521f, 0.013931900262832642f, 0.02475159801542759f, 0.007736429572105408f, 0.007852689363062382f, 0.010901279747486115f, 0.01225515641272068f, 0.016050012782216072f, 0.004834366962313652f, 0.010232610628008842f, 0.008212012238800526f, 0.010164567269384861f, 0.006123450584709644f, 0.006280060857534409f, 0.00537941325455904f, 0.0068782600574195385f, 0.02251252718269825f, 0.015596259385347366f, 0.007275816053152084f, 0.003692690283060074f, 0.00842206459492445f, 0.009991033934056759f, 0.007757183630019426f, 0.006793218199163675f, 0.0059490203857421875f, 0.011987859383225441f, 0.009245596826076508f, 0.008405259810388088f);
static const ai_u16 conv2d_61_t_out_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_61_t_out_0_shape_h_const_u16 = 4;

static const ai_u16 conv2d_62_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_62_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_62_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_62_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_62_t_in_0_shape_ch_const_u16 = 288;
static const ai_u16 conv2d_62_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 conv2d_62_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_62_t_out_0_fmt_zero_const_s8 = -3;
static const ai_float conv2d_62_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_62_t_out_0_fmt_scale_const_f32 = 0.05003214627504349f;
static const ai_float conv2d_62_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0018853233195841312f, 0.002850206336006522f, 0.0023209168575704098f, 0.002447610953822732f, 0.0025073892902582884f, 0.002741696545854211f, 0.00239964434877038f, 0.0013828924857079983f, 0.001419689622707665f, 0.001363790244795382f, 0.0011723918141797185f, 0.0009748070151545107f, 0.001570547348819673f, 0.0013118620263412595f, 0.0015805026050657034f, 0.0010838580783456564f, 0.0015191156417131424f, 0.0015543580520898104f, 0.001359036541543901f, 0.001500116428360343f, 0.0014133421937003732f, 0.0014055160572752357f, 0.0013885640073567629f, 0.0015626561362296343f, 0.001445096917450428f, 0.0015432117506861687f, 0.0016246538143604994f, 0.00157394097186625f, 0.0015341024845838547f, 0.0014066108269616961f, 0.0015607608947902918f, 0.0016652934718877077f, 0.0010015243897214532f, 0.001717739854939282f, 0.0016539351781830192f, 0.0015825786394998431f, 0.0013813057448714972f, 0.0013160931412130594f, 0.0017912465846166015f, 0.0017633795505389571f, 0.0017366947140544653f, 0.0015018365811556578f, 0.0013326259795576334f, 0.0012548795202746987f, 0.0016335537657141685f, 0.001837188028730452f, 0.001162809319794178f, 0.0013648513704538345f);
static const ai_layer_format_type conv2d_62_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;




static const ai_u16 conv2d_67_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_67_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_67_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_67_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_67_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 conv2d_67_t_out_0_shape_ch_const_u16 = 288;
static const ai_i8 conv2d_67_t_in_0_fmt_zero_const_s8 = -3;
static const ai_i8 conv2d_67_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_67_t_in_0_fmt_scale_const_f32 = 0.05003214627504349f;
static const ai_float conv2d_67_t_out_0_fmt_scale_const_f32 = 0.021315094083547592f;
static const ai_float conv2d_67_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0019069722620770335f, 0.001958408858627081f, 0.001186078879982233f, 0.0013860843610018492f, 0.0011778161861002445f, 0.0027170302346348763f, 0.001956067280843854f, 0.002327296882867813f, 0.0013325344771146774f, 0.001771134091541171f, 0.0026482101529836655f, 0.0010003787465393543f, 0.0024313582107424736f, 0.0019802020397037268f, 0.0017855570185929537f, 0.0019721011631190777f, 0.0013286885805428028f, 0.0019913690630346537f, 0.002324693836271763f, 0.0015315655618906021f, 0.0013083673547953367f, 0.0019842172041535378f, 0.0012321190442889929f, 0.0014900733949616551f, 0.0011457770597189665f, 0.0012801304692402482f, 0.0027237245813012123f, 0.0014155516400933266f, 0.0017639059806242585f, 0.0012280261144042015f, 0.0017797785112634301f, 0.00112805119715631f, 0.0018488919595256448f, 0.0012830705381929874f, 0.0015251032309606671f, 0.0015620285412296653f, 0.0014744490617886186f, 0.001725984737277031f, 0.0017230239463970065f, 0.0016371285310015082f, 0.002075787400826812f, 0.0019377021817490458f, 0.002477280329912901f, 0.0009221949730999768f, 0.0014258844312280416f, 0.0018215496093034744f, 0.002201543189585209f, 0.002815169282257557f, 0.0011501857079565525f, 0.0020106364972889423f, 0.001991723896935582f, 0.0012025211472064257f, 0.001994708087295294f, 0.0015527575742453337f, 0.0021437571849673986f, 0.001619383692741394f, 0.0017455180641263723f, 0.0014165414031594992f, 0.002057228237390518f, 0.002301676431670785f, 0.0012953159166499972f, 0.0020790360867977142f, 0.002001004060730338f, 0.0010889519471675158f, 0.001750148250721395f, 0.0008239385788328946f, 0.0018568537198007107f, 0.0019415842834860086f, 0.0011744077783077955f, 0.0016126902773976326f, 0.0012419121339917183f, 0.0010097401682287455f, 0.0010907837422564626f, 0.001582532306201756f, 0.0015242589870467782f, 0.0018058656714856625f, 0.0013614053605124354f, 0.0016046351520344615f, 0.0012136106379330158f, 0.0017540014814585447f, 0.001125533482991159f, 0.0012162298662588f, 0.0014610296348109841f, 0.0012089414522051811f, 0.000787912926170975f, 0.0021251016296446323f, 0.0015228327829390764f, 0.001184927299618721f, 0.0021716596093028784f, 0.0014964289730414748f, 0.0024517325218766928f, 0.0014379924396052957f, 0.002123346319422126f, 0.001882172771729529f, 0.0022625690326094627f, 0.0016040574992075562f, 0.0020086674485355616f, 0.0021599309984594584f, 0.0024638473987579346f, 0.0015188171528279781f, 0.0013407012447714806f, 0.0017007042188197374f, 0.002369148191064596f, 0.0018610935658216476f, 0.0016840494936332107f, 0.0023047630675137043f, 0.0024218736216425896f, 0.0010822388576343656f, 0.0015705974074080586f, 0.0017981366254389286f, 0.0018877297407016158f, 0.0022638856898993254f, 0.0015428363112732768f, 0.0018540521850809455f, 0.0010603400878608227f, 0.0017352167051285505f, 0.002088424051180482f, 0.0010418703313916922f, 0.002235855208709836f, 0.0017588607734069228f, 0.0019902808126062155f, 0.0022306262981146574f, 0.0014367243275046349f, 0.002318692160770297f, 0.0015048326458781958f, 0.0016459781909361482f, 0.001970441546291113f, 0.0016951666912063956f, 0.0012263974640518427f, 0.0015574546996504068f, 0.0017429263098165393f, 0.00190040934830904f, 0.0018444580491632223f, 0.002332597505301237f, 0.002688034437596798f, 0.0014005090342834592f, 0.0016971700824797153f, 0.001582488534040749f, 0.0013602343387901783f, 0.0014938123058527708f, 0.0013959376374259591f, 0.0015871997456997633f, 0.0019319414859637618f, 0.0012807098682969809f, 0.0015702178934589028f, 0.0019044899381697178f, 0.0018313426990061998f, 0.0016332764644175768f, 0.0018368690507486463f, 0.001361976028420031f, 0.0016296858666464686f, 0.001523672486655414f, 0.0011555630480870605f, 0.0013260195264592767f, 0.0014796573668718338f, 0.0011871001916006207f, 0.0022770871873944998f, 0.0023999919649213552f, 0.0015944507904350758f, 0.0018977741710841656f, 0.0016495531890541315f, 0.0018718601204454899f, 0.00196131132543087f, 0.0018361675320193172f, 0.001412705285474658f, 0.001559761120006442f, 0.002144589787349105f, 0.0020832684822380543f, 0.0025314532686024904f, 0.0014951108023524284f, 0.0019949774723500013f, 0.0023384862579405308f, 0.0016267255414277315f, 0.0013867171946913004f, 0.0012394370278343558f, 0.0012729113223031163f, 0.0010383314220234752f, 0.0013911110581830144f, 0.0018335263011977077f, 0.0027518165297806263f, 0.0021691827569156885f, 0.001187662361189723f, 0.0031045919749885798f, 0.00185332668479532f, 0.001348853693343699f, 0.0022636763751506805f, 0.001980068162083626f, 0.001671990379691124f, 0.001748743117786944f, 0.001599848852492869f, 0.001386017887853086f, 0.0018124072812497616f, 0.0018167729722335935f, 0.0021836564410477877f, 0.0019391755340620875f, 0.0020692532416433096f, 0.0010076749604195356f, 0.0016118072671815753f, 0.0018008488696068525f, 0.001655317610129714f, 0.0018280951771885157f, 0.0010209769243374467f, 0.003530947957187891f, 0.0017622675513848662f, 0.0013880057958886027f, 0.0012393329525366426f, 0.0012914136750623584f, 0.0014827769482508302f, 0.0013425590004771948f, 0.0020581132266670465f, 0.0008723769569769502f, 0.002411898225545883f, 0.002002033405005932f, 0.0013656445080414414f, 0.002026944188401103f, 0.001279412885196507f, 0.0038211727514863014f, 0.0017699149902909994f, 0.002292710356414318f, 0.001822849502786994f, 0.0009632616420276463f, 0.0015827333554625511f, 0.0021331999450922012f, 0.0018213624134659767f, 0.0016112226294353604f, 0.002120860619470477f, 0.0017384093953296542f, 0.0015007673064246774f, 0.0018668537959456444f, 0.002681539161130786f, 0.0013043321669101715f, 0.00107511633541435f, 0.0007884969236329198f, 0.0011242071632295847f, 0.0010189998429268599f, 0.0012007574550807476f, 0.0009962304029613733f, 0.0032734274864196777f, 0.0026583364233374596f, 0.00147808447945863f, 0.0017833648016676307f, 0.0013925065286457539f, 0.0018309630686417222f, 0.0016261565033346415f, 0.0015716149937361479f, 0.002204892225563526f, 0.0019476681482046843f, 0.0018626950914040208f, 0.0014876011991873384f, 0.0015012611402198672f, 0.0017796432366594672f, 0.0013718338450416923f, 0.0016642942791804671f, 0.00179734593257308f, 0.0013499786145985126f, 0.0023919823579490185f, 0.001852227607741952f, 0.0012396381935104728f, 0.0016717419493943453f, 0.0021851458586752415f, 0.0016442456981167197f, 0.002078721532598138f, 0.0028245027642697096f, 0.001505363848991692f, 0.0012361793778836727f, 0.0015225396491587162f, 0.0013874429278075695f, 0.0031637961510568857f, 0.002319141523912549f, 0.0019869268871843815f, 0.0020900072995573282f, 0.0013880078913643956f, 0.0013620471581816673f, 0.0020419065840542316f, 0.0013960535870864987f, 0.0017127488972619176f, 0.0012765952851623297f, 0.001844886690378189f, 0.002128572901710868f, 0.002022130647674203f, 0.0016672934871166945f, 0.002004373585805297f, 0.0017312417039647698f, 0.0011353681329637766f, 0.001391500816680491f, 0.0010330667719244957f, 0.0015010578790679574f, 0.0011686217039823532f);
static const ai_layer_format_type conv2d_67_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_68_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_68_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_68_pad_before_t_in_0_shape_h_const_u32 = 4;

static const ai_u16 conv2d_68_t_in_0_shape_w_const_u16 = 6;
static const ai_u16 conv2d_68_t_in_0_shape_h_const_u16 = 6;
static const ai_u16 conv2d_68_t_in_0_shape_ch_const_u16 = 288;
static const ai_u16 conv2d_68_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_68_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_68_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_68_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_68_t_in_0_fmt_scale_const_f32 = 0.021315094083547592f;
static const ai_float conv2d_68_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_68_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.01569771021604538f, 0.00777113251388073f, 0.012908323667943478f, 0.012122219428420067f, 0.01849464140832424f, 0.00794792640954256f, 0.006048446986824274f, 0.005193318706005812f, 0.007715694140642881f, 0.01285572163760662f, 0.006589110940694809f, 0.03783109784126282f, 0.008051657117903233f, 0.009919748641550541f, 0.010274721309542656f, 0.004900272935628891f, 0.0071395873092114925f, 0.006813730113208294f, 0.009059411473572254f, 0.013609054498374462f, 0.012145410291850567f, 0.009706655517220497f, 0.013116958551108837f, 0.005023356061428785f, 0.010032298974692822f, 0.013295283541083336f, 0.00587317394092679f, 0.013158846646547318f, 0.007721183821558952f, 0.010969512164592743f, 0.008954495191574097f, 0.012777682393789291f, 0.011977148242294788f, 0.010803149081766605f, 0.01800188235938549f, 0.009810042567551136f, 0.020376095548272133f, 0.008154275827109814f, 0.009110515005886555f, 0.00903193186968565f, 0.008072448894381523f, 0.010258032009005547f, 0.0047594825737178326f, 0.01598004251718521f, 0.006576463580131531f, 0.004757635295391083f, 0.0061962357722222805f, 0.008138078264892101f, 0.019738351926207542f, 0.008028591051697731f, 0.008700862526893616f, 0.010411329567432404f, 0.006935533136129379f, 0.01985633745789528f, 0.0116861118003726f, 0.005415397230535746f, 0.01598196290433407f, 0.008720696903765202f, 0.004981746897101402f, 0.009393969550728798f, 0.011402803473174572f, 0.005179332569241524f, 0.009769922122359276f, 0.010988391935825348f, 0.009013486094772816f, 0.019221937283873558f, 0.011390251107513905f, 0.006339839659631252f, 0.009748250246047974f, 0.005090577527880669f, 0.008684123866260052f, 0.007610177621245384f, 0.00800994224846363f, 0.017935404554009438f, 0.00946072768419981f, 0.008465852588415146f, 0.018179429695010185f, 0.007568822707980871f, 0.018542146310210228f, 0.006579228211194277f, 0.006759085226804018f, 0.007576278876513243f, 0.015196137130260468f, 0.008546062745153904f, 0.012754101306200027f, 0.005251850932836533f, 0.006761619821190834f, 0.01796330139040947f, 0.010718528181314468f, 0.007680669892579317f, 0.0032989762257784605f, 0.009456540457904339f, 0.0052465046755969524f, 0.011740385554730892f, 0.005367717240005732f, 0.0058472249656915665f, 0.010044245980679989f, 0.005840257741510868f, 0.010501528158783913f, 0.00920669175684452f, 0.016038354486227036f, 0.008199777454137802f, 0.005902301985770464f, 0.009917632676661015f, 0.005442970898002386f, 0.007655210327357054f, 0.0037854197435081005f, 0.01752847619354725f, 0.0064635612070560455f, 0.009547989815473557f, 0.0072951605543494225f, 0.00406802911311388f, 0.012965633533895016f, 0.0058309524320065975f, 0.013251728378236294f, 0.007054474204778671f, 0.008478292264044285f, 0.009351846762001514f, 0.004270290024578571f, 0.008717907592654228f, 0.007926522754132748f, 0.008609374985098839f, 0.005510393064469099f, 0.006662278436124325f, 0.00652474258095026f, 0.01225172821432352f, 0.00580723537132144f, 0.0073641883209347725f, 0.007737900596112013f, 0.007848097942769527f, 0.005034883506596088f, 0.008395453914999962f, 0.013448855839669704f, 0.005953900050371885f, 0.004882436245679855f, 0.005046676844358444f, 0.008268548175692558f, 0.008313500322401524f, 0.007206047885119915f, 0.010983617976307869f, 0.008792132139205933f, 0.008361323736608028f, 0.004687383305281401f, 0.018390515819191933f, 0.010352144949138165f, 0.0073042879812419415f, 0.0070402780547738075f, 0.011748768389225006f, 0.005874320399016142f, 0.018193287774920464f, 0.009711976163089275f, 0.011646765284240246f, 0.031312283128499985f, 0.011010896414518356f, 0.0059444354847073555f, 0.020933538675308228f, 0.01040777750313282f, 0.004544047638773918f, 0.0060424827970564365f, 0.00930881593376398f, 0.009223934262990952f, 0.009327027946710587f, 0.01023007184267044f, 0.015329508110880852f, 0.01389678381383419f, 0.009005723521113396f, 0.005562940612435341f, 0.009623139165341854f, 0.005080962087959051f, 0.01425776444375515f, 0.010286272503435612f, 0.007127377670258284f, 0.010599704459309578f, 0.006011016666889191f, 0.016658678650856018f, 0.008722878061234951f, 0.009577671065926552f, 0.02022596076130867f, 0.006368977017700672f, 0.0032586469314992428f, 0.01005463395267725f, 0.021668709814548492f, 0.007395006250590086f, 0.006095934193581343f, 0.009215562604367733f, 0.005424981005489826f, 0.00590317090973258f, 0.007129940669983625f, 0.006208494305610657f, 0.010134599171578884f, 0.010914798825979233f, 0.011705632321536541f, 0.009422507137060165f, 0.005677947774529457f, 0.007837784476578236f, 0.005517059005796909f, 0.012583629228174686f, 0.0049663688987493515f, 0.004209148697555065f, 0.00785794761031866f, 0.0069424803368747234f, 0.012687926180660725f, 0.006090222392231226f, 0.013719526119530201f, 0.008512219414114952f, 0.015218977816402912f, 0.007495305500924587f, 0.007578405551612377f, 0.013437613844871521f, 0.0076213073916733265f, 0.018032865598797798f, 0.010087095201015472f, 0.00421174569055438f, 0.009251400828361511f, 0.003957567736506462f, 0.014442563988268375f, 0.005648870952427387f, 0.005175184924155474f, 0.0058786338195204735f, 0.009738351218402386f, 0.0139557383954525f, 0.006667677313089371f, 0.005979337263852358f, 0.006126148626208305f, 0.009871742688119411f, 0.00498112291097641f, 0.01115291379392147f, 0.005788409151136875f, 0.014550854451954365f, 0.006796903442591429f, 0.012792370282113552f, 0.016682513058185577f, 0.034051552414894104f, 0.011957148090004921f, 0.019800465553998947f, 0.01706370711326599f, 0.012573915533721447f, 0.008546669967472553f, 0.009807374328374863f, 0.00614335248246789f, 0.008034983649849892f, 0.011962353251874447f, 0.003769251750782132f, 0.013250105082988739f, 0.006671902257949114f, 0.005560701712965965f, 0.01610325090587139f, 0.006748069543391466f, 0.011949166655540466f, 0.01009626965969801f, 0.004798226989805698f, 0.00614188751205802f, 0.018584884703159332f, 0.004865197464823723f, 0.007277272641658783f, 0.008773664012551308f, 0.004822972230613232f, 0.014778914861381054f, 0.009065584279596806f, 0.008817861787974834f, 0.009874293580651283f, 0.005283680744469166f, 0.009656615555286407f, 0.0076222107745707035f, 0.013346529565751553f, 0.018293842673301697f, 0.009103170596063137f, 0.0046291411854326725f, 0.0075953188352286816f, 0.005252044182270765f, 0.011794437654316425f, 0.0053765238262712955f, 0.021697062999010086f, 0.009496576152741909f, 0.016825783997774124f, 0.010026288218796253f, 0.014180535450577736f, 0.02330932393670082f, 0.005778481252491474f, 0.008762959390878677f, 0.009074991568922997f, 0.005845863372087479f, 0.006972428876906633f, 0.016370967030525208f, 0.007403335999697447f, 0.018042515963315964f, 0.006853658240288496f, 0.012587434612214565f);
static const ai_u16 conv2d_68_t_out_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_68_t_out_0_shape_h_const_u16 = 4;

static const ai_u16 conv2d_69_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_69_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_69_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_69_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_69_t_in_0_shape_ch_const_u16 = 288;
static const ai_u16 conv2d_69_t_out_0_shape_ch_const_u16 = 48;
static const ai_i8 conv2d_69_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_69_t_out_0_fmt_zero_const_s8 = -5;
static const ai_float conv2d_69_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_69_t_out_0_fmt_scale_const_f32 = 0.05984853208065033f;
static const ai_float conv2d_69_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0015927506610751152f, 0.001374837476760149f, 0.0016544013051316142f, 0.0017380552599206567f, 0.0018565155332908034f, 0.0015961563913151622f, 0.0021063596941530704f, 0.00189470243640244f, 0.0018837966490536928f, 0.0026044417172670364f, 0.002051419345661998f, 0.0022766117472201586f, 0.002802618546411395f, 0.0013806319329887629f, 0.0015076275449246168f, 0.001488422160036862f, 0.0029215537942945957f, 0.0018641520291566849f, 0.002211889950558543f, 0.0014973119832575321f, 0.001762216561473906f, 0.0019627255387604237f, 0.0017953371861949563f, 0.0017209657235071063f, 0.0025469507090747356f, 0.0012049912475049496f, 0.0019582463428378105f, 0.0029038391076028347f, 0.0016585738630965352f, 0.0021587603259831667f, 0.0019265655428171158f, 0.0013654110953211784f, 0.0016155631747096777f, 0.0030191477853804827f, 0.0022909233812242746f, 0.0019081280333921313f, 0.0021556976716965437f, 0.0016829216619953513f, 0.0020331102423369884f, 0.0022459235042333603f, 0.0014165440807119012f, 0.0021637356840074062f, 0.0019522570073604584f, 0.0016931917052716017f, 0.0018799824174493551f, 0.0022505836095660925f, 0.00184804352466017f, 0.0025135783944278955f);
static const ai_layer_format_type conv2d_69_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_71_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_71_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_71_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_71_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_71_t_in_0_shape_ch_const_u16 = 48;
static const ai_u16 conv2d_71_t_out_0_shape_ch_const_u16 = 288;
static const ai_i8 conv2d_71_t_in_0_fmt_zero_const_s8 = -5;
static const ai_i8 conv2d_71_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_71_t_in_0_fmt_scale_const_f32 = 0.05984853208065033f;
static const ai_float conv2d_71_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_71_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.000995952868834138f, 0.0022306886967271566f, 0.0012051250087097287f, 0.0020315637812018394f, 0.003013614099472761f, 0.0010480884229764342f, 0.0013039730256423354f, 0.0011560030980035663f, 0.0029196436516940594f, 0.000839005399029702f, 0.001104912138544023f, 0.0017735662404447794f, 0.0010450583649799228f, 0.0011274456046521664f, 0.0017594390083104372f, 0.0014058052329346538f, 0.0016830385429784656f, 0.0015506164636462927f, 0.0018599587492644787f, 0.0009186507668346167f, 0.0008943414431996644f, 0.0015761158429086208f, 0.0009463402675464749f, 0.0011178605491295457f, 0.0015529233496636152f, 0.0013540438376367092f, 0.0012041719164699316f, 0.0015692593296989799f, 0.0015919249271973968f, 0.0011765157105401158f, 0.0015527184586971998f, 0.0008186650811694562f, 0.0018721141386777163f, 0.001790109439752996f, 0.0013991874875500798f, 0.002055620076134801f, 0.0006966860964894295f, 0.001991268480196595f, 0.0009142691851593554f, 0.0012603142531588674f, 0.0012061398010700941f, 0.0014417505590245128f, 0.0016399157466366887f, 0.001650556456297636f, 0.0013545644469559193f, 0.0014793528243899345f, 0.0017376452451571822f, 0.0009952380787581205f, 0.0008952688076533377f, 0.0013899303739890456f, 0.0009565710206516087f, 0.0009904453763738275f, 0.0015116144204512239f, 0.000871894066222012f, 0.0008510709158144891f, 0.0015463518211618066f, 0.002148806117475033f, 0.001769809052348137f, 0.001147180562838912f, 0.0018314803019165993f, 0.001714421552605927f, 0.0010366453789174557f, 0.000928784254938364f, 0.0013654612703248858f, 0.0010710919741541147f, 0.0020085994619876146f, 0.0012780848192051053f, 0.0025147320702672005f, 0.0012319752713665366f, 0.0009116291766986251f, 0.0017050046008080244f, 0.0021877619437873363f, 0.0012188792461529374f, 0.0014487870503216982f, 0.0011831116862595081f, 0.0008668529917486012f, 0.0008607337949797511f, 0.001035830588079989f, 0.0015375096118077636f, 0.0010787926148623228f, 0.0009606218663975596f, 0.0012310306774452329f, 0.004313313402235508f, 0.0012733567273244262f, 0.0011908488813787699f, 0.0013242063578218222f, 0.0017707232618704438f, 0.0021320516243577003f, 0.0018107611685991287f, 0.0018497894052416086f, 0.0012378806713968515f, 0.0011201080633327365f, 0.0015889513306319714f, 0.0031714115757495165f, 0.0011611226946115494f, 0.0015432186191901565f, 0.0011884687701240182f, 0.0009765504510141909f, 0.0014534754445776343f, 0.0015442969743162394f, 0.0008811951265670359f, 0.0016602854011580348f, 0.0009925102349370718f, 0.001190545386634767f, 0.0009211252327077091f, 0.0015134281711652875f, 0.0017768096877261996f, 0.0013625968713313341f, 0.0008380040526390076f, 0.0010602367110550404f, 0.0011408583959564567f, 0.0012283489340916276f, 0.001220007543452084f, 0.0014163526939228177f, 0.0019785920158028603f, 0.0018776101060211658f, 0.0016920027555897832f, 0.002280710730701685f, 0.0014659111620858312f, 0.0014357598265632987f, 0.001652355887927115f, 0.0006071780808269978f, 0.0014648993965238333f, 0.0010723230661824346f, 0.0015541886677965522f, 0.0010231875348836184f, 0.001921477378346026f, 0.0014431601157411933f, 0.0015307968715205789f, 0.0010034021688625216f, 0.0012031918158754706f, 0.0012331516481935978f, 0.001176029210910201f, 0.001731525408104062f, 0.0020101037807762623f, 0.0021528988145291805f, 0.0013850401155650616f, 0.0018633869476616383f, 0.0014662249013781548f, 0.0015735722845420241f, 0.001794450101442635f, 0.0011927683372050524f, 0.0014458500081673265f, 0.00193235301412642f, 0.000760009977966547f, 0.0012655722675845027f, 0.0008195033296942711f, 0.0019126262050122023f, 0.0016721347346901894f, 0.0018575562862679362f, 0.0010972301242873073f, 0.00205807713791728f, 0.0011342980433255434f, 0.0013766339980065823f, 0.0010340494336560369f, 0.001192416180856526f, 0.0017557519022375345f, 0.0013526977272704244f, 0.0013327306369319558f, 0.0011432942701503634f, 0.0012749886373057961f, 0.0024257369805127382f, 0.0008629971416667104f, 0.0016493768198415637f, 0.0015882238512858748f, 0.0012832090724259615f, 0.0008880746900103986f, 0.0008795449975878f, 0.0010574311017990112f, 0.0013300186255946755f, 0.0014907048316672444f, 0.0017368149710819125f, 0.0011613988317549229f, 0.001445218687877059f, 0.0018630424747243524f, 0.0012656168546527624f, 0.0013152766041457653f, 0.0010713873198255897f, 0.001990517368540168f, 0.001387519296258688f, 0.001300971256569028f, 0.000973294023424387f, 0.0014925315044820309f, 0.0011844008695334196f, 0.00145694799721241f, 0.0018671865109354258f, 0.001191740739159286f, 0.0008623486501164734f, 0.0008359494968317449f, 0.0015677949413657188f, 0.0014622625894844532f, 0.0011673002736642957f, 0.0014722957275807858f, 0.0011543306754902005f, 0.0017466699937358499f, 0.0014491098700091243f, 0.0010282947914674878f, 0.0014319616602733731f, 0.0013377676950767636f, 0.0008788154227659106f, 0.001369388191960752f, 0.0017795765306800604f, 0.0012687953421846032f, 0.0018147403607144952f, 0.0011637561256065965f, 0.0010568945435807109f, 0.001673094229772687f, 0.001443598186597228f, 0.001446531736291945f, 0.0012412202777341008f, 0.0018926546908915043f, 0.0037993749137967825f, 0.0012085175840184093f, 0.0010102110682055354f, 0.0012407603207975626f, 0.0011778533225879073f, 0.0016108062118291855f, 0.0014117024838924408f, 0.0013893720461055636f, 0.000980163924396038f, 0.0015634382143616676f, 0.001293644425459206f, 0.0011123018339276314f, 0.0016086186515167356f, 0.002131801098585129f, 0.0009449671488255262f, 0.0015097497962415218f, 0.001303700148127973f, 0.0009296200587414205f, 0.0013192014303058386f, 0.0010212870547547936f, 0.0008968330803327262f, 0.001089767087250948f, 0.0011267695808783174f, 0.0010045316303148866f, 0.00121268630027771f, 0.0016322422306984663f, 0.001235207892023027f, 0.0012389086186885834f, 0.0012102401815354824f, 0.0025802054442465305f, 0.000752518477384001f, 0.0012034035753458738f, 0.0013071229914203286f, 0.0013025933876633644f, 0.001086301519535482f, 0.0011383764212951064f, 0.001172978663817048f, 0.0010593371698632836f, 0.0012281984090805054f, 0.0008848810684867203f, 0.002338893013074994f, 0.0013047086540609598f, 0.0014509756583720446f, 0.0014655317645519972f, 0.0011494369246065617f, 0.0030214106664061546f, 0.001743915257975459f, 0.000841079861856997f, 0.0008047716110013425f, 0.001345326192677021f, 0.0010003041243180633f, 0.0009247564012184739f, 0.0011379143688827753f, 0.0014412663877010345f, 0.000697814510203898f, 0.0010505500249564648f, 0.0010894896695390344f, 0.001181629952043295f, 0.0017648087814450264f, 0.0015707615530118346f, 0.0013433907879516482f, 0.0012976275756955147f, 0.0010560808004811406f, 0.0017335133161395788f, 0.0012239660136401653f, 0.0019197015790268779f, 0.0011786825489252806f, 0.0011543071595951915f, 0.0010166943538933992f, 0.0007598779047839344f, 0.001033357810229063f, 0.001311671920120716f, 0.0009414242231287062f, 0.0012875926913693547f, 0.0012777975061908364f, 0.0008135398384183645f, 0.0014102705754339695f);
static const ai_layer_format_type conv2d_71_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_72_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_72_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_72_pad_before_t_in_0_shape_h_const_u32 = 4;

static const ai_u16 conv2d_72_t_in_0_shape_w_const_u16 = 6;
static const ai_u16 conv2d_72_t_in_0_shape_h_const_u16 = 6;
static const ai_u16 conv2d_72_t_in_0_shape_ch_const_u16 = 288;
static const ai_u16 conv2d_72_l_stride_1_const_u16 = 2;
static const ai_u16 conv2d_72_l_stride_0_const_u16 = 2;
static const ai_i8 conv2d_72_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_72_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_72_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_72_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_72_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.011445373296737671f, 0.005883882287889719f, 0.008008651435375214f, 0.005547997076064348f, 0.004023008979856968f, 0.004991307854652405f, 0.005845989100635052f, 0.01305311731994152f, 0.004479929339140654f, 0.017707649618387222f, 0.009057929739356041f, 0.0033130235970020294f, 0.0071589346043765545f, 0.0056014531292021275f, 0.00315372203476727f, 0.005598429590463638f, 0.0043797618709504604f, 0.007059758994728327f, 0.005570715293288231f, 0.01020404789596796f, 0.005142571870237589f, 0.0048927124589681625f, 0.006899235770106316f, 0.012089352123439312f, 0.006016204599291086f, 0.005241183564066887f, 0.004915135912597179f, 0.005252430681139231f, 0.006623655091971159f, 0.007218214683234692f, 0.0092921769246459f, 0.0069702728651463985f, 0.004622127860784531f, 0.00651252968236804f, 0.00447428785264492f, 0.0030474180821329355f, 0.01444556936621666f, 0.004656834062188864f, 0.010846949182450771f, 0.011100796982645988f, 0.005987480282783508f, 0.008549162186682224f, 0.005496513564139605f, 0.005501389037817717f, 0.007298996206372976f, 0.00420328788459301f, 0.005829997360706329f, 0.006373769603669643f, 0.006937715224921703f, 0.0048977890983223915f, 0.00870030652731657f, 0.006886956747621298f, 0.00424961606040597f, 0.016835307702422142f, 0.006880614440888166f, 0.005818388890475035f, 0.003907940350472927f, 0.004481286741793156f, 0.006299206055700779f, 0.003809818532317877f, 0.004316330421715975f, 0.00750904344022274f, 0.006233215797692537f, 0.006269205827265978f, 0.0061615766026079655f, 0.003208729438483715f, 0.008173179812729359f, 0.003147637704387307f, 0.003962031565606594f, 0.010232924483716488f, 0.003980620298534632f, 0.005913167260587215f, 0.0054314518347382545f, 0.00812993012368679f, 0.005244066007435322f, 0.011974429711699486f, 0.014672105200588703f, 0.004724637605249882f, 0.008018330670893192f, 0.014237349852919579f, 0.011646021157503128f, 0.005200912244617939f, 0.0037172327283769846f, 0.004974901210516691f, 0.00644147302955389f, 0.004246722441166639f, 0.004606826230883598f, 0.0034862516913563013f, 0.0044062696397304535f, 0.004247582051903009f, 0.006326829083263874f, 0.012579910457134247f, 0.004862174857407808f, 0.001953676575794816f, 0.00722386222332716f, 0.005694340914487839f, 0.011821572668850422f, 0.008184650912880898f, 0.007279673125594854f, 0.006571340374648571f, 0.008340597152709961f, 0.007552226539701223f, 0.005664881318807602f, 0.005769035313278437f, 0.010980520397424698f, 0.00974670797586441f, 0.0052897995337843895f, 0.010565767996013165f, 0.010022524744272232f, 0.007384253200143576f, 0.006004273891448975f, 0.0067738802172243595f, 0.005286032799631357f, 0.005613462999463081f, 0.004604065325111151f, 0.004495775792747736f, 0.010444467887282372f, 0.004080924671143293f, 0.005513391923159361f, 0.0036445094738155603f, 0.006113034673035145f, 0.025677673518657684f, 0.005748093593865633f, 0.00944688729941845f, 0.004574417602270842f, 0.0071863471530377865f, 0.006213456857949495f, 0.006111874245107174f, 0.0053488388657569885f, 0.0068155755288898945f, 0.006439311429858208f, 0.00921908300369978f, 0.004210925195366144f, 0.005508867558091879f, 0.005451702978461981f, 0.0028699650429189205f, 0.009045855142176151f, 0.005335300229489803f, 0.009404364973306656f, 0.0037954600993543863f, 0.00669235922396183f, 0.007480533793568611f, 0.006072863005101681f, 0.006440537981688976f, 0.011911032721400261f, 0.007104948163032532f, 0.008025641553103924f, 0.009327097795903683f, 0.004403454251587391f, 0.004075917880982161f, 0.007644375320523977f, 0.005022170953452587f, 0.009404347278177738f, 0.003853158326819539f, 0.0049454402178525925f, 0.008281126618385315f, 0.004686211235821247f, 0.009570730850100517f, 0.0050504496321082115f, 0.010030461475253105f, 0.006792009808123112f, 0.005987504031509161f, 0.015390130691230297f, 0.005295526701956987f, 0.005736775230616331f, 0.0048577808775007725f, 0.010229155421257019f, 0.006971718743443489f, 0.010520831681787968f, 0.008619331754744053f, 0.008911577984690666f, 0.003302275435999036f, 0.0072500682435929775f, 0.006963945459574461f, 0.00551108131185174f, 0.007302915211766958f, 0.005804057698696852f, 0.012378804385662079f, 0.004963298328220844f, 0.005314414855092764f, 0.007582828402519226f, 0.007118260953575373f, 0.004268022254109383f, 0.008918012492358685f, 0.007240733597427607f, 0.003561424557119608f, 0.00916904304176569f, 0.013246097601950169f, 0.009838557802140713f, 0.006704088766127825f, 0.005661660805344582f, 0.007900950498878956f, 0.0048555852845311165f, 0.008271480910480022f, 0.005060206633061171f, 0.0035280052106827497f, 0.012529960833489895f, 0.005371539853513241f, 0.006024349480867386f, 0.004434571135789156f, 0.007180420681834221f, 0.0034895073622465134f, 0.0163099467754364f, 0.004874620586633682f, 0.006848473101854324f, 0.009192146360874176f, 0.006122722290456295f, 0.007472243160009384f, 0.007979413494467735f, 0.004676738288253546f, 0.003587795654311776f, 0.0016248690662905574f, 0.003371428232640028f, 0.012553811073303223f, 0.008913307450711727f, 0.0056804269552230835f, 0.004782006610184908f, 0.005195076577365398f, 0.0058470601215958595f, 0.005621924065053463f, 0.004408704582601786f, 0.005529855843633413f, 0.00709404656663537f, 0.009084820747375488f, 0.004030982963740826f, 0.0062196203507483006f, 0.005236673168838024f, 0.00712137296795845f, 0.006356547120958567f, 0.006083028391003609f, 0.006091603077948093f, 0.011870921589434147f, 0.0036337946075946093f, 0.008892674930393696f, 0.012720982544124126f, 0.006915002595633268f, 0.005115658510476351f, 0.006337310653179884f, 0.005225770641118288f, 0.0035961989779025316f, 0.003448772244155407f, 0.01009584404528141f, 0.0056090508587658405f, 0.008092875592410564f, 0.006514230277389288f, 0.007507100701332092f, 0.0064805601723492146f, 0.005889533087611198f, 0.006019731517881155f, 0.010110554285347462f, 0.013381415978074074f, 0.004264150746166706f, 0.011153738014400005f, 0.008522821590304375f, 0.00532585708424449f, 0.007253168150782585f, 0.0021679294295608997f, 0.0034257969819009304f, 0.00723686208948493f, 0.013077035546302795f, 0.005462575238198042f, 0.013004777021706104f, 0.008148713037371635f, 0.005575970280915499f, 0.006167163606733084f, 0.016893602907657623f, 0.007354738190770149f, 0.006782608572393656f, 0.007875536568462849f, 0.005016196519136429f, 0.005110716912895441f, 0.006522322539240122f, 0.006495111621916294f, 0.008247481659054756f, 0.0041178022511303425f, 0.007494475692510605f, 0.004213497042655945f, 0.005030923523008823f, 0.006181870587170124f, 0.006814185064285994f, 0.008989117108285427f, 0.008543509989976883f, 0.005428142845630646f, 0.009380154311656952f, 0.008345523849129677f, 0.0055525717325508595f, 0.006739907432347536f, 0.006121374201029539f);
static const ai_u16 conv2d_72_t_out_0_shape_w_const_u16 = 2;
static const ai_u16 conv2d_72_t_out_0_shape_h_const_u16 = 2;

static const ai_u16 conv2d_73_t_in_0_shape_w_const_u16 = 2;
static const ai_u16 conv2d_73_t_in_0_shape_h_const_u16 = 2;
static const ai_u16 conv2d_73_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_73_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_73_t_in_0_shape_ch_const_u16 = 288;
static const ai_u16 conv2d_73_t_out_0_shape_ch_const_u16 = 80;
static const ai_i8 conv2d_73_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_73_t_out_0_fmt_zero_const_s8 = 10;
static const ai_float conv2d_73_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_73_t_out_0_fmt_scale_const_f32 = 0.03131386637687683f;
static const ai_float conv2d_73_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0017410855507478118f, 0.0018236498581245542f, 0.0021895503159612417f, 0.0015480824513360858f, 0.0019102341029793024f, 0.0022037383168935776f, 0.0016281246207654476f, 0.001592891407199204f, 0.0024073573295027018f, 0.0020465378183871508f, 0.0011539102997630835f, 0.0007030309061519802f, 0.0009003835148178041f, 0.001166866859421134f, 0.0009979559108614922f, 0.001412855344824493f, 0.0008663451881147921f, 0.0009847842156887054f, 0.0007561667007394135f, 0.0007750263903290033f, 0.0009324167622253299f, 0.0008837223285809159f, 0.0008948231115937233f, 0.0013227899326011539f, 0.0009386158781126142f, 0.0012780529214069247f, 0.0012075857957825065f, 0.0009756683721207082f, 0.0014148137997835875f, 0.001490450114943087f, 0.0010426692897453904f, 0.0011827924754470587f, 0.001708835014142096f, 0.0010359592270106077f, 0.0009453762322664261f, 0.0010755964322015643f, 0.001322766998782754f, 0.0011127390898764133f, 0.001241127261891961f, 0.0012779540847986937f, 0.0010208336170762777f, 0.0009895090479403734f, 0.0010331801604479551f, 0.001396381063386798f, 0.0011740789050236344f, 0.0010980035876855254f, 0.0010161536047235131f, 0.0012588223908096552f, 0.000820384593680501f, 0.0009535612189210951f, 0.0009578100871294737f, 0.001165237044915557f, 0.0010601445101201534f, 0.0009966074721887708f, 0.0012153030838817358f, 0.0010916849132627249f, 0.0012006118195131421f, 0.0010307814227417111f, 0.0011860488448292017f, 0.0014295947039499879f, 0.0011072828201577067f, 0.001230559777468443f, 0.0009052959503605962f, 0.0013479130575433373f, 0.000791068363469094f, 0.00109806505497545f, 0.0009746880386956036f, 0.0010582574177533388f, 0.001227191649377346f, 0.0011701845796778798f, 0.001514377654530108f, 0.0009670323343016207f, 0.0010996176861226559f, 0.001318002468906343f, 0.0013557716738432646f, 0.001220870646648109f, 0.001010380801744759f, 0.000999311450868845f, 0.001003430807031691f, 0.0008867832948453724f);
static const ai_layer_format_type conv2d_73_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;



static const ai_u16 conv2d_77_t_in_0_shape_w_const_u16 = 2;
static const ai_u16 conv2d_77_t_in_0_shape_h_const_u16 = 2;
static const ai_u16 conv2d_77_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_77_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_77_t_in_0_shape_ch_const_u16 = 80;
static const ai_u16 conv2d_77_t_out_0_shape_ch_const_u16 = 480;
static const ai_i8 conv2d_77_t_in_0_fmt_zero_const_s8 = 10;
static const ai_i8 conv2d_77_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_77_t_in_0_fmt_scale_const_f32 = 0.03131386637687683f;
static const ai_float conv2d_77_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_77_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0018982701003551483f, 0.0012898033019155264f, 0.001901755458675325f, 0.0020514586940407753f, 0.0022859482560306787f, 0.0016701591666787863f, 0.001959065906703472f, 0.0012859866255894303f, 0.0019572158344089985f, 0.002371264388784766f, 0.0020161636639386415f, 0.002096934476867318f, 0.0014831545995548368f, 0.0022426906507462263f, 0.0016828334191814065f, 0.002699068980291486f, 0.001756836660206318f, 0.0018920315196737647f, 0.0022012032568454742f, 0.0018498885910958052f, 0.0019860633183270693f, 0.002014359226450324f, 0.0019952121656388044f, 0.0010666290763765574f, 0.0015719749499112368f, 0.0016712562646716833f, 0.0025702319107949734f, 0.0024096218403428793f, 0.002339667174965143f, 0.0013661144766956568f, 0.0019030107650905848f, 0.002035396173596382f, 0.0019127537962049246f, 0.0020057810470461845f, 0.001997794955968857f, 0.0016401878092437983f, 0.00204584957100451f, 0.001968539319932461f, 0.0038798030000180006f, 0.0012567401863634586f, 0.0021058646962046623f, 0.001865321071818471f, 0.002601565793156624f, 0.0010816280264407396f, 0.0022496895398944616f, 0.0017845233669504523f, 0.0027224139776080847f, 0.0018519536824896932f, 0.0022609045263379812f, 0.0013231554767116904f, 0.0022891962435096502f, 0.002701573772355914f, 0.0021050083450973034f, 0.0024852382484823465f, 0.002010098658502102f, 0.0025105199310928583f, 0.0014703929191455245f, 0.0025043166242539883f, 0.0017537258099764585f, 0.0015189896803349257f, 0.001915656146593392f, 0.0014471763279289007f, 0.0021406624000519514f, 0.0015641720965504646f, 0.0014426762936636806f, 0.0013925351668149233f, 0.0023358813486993313f, 0.0015663070371374488f, 0.001467896276153624f, 0.0021698479540646076f, 0.0018951374804601073f, 0.002162669552490115f, 0.002336072502657771f, 0.0018632543506100774f, 0.0016029434045776725f, 0.0016129320720210671f, 0.0020793883595615625f, 0.002805096795782447f, 0.0023373037111014128f, 0.0015902358572930098f, 0.001607114914804697f, 0.002407260937616229f, 0.0017652695532888174f, 0.0011707879602909088f, 0.0014041023096069694f, 0.0012173161376267672f, 0.0021773199550807476f, 0.0019460475305095315f, 0.0019414144335314631f, 0.0026706289499998093f, 0.0020141282584518194f, 0.0023674634285271168f, 0.002281958470121026f, 0.002627635607495904f, 0.0013544439570978284f, 0.0018777363002300262f, 0.0015822186833247542f, 0.0027761629316955805f, 0.0023109479807317257f, 0.0021426561288535595f, 0.0030614736024290323f, 0.0013538774801418185f, 0.0016971374861896038f, 0.0015507523203268647f, 0.0021173555869609118f, 0.0021221553906798363f, 0.002984096063300967f, 0.0015551007818430662f, 0.0011102050775662065f, 0.001945659751072526f, 0.0018952579703181982f, 0.0019693556241691113f, 0.0024312869645655155f, 0.0018194131553173065f, 0.003274613758549094f, 0.002028216840699315f, 0.0028436423745006323f, 0.0026482632383704185f, 0.0018254750175401568f, 0.0025235279463231564f, 0.0009864728199318051f, 0.0015228079864755273f, 0.0016460027545690536f, 0.0017091803019866347f, 0.0021420663688331842f, 0.002323590451851487f, 0.0013880045153200626f, 0.0017015959601849318f, 0.0023737566079944372f, 0.0015759584493935108f, 0.0016719928244128823f, 0.0018862619763240218f, 0.0017478157533332705f, 0.0017817154293879867f, 0.0023496090434491634f, 0.002065395237877965f, 0.00200285529717803f, 0.0014429667498916388f, 0.0017268687952309847f, 0.0025292171631008387f, 0.0025413052644580603f, 0.001690462464466691f, 0.0020697875879704952f, 0.0022865263745188713f, 0.0026478527579456568f, 0.0016567312413826585f, 0.0018832918722182512f, 0.0013004242209717631f, 0.0021108570508658886f, 0.002817339962348342f, 0.0022439423482865095f, 0.0024348993320018053f, 0.0020632275845855474f, 0.002002218971028924f, 0.0019148718565702438f, 0.0017777543980628252f, 0.0017030806047841907f, 0.001789240282960236f, 0.0023114322684705257f, 0.0021326022688299417f, 0.003029508050531149f, 0.0023907488211989403f, 0.001290730433538556f, 0.0017304318025708199f, 0.0019919381011277437f, 0.0017401637742295861f, 0.0014104919973760843f, 0.0021790210157632828f, 0.0015864608576521277f, 0.0010012916754931211f, 0.0015863599255681038f, 0.002835072809830308f, 0.0016942209331318736f, 0.0021713196765631437f, 0.0028994260355830193f, 0.0018171998672187328f, 0.0015315115451812744f, 0.002036362886428833f, 0.0024727904237806797f, 0.002348599024116993f, 0.002226717071607709f, 0.0017026468412950635f, 0.0014307030942291021f, 0.0015635084128007293f, 0.0018452565418556333f, 0.0019748725462704897f, 0.0021117995493113995f, 0.002468852559104562f, 0.0018463486339896917f, 0.0017594764940440655f, 0.002803091425448656f, 0.0028763199225068092f, 0.0014470735331997275f, 0.0028766717296093702f, 0.002118492964655161f, 0.001736131263896823f, 0.001176526304334402f, 0.0013983029639348388f, 0.0017980597913265228f, 0.0017906336579471827f, 0.0019461128395050764f, 0.0018178209429606795f, 0.0012598260072991252f, 0.0011348900152370334f, 0.0010624970309436321f, 0.0020346026867628098f, 0.002421795390546322f, 0.003782915649935603f, 0.002194571541622281f, 0.0021165679208934307f, 0.0019016740843653679f, 0.001923388452269137f, 0.002132608089596033f, 0.0016621709801256657f, 0.0015958835138007998f, 0.0016893136780709028f, 0.001331938779912889f, 0.0021390216425061226f, 0.0021602564956992865f, 0.0022273496724665165f, 0.0016639851965010166f, 0.0017780984053388238f, 0.0021802885457873344f, 0.0013914995361119509f, 0.0034979574847966433f, 0.0016406469512730837f, 0.002042199485003948f, 0.0023965502623468637f, 0.0014889540616422892f, 0.0016145387198776007f, 0.0013164165429770947f, 0.0023982366546988487f, 0.0023769824765622616f, 0.0015684157842770219f, 0.0012030807556584477f, 0.0017437320202589035f, 0.0015027433400973678f, 0.002169383689761162f, 0.0021830848418176174f, 0.0016447040252387524f, 0.0016926215030252934f, 0.0018025445751845837f, 0.0018013614462688565f, 0.0020850920118391514f, 0.0014658732106909156f, 0.001679091015830636f, 0.0019832791294902563f, 0.0016740703722462058f, 0.0019098836928606033f, 0.002596969483420253f, 0.0014702635817229748f, 0.002375108189880848f, 0.0026221235748380423f, 0.0015209597768262029f, 0.0018635995220392942f, 0.00179057358764112f, 0.0022711956407874823f, 0.0017312620766460896f, 0.0021429178304970264f, 0.0015080668963491917f, 0.0015793959610164165f, 0.002044196240603924f, 0.0016590601298958063f, 0.001478868187405169f, 0.0024794666096568108f, 0.0015940098091959953f, 0.003198379185050726f, 0.0025325389578938484f, 0.0022578476928174496f, 0.0015713635366410017f, 0.0024228570982813835f, 0.0015436848625540733f, 0.0019979767967015505f, 0.0018240738427266479f, 0.0014470104360952973f, 0.002286835340783f, 0.001553728012368083f, 0.0018750770250335336f, 0.003087470307946205f, 0.0008615013211965561f, 0.002574316458776593f, 0.002797238063067198f, 0.0013337824493646622f, 0.0017626803601160645f, 0.002306865295395255f, 0.0017756508896127343f, 0.0016169912414625287f, 0.0015322333201766014f, 0.0025827093049883842f, 0.001503186416812241f, 0.001513603376224637f, 0.0015152922132983804f, 0.0020454281475394964f, 0.001452382537536323f, 0.0017676308052614331f, 0.0019005637150257826f, 0.0021851803176105022f, 0.0024137881118804216f, 0.0019310141215100884f, 0.0016174608608707786f, 0.0014775303425267339f, 0.002619644161313772f, 0.0017154165543615818f, 0.0026144678704440594f, 0.0019814465194940567f, 0.002312918659299612f, 0.001640496775507927f, 0.0015883662272244692f, 0.002619125647470355f, 0.0020689633674919605f, 0.00220963591709733f, 0.0012968670343980193f, 0.0014605853939428926f, 0.0022023438941687346f, 0.0013994533801451325f, 0.0017239167355000973f, 0.0026661099400371313f, 0.0021259074565023184f, 0.0011194003745913506f, 0.0023223047610372305f, 0.0014916852815076709f, 0.0023689572699368f, 0.0016048793913796544f, 0.002073027892038226f, 0.0027337404899299145f, 0.004238222725689411f, 0.0017186773475259542f, 0.0020402127411216497f, 0.0012258683564141393f, 0.002426755614578724f, 0.002760552568361163f, 0.0032835444435477257f, 0.0019742511212825775f, 0.0017513532657176256f, 0.0019045836525037885f, 0.004068419802933931f, 0.002569924807175994f, 0.002445509657263756f, 0.0021838583052158356f, 0.0014531520428135991f, 0.0010214971844106913f, 0.0019592237658798695f, 0.0021218517795205116f, 0.002146568149328232f, 0.0017938287928700447f, 0.0022110315039753914f, 0.002917633159086108f, 0.0014400656800717115f, 0.002047609305009246f, 0.0016112022567540407f, 0.0026237063575536013f, 0.00235443701967597f, 0.0013333121314644814f, 0.002016967162489891f, 0.001462477259337902f, 0.001209923648275435f, 0.001637294772081077f, 0.0016399679007008672f, 0.003334333188831806f, 0.001406102441251278f, 0.0015090880915522575f, 0.0008585801697336137f, 0.0016850418178364635f, 0.0016905933152884245f, 0.0022728031035512686f, 0.0024794701021164656f, 0.0023461435921490192f, 0.0016945380484685302f, 0.00142639537807554f, 0.0023860891815274954f, 0.0023972284980118275f, 0.0016888739774003625f, 0.0018592786509543657f, 0.002643702318891883f, 0.002296925289556384f, 0.001750405877828598f, 0.0019881445914506912f, 0.001830043620429933f, 0.002216577297076583f, 0.0020781310740858316f, 0.0012147636152803898f, 0.002932795090600848f, 0.001579717150889337f, 0.0015545218484476209f, 0.0015525267226621509f, 0.0014567591715604067f, 0.0014697788283228874f, 0.0015447235200554132f, 0.0022896667942404747f, 0.0020230989903211594f, 0.0015102687757462263f, 0.001856814487837255f, 0.001763714011758566f, 0.0024374688509851694f, 0.0025963580701500177f, 0.0014716323930770159f, 0.0022270730696618557f, 0.002146076411008835f, 0.0018018301343545318f, 0.003081293310970068f, 0.0021517432760447264f, 0.0018793452763929963f, 0.0020695445127785206f, 0.0018286752747371793f, 0.0014979590196162462f, 0.002130154985934496f, 0.002049261936917901f, 0.0020946357399225235f, 0.0023807089310139418f, 0.0014593712985515594f, 0.0019210075261071324f, 0.001436798251233995f, 0.003251362359151244f, 0.0022358421701937914f, 0.0020919477101415396f, 0.002743078861385584f, 0.0021875533275306225f, 0.0016639394452795386f, 0.001147471135482192f, 0.0020257425494492054f, 0.001144901616498828f, 0.0020769948605448008f, 0.002392437309026718f, 0.0016724253073334694f, 0.0018904044991359115f, 0.0016334864776581526f, 0.0016266836319118738f, 0.001347366371192038f, 0.0018094921251758933f, 0.002051640534773469f, 0.0020351968705654144f, 0.003339616348966956f, 0.002724846825003624f, 0.0023845792748034f, 0.0016953765880316496f, 0.0018293904140591621f, 0.002034233184531331f, 0.001216190867125988f, 0.0009956293506547809f, 0.001599309965968132f, 0.0024960904847830534f, 0.002701250836253166f, 0.002090047812089324f, 0.00182624242734164f, 0.0017822383670136333f, 0.0029109343886375427f, 0.001599159906618297f, 0.0010106536792591214f, 0.0021187555976212025f, 0.0014960472472012043f, 0.0023013439495116472f, 0.002350226743146777f, 0.0019364102045074105f, 0.0016330130165442824f, 0.002013174118474126f, 0.0021039689891040325f, 0.0017875477205961943f, 0.0013768142089247704f, 0.001757040387019515f, 0.0019455617293715477f, 0.0014831386506557465f, 0.001641551498323679f, 0.0027706045657396317f, 0.0023975353688001633f, 0.002282144268974662f, 0.0018588812090456486f, 0.0014842828968539834f, 0.003202480962499976f, 0.0023806593380868435f, 0.001891966676339507f, 0.0023040033411234617f, 0.0019038927275687456f, 0.0019640710670500994f, 0.0019296619575470686f, 0.0020153624936938286f, 0.0018235527677461505f, 0.001959544839337468f, 0.0017988518811762333f, 0.0014883860712870955f, 0.0015055707190185785f);
static const ai_layer_format_type conv2d_77_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_78_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_78_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_78_pad_before_t_in_0_shape_h_const_u32 = 2;

static const ai_u16 conv2d_78_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_78_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_78_t_in_0_shape_ch_const_u16 = 480;
static const ai_u16 conv2d_78_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_78_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_78_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_78_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_78_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_78_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_78_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.010945074260234833f, 0.012699291110038757f, 0.011402406729757786f, 0.011571393348276615f, 0.01130964420735836f, 0.015353942289948463f, 0.008747410960495472f, 0.019240980967879295f, 0.011101041920483112f, 0.011719629168510437f, 0.010842873714864254f, 0.0076947868801653385f, 0.013563037849962711f, 0.01090543158352375f, 0.007366792764514685f, 0.006473823916167021f, 0.007710359524935484f, 0.008992263115942478f, 0.01711372099816799f, 0.021317552775144577f, 0.010310639627277851f, 0.009333375841379166f, 0.009545547887682915f, 0.027198757976293564f, 0.00788911059498787f, 0.008321673609316349f, 0.007218424696475267f, 0.013578926213085651f, 0.01217681635171175f, 0.020789390429854393f, 0.021837569773197174f, 0.009435741230845451f, 0.009952301159501076f, 0.014832683838903904f, 0.00949585996568203f, 0.013390352949500084f, 0.009676032699644566f, 0.008834107778966427f, 0.005557295400649309f, 0.02187476120889187f, 0.016620486974716187f, 0.007973738014698029f, 0.006328409072011709f, 0.025713203474879265f, 0.012452810071408749f, 0.011962208896875381f, 0.009870677255094051f, 0.011116132140159607f, 0.004851812031120062f, 0.01752607524394989f, 0.01513206958770752f, 0.00997909065335989f, 0.008989403024315834f, 0.010448988527059555f, 0.006772486958652735f, 0.006105263717472553f, 0.01053896639496088f, 0.01686892844736576f, 0.006045730784535408f, 0.0074320281855762005f, 0.01380268856883049f, 0.01775435544550419f, 0.00989283062517643f, 0.01852116920053959f, 0.01319120917469263f, 0.007621580269187689f, 0.010280966758728027f, 0.013690242543816566f, 0.012263879179954529f, 0.01441290881484747f, 0.01277785375714302f, 0.013875317759811878f, 0.008396404795348644f, 0.005367123056203127f, 0.005204470362514257f, 0.015279153361916542f, 0.00692316610366106f, 0.008614464662969112f, 0.007955004461109638f, 0.01335220132023096f, 0.021745676174759865f, 0.007373222149908543f, 0.009980565868318081f, 0.012646187096834183f, 0.011246863752603531f, 0.023349864408373833f, 0.017012130469083786f, 0.008851679041981697f, 0.010336515493690968f, 0.010143036022782326f, 0.008544358424842358f, 0.010617558844387531f, 0.008032187819480896f, 0.009451234713196754f, 0.010297650471329689f, 0.015708770602941513f, 0.020915912464261055f, 0.009040001779794693f, 0.0104370741173625f, 0.006502819713205099f, 0.005799450911581516f, 0.007341120392084122f, 0.017351046204566956f, 0.011855260469019413f, 0.007440059911459684f, 0.00963749922811985f, 0.0063011860474944115f, 0.013134422712028027f, 0.03953138366341591f, 0.008787413127720356f, 0.008230607025325298f, 0.021481603384017944f, 0.007561766542494297f, 0.010575349442660809f, 0.008456673473119736f, 0.007891815155744553f, 0.005509117618203163f, 0.0091756796464324f, 0.010501939803361893f, 0.0057386779226362705f, 0.021201835945248604f, 0.011470555327832699f, 0.011698423884809017f, 0.011404035612940788f, 0.009660420008003712f, 0.0076769073493778706f, 0.03138657286763191f, 0.017624661326408386f, 0.004766541067510843f, 0.01684216968715191f, 0.01065884530544281f, 0.010620370507240295f, 0.011564411222934723f, 0.014269494451582432f, 0.010254471562802792f, 0.01268452126532793f, 0.008479061536490917f, 0.009913638234138489f, 0.008017062209546566f, 0.01135989185422659f, 0.009370274841785431f, 0.014274412766098976f, 0.006840880960226059f, 0.009777017869055271f, 0.012377978302538395f, 0.006201581098139286f, 0.010258736088871956f, 0.015335677191615105f, 0.01004116889089346f, 0.011102795600891113f, 0.006815256550908089f, 0.008567563258111477f, 0.013893768191337585f, 0.012425018474459648f, 0.010306566022336483f, 0.007736746221780777f, 0.011661911383271217f, 0.01869756169617176f, 0.005384745076298714f, 0.006561251822859049f, 0.007038133218884468f, 0.010260460898280144f, 0.011669849045574665f, 0.015618255361914635f, 0.015538152307271957f, 0.007929130457341671f, 0.021712452173233032f, 0.008623112924396992f, 0.011793000623583794f, 0.019904445856809616f, 0.0065215290524065495f, 0.007944504730403423f, 0.006252088584005833f, 0.00910299364477396f, 0.007311978843063116f, 0.017203107476234436f, 0.007802285719662905f, 0.008263632655143738f, 0.009565887972712517f, 0.00589585630223155f, 0.011082908138632774f, 0.008110014721751213f, 0.016412274911999702f, 0.021173302084207535f, 0.009907258674502373f, 0.012825350277125835f, 0.007075794972479343f, 0.007149895653128624f, 0.006834565196186304f, 0.013275571167469025f, 0.009351338259875774f, 0.007507222704589367f, 0.013679742813110352f, 0.007394523359835148f, 0.01609962247312069f, 0.022813603281974792f, 0.041671864688396454f, 0.014382747933268547f, 0.007028860040009022f, 0.012382006272673607f, 0.010707308538258076f, 0.010599656961858273f, 0.0117866275832057f, 0.021537059918045998f, 0.01682812161743641f, 0.015206564217805862f, 0.013358073309063911f, 0.003949666395783424f, 0.0095627186819911f, 0.012161164544522762f, 0.013381936587393284f, 0.01404223870486021f, 0.018204709514975548f, 0.015730654820799828f, 0.013461681082844734f, 0.01779678836464882f, 0.02827605977654457f, 0.007513462100178003f, 0.007587245665490627f, 0.006983779836446047f, 0.014939422719180584f, 0.009064588695764542f, 0.00633013341575861f, 0.008062208071351051f, 0.010943799279630184f, 0.011334802024066448f, 0.007834822870790958f, 0.006071425508707762f, 0.007335883565247059f, 0.017931954935193062f, 0.0143215861171484f, 0.01148562878370285f, 0.0058089070953428745f, 0.015655962750315666f, 0.022570934146642685f, 0.01052735187113285f, 0.010569911450147629f, 0.008480574004352093f, 0.009187135845422745f, 0.016969922930002213f, 0.010462802834808826f, 0.015745656564831734f, 0.009949474595487118f, 0.01726745069026947f, 0.015390532091259956f, 0.008382380940020084f, 0.009954032488167286f, 0.010631195269525051f, 0.012847594916820526f, 0.005753945093601942f, 0.01037900522351265f, 0.007675096392631531f, 0.010349022224545479f, 0.010753938928246498f, 0.012427656911313534f, 0.0076293302699923515f, 0.00853662844747305f, 0.00917639210820198f, 0.019867276772856712f, 0.009897567331790924f, 0.025315428152680397f, 0.01328660175204277f, 0.02159184217453003f, 0.014561166055500507f, 0.006717598997056484f, 0.013325594365596771f, 0.009803908877074718f, 0.011430315673351288f, 0.01600056327879429f, 0.009891361929476261f, 0.012129126116633415f, 0.013303925283253193f, 0.016457047313451767f, 0.013701081275939941f, 0.010593519546091557f, 0.005774487741291523f, 0.008825032040476799f, 0.009105159901082516f, 0.0057810950092971325f, 0.027779312804341316f, 0.009230995550751686f, 0.00809263065457344f, 0.016558950766921043f, 0.010916725732386112f, 0.007842790335416794f, 0.00681126955896616f, 0.010154279880225658f, 0.020891495048999786f, 0.010356415994465351f, 0.010751399211585522f, 0.024189747869968414f, 0.010734791867434978f, 0.008221483789384365f, 0.018800407648086548f, 0.02208372950553894f, 0.008544815704226494f, 0.006244183983653784f, 0.007061207201331854f, 0.009913071990013123f, 0.01791299320757389f, 0.011931782588362694f, 0.008916166611015797f, 0.013101658783853054f, 0.007799504790455103f, 0.007110846694558859f, 0.010541211813688278f, 0.0162039864808321f, 0.012928525917232037f, 0.004396464210003614f, 0.012667602859437466f, 0.0054022022522985935f, 0.015389928594231606f, 0.01906559243798256f, 0.012257545255124569f, 0.02354917861521244f, 0.03910479694604874f, 0.00835101492702961f, 0.012115567922592163f, 0.019316697493195534f, 0.009987671859562397f, 0.012359535321593285f, 0.01630243845283985f, 0.014089514501392841f, 0.024013740941882133f, 0.012841611169278622f, 0.005619964562356472f, 0.010625789873301983f, 0.010863302275538445f, 0.014424921944737434f, 0.007903979159891605f, 0.005890802945941687f, 0.011777600273489952f, 0.01396085973829031f, 0.012419209815561771f, 0.006859258748590946f, 0.005751881282776594f, 0.008093688637018204f, 0.014901320450007915f, 0.012698686681687832f, 0.016028450801968575f, 0.008625589311122894f, 0.006754016503691673f, 0.006676473654806614f, 0.008158938027918339f, 0.01831742748618126f, 0.008393931202590466f, 0.008140261285007f, 0.02847331203520298f, 0.008961259387433529f, 0.010446975938975811f, 0.009425245225429535f, 0.010624676942825317f, 0.014144182205200195f, 0.00906699150800705f, 0.006559683475643396f, 0.011163291521370411f, 0.010632215067744255f, 0.01825805939733982f, 0.005112909246236086f, 0.013877270743250847f, 0.008608578704297543f, 0.02045239508152008f, 0.008091708645224571f, 0.018850065767765045f, 0.011534295976161957f, 0.01088419184088707f, 0.006320890970528126f, 0.009803537279367447f, 0.009942445904016495f, 0.007921525277197361f, 0.010427766479551792f, 0.009864958003163338f, 0.0057847886346280575f, 0.009377537295222282f, 0.016352558508515358f, 0.011828145943582058f, 0.016200389713048935f, 0.0123119642958045f, 0.010511275380849838f, 0.014337779022753239f, 0.016444822773337364f, 0.012747741304337978f, 0.012533673085272312f, 0.010383781045675278f, 0.010172008536756039f, 0.010251662693917751f, 0.01793322153389454f, 0.01956143230199814f, 0.004232939798384905f, 0.015659920871257782f, 0.011850656010210514f, 0.012479634955525398f, 0.007667311001569033f, 0.008001109585165977f, 0.013471613638103008f, 0.010107673704624176f, 0.01045705284923315f, 0.013183417730033398f, 0.011027771979570389f, 0.00837442371994257f, 0.01014014519751072f, 0.013247471302747726f, 0.009985172189772129f, 0.01071273721754551f, 0.010497014038264751f, 0.012823685072362423f, 0.011787696741521358f, 0.012762868776917458f, 0.011731539852917194f, 0.010052484460175037f, 0.018439752981066704f, 0.007301181089133024f, 0.010901479050517082f, 0.012889904901385307f, 0.006724358536303043f, 0.01040749903768301f, 0.006717434152960777f, 0.01374292653053999f, 0.009910495951771736f, 0.013132837601006031f, 0.01841125078499317f, 0.01029863953590393f, 0.008855216205120087f, 0.008228820748627186f, 0.01566310040652752f, 0.00831136479973793f, 0.007156435865908861f, 0.018975261598825455f, 0.0110824229195714f, 0.0064383880235254765f, 0.0101266885176301f, 0.004419988952577114f, 0.006820047274231911f, 0.009293628856539726f, 0.006251021288335323f, 0.015836462378501892f, 0.007214178331196308f, 0.014511965215206146f, 0.02460147999227047f, 0.016142448410391808f, 0.005305113270878792f, 0.010537085123360157f, 0.011288133449852467f, 0.01067732460796833f, 0.01100139506161213f, 0.0059159598313272f, 0.009261813014745712f, 0.018795859068632126f, 0.009298248216509819f, 0.012654900550842285f, 0.006776310037821531f, 0.008463320322334766f, 0.009911415167152882f, 0.009344029240310192f, 0.011895626783370972f, 0.0053841304033994675f, 0.015389306470751762f, 0.014684857800602913f, 0.010833288542926311f, 0.011519743129611015f, 0.008116000331938267f, 0.010226844809949398f, 0.00922151654958725f, 0.008049901574850082f, 0.008055631071329117f, 0.011110779829323292f, 0.011127909645438194f, 0.010646381415426731f, 0.0144283976405859f, 0.012320329435169697f, 0.007655632682144642f, 0.012760556302964687f, 0.0057244086638092995f, 0.014594219624996185f, 0.013663800433278084f, 0.01582574099302292f, 0.006590352393686771f, 0.0070106266066432f, 0.013933266513049603f, 0.015384308062493801f);
static const ai_u16 conv2d_78_t_out_0_shape_w_const_u16 = 2;
static const ai_u16 conv2d_78_t_out_0_shape_h_const_u16 = 2;

static const ai_u16 conv2d_79_t_in_0_shape_w_const_u16 = 2;
static const ai_u16 conv2d_79_t_in_0_shape_h_const_u16 = 2;
static const ai_u16 conv2d_79_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_79_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_79_t_in_0_shape_ch_const_u16 = 480;
static const ai_u16 conv2d_79_t_out_0_shape_ch_const_u16 = 80;
static const ai_i8 conv2d_79_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_79_t_out_0_fmt_zero_const_s8 = 4;
static const ai_float conv2d_79_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_79_t_out_0_fmt_scale_const_f32 = 0.038860611617565155f;
static const ai_float conv2d_79_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0023822379298508167f, 0.0015376356896013021f, 0.002025319030508399f, 0.0017374914605170488f, 0.0014292142586782575f, 0.001546230399981141f, 0.0016008622478693724f, 0.0015284379478543997f, 0.0014499331591650844f, 0.001778727164492011f, 0.0009360573021695018f, 0.000742296630050987f, 0.0013688455801457167f, 0.001011629356071353f, 0.0010453155264258385f, 0.0010657262755557895f, 0.0011332500725984573f, 0.0007740809232927859f, 0.0010178879601880908f, 0.001452327356673777f, 0.0009697560453787446f, 0.0012835938250645995f, 0.0010999629739671946f, 0.001313504995778203f, 0.00119175692088902f, 0.0012496620183810592f, 0.0011053706984966993f, 0.0011051190085709095f, 0.000788730219937861f, 0.0010265698656439781f, 0.0009033262613229454f, 0.0011106644524261355f, 0.0009061119053512812f, 0.0011048127198591828f, 0.0008675114368088543f, 0.0011373658198863268f, 0.001197941368445754f, 0.0011675999267026782f, 0.0012987450463697314f, 0.0012169214896857738f, 0.0008378701168112457f, 0.001120748114772141f, 0.001023618970066309f, 0.00115799845661968f, 0.0009963102638721466f, 0.0007575263152830303f, 0.0010135533520951867f, 0.0011230779346078634f, 0.0007579431985504925f, 0.0006010023062117398f, 0.0010271371575072408f, 0.0013441875344142318f, 0.0009574469295330346f, 0.0010375346755608916f, 0.0011573502561077476f, 0.00094530766364187f, 0.000885673041921109f, 0.0010202151024714112f, 0.0014429831644520164f, 0.0009113634587265551f, 0.0008690492250025272f, 0.0010081087239086628f, 0.0008924466092139482f, 0.001717988634482026f, 0.0009042484452947974f, 0.0011300805490463972f, 0.0009742028778418899f, 0.00095350545598194f, 0.0013515067985281348f, 0.0012147282250225544f, 0.0009638042538426816f, 0.0010142801329493523f, 0.0008910217438824475f, 0.001397274318151176f, 0.0005776313482783735f, 0.0009204965899698436f, 0.0011167669435963035f, 0.0016014744760468602f, 0.0012180089252069592f, 0.0013927824329584837f);
static const ai_layer_format_type conv2d_79_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;




static const ai_u16 conv2d_84_t_in_0_shape_w_const_u16 = 2;
static const ai_u16 conv2d_84_t_in_0_shape_h_const_u16 = 2;
static const ai_u16 conv2d_84_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_84_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_84_t_in_0_shape_ch_const_u16 = 80;
static const ai_u16 conv2d_84_t_out_0_shape_ch_const_u16 = 480;
static const ai_i8 conv2d_84_t_in_0_fmt_zero_const_s8 = 4;
static const ai_i8 conv2d_84_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_84_t_in_0_fmt_scale_const_f32 = 0.038860611617565155f;
static const ai_float conv2d_84_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_84_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00123144022654742f, 0.001332299318164587f, 0.001128128613345325f, 0.0011520052794367075f, 0.0011913538910448551f, 0.0015644109807908535f, 0.001105939969420433f, 0.0015942336758598685f, 0.0012150739785283804f, 0.00098819297272712f, 0.001715948455967009f, 0.0013662249548360705f, 0.001667722943238914f, 0.0020490798633545637f, 0.001212310860864818f, 0.0015844288282096386f, 0.001758238417096436f, 0.001521263737231493f, 0.0015003889566287398f, 0.0016428619856014848f, 0.0016648877644911408f, 0.001326325349509716f, 0.001006383798085153f, 0.0016474936855956912f, 0.0011482744012027979f, 0.00144639378413558f, 0.0017452625324949622f, 0.001384868985041976f, 0.001601186115294695f, 0.001531893969513476f, 0.001114879734814167f, 0.0012862238800153136f, 0.0018081339076161385f, 0.0017249444499611855f, 0.0016676504164934158f, 0.0007489876006729901f, 0.0015841933200135827f, 0.0010113578755408525f, 0.0010171595495194197f, 0.001711271470412612f, 0.001284661702811718f, 0.0019192654872313142f, 0.0014947339659556746f, 0.0021550841629505157f, 0.001140808453783393f, 0.0010759815340861678f, 0.0008418639190495014f, 0.0012917806161567569f, 0.001751522533595562f, 0.0012155896984040737f, 0.0015732793835923076f, 0.0010314829414710402f, 0.0017885920824483037f, 0.0012044231407344341f, 0.0010982706444337964f, 0.0010182803962379694f, 0.0011851072777062654f, 0.0012957396684214473f, 0.0014278913149610162f, 0.0013163171242922544f, 0.0009865271858870983f, 0.0017959693213924766f, 0.0012511992827057838f, 0.001882050302810967f, 0.0011272940319031477f, 0.0017632393864914775f, 0.0016857021255418658f, 0.0009000564459711313f, 0.0011220796732231975f, 0.001302754390053451f, 0.0015414096415042877f, 0.001215257914736867f, 0.0016337501583620906f, 0.0014313369756564498f, 0.002026553964242339f, 0.001493602991104126f, 0.0010472062276676297f, 0.0014983672881498933f, 0.0013187440345063806f, 0.001063087722286582f, 0.0012128613889217377f, 0.0011638845317065716f, 0.0006643558153882623f, 0.0011405883124098182f, 0.0018437361577525735f, 0.0009916395647451282f, 0.0026195209939032793f, 0.0013046814128756523f, 0.0015132022090256214f, 0.0016175347846001387f, 0.0017961853882297873f, 0.0016166857676580548f, 0.0016682952409610152f, 0.0013768826611340046f, 0.0012143476633355021f, 0.0014506964944303036f, 0.0015945681370794773f, 0.0016376629937440157f, 0.0011921870755031705f, 0.0008952968055382371f, 0.0010171935427933931f, 0.0011047335574403405f, 0.0009215034078806639f, 0.0008287967066280544f, 0.0015058615244925022f, 0.0018977430881932378f, 0.0010432512499392033f, 0.0016671833582222462f, 0.0015057997079566121f, 0.0015187200624495745f, 0.001496929326094687f, 0.0012148254318162799f, 0.0022455272264778614f, 0.0015717055648565292f, 0.001269762055017054f, 0.001779663609340787f, 0.0012299424270167947f, 0.000820474058855325f, 0.0016685001319274306f, 0.0009073632536455989f, 0.0010700174607336521f, 0.0015745575074106455f, 0.0012658253544941545f, 0.001312699750997126f, 0.0012126431101933122f, 0.001336451736278832f, 0.0011538848048076034f, 0.0011261791223660111f, 0.0010289627825841308f, 0.002156888833269477f, 0.003253374481573701f, 0.0018143553752452135f, 0.0014061657711863518f, 0.0010878680041059852f, 0.0015943233156576753f, 0.0010873101418837905f, 0.0016008431557565928f, 0.0009930492378771305f, 0.0017720108153298497f, 0.0010714008240029216f, 0.0009404981974512339f, 0.0016001821495592594f, 0.0017774120206013322f, 0.0012094061821699142f, 0.001080009271390736f, 0.0010299901477992535f, 0.0015932921087369323f, 0.0018379937391728163f, 0.0014559709234163165f, 0.0014658182626590133f, 0.0014494045171886683f, 0.0022067870013415813f, 0.0017117572715505958f, 0.0015760393580421805f, 0.001439176849089563f, 0.0013103444362059236f, 0.0009357301751151681f, 0.0022110913414508104f, 0.0018255084287375212f, 0.0016353874234482646f, 0.0017032966716215014f, 0.0015144047793000937f, 0.0013702150899916887f, 0.0020988367032259703f, 0.0008590726065449417f, 0.001845035352744162f, 0.0013270038180053234f, 0.0009474821854382753f, 0.0026109451428055763f, 0.001335937064141035f, 0.0017071871552616358f, 0.0016164571279659867f, 0.0018713263561949134f, 0.0010299356654286385f, 0.002570314798504114f, 0.0016671069897711277f, 0.0019286333117634058f, 0.0010645040310919285f, 0.0016747144982218742f, 0.0019206202123314142f, 0.0013117255875840783f, 0.0011990193743258715f, 0.001232727081514895f, 0.001415728940628469f, 0.0015748647274449468f, 0.0015639896737411618f, 0.0011373094748705626f, 0.0024014420341700315f, 0.0017358779441565275f, 0.0016435531433671713f, 0.0016545694088563323f, 0.0014946862356737256f, 0.0013603660045191646f, 0.0013508256524801254f, 0.0015949348453432322f, 0.0010394133860245347f, 0.0011747998651117086f, 0.0012334838975220919f, 0.0018970132805407047f, 0.0010781529126688838f, 0.0012623696820810437f, 0.0014098287792876363f, 0.0010744541650637984f, 0.002226982032880187f, 0.001018643262796104f, 0.0012496308190748096f, 0.001169684692285955f, 0.0014535461086779833f, 0.001238431897945702f, 0.0010648434981703758f, 0.001057822024449706f, 0.0012613574508577585f, 0.0010027807438746095f, 0.0015809513861313462f, 0.001487795845605433f, 0.0011198625434190035f, 0.0018027994083240628f, 0.0014490674948319793f, 0.001741032931022346f, 0.0012073091929778457f, 0.001536605996079743f, 0.0013515434693545103f, 0.001249235006980598f, 0.0015497622080147266f, 0.001294609741307795f, 0.0009234296157956123f, 0.0013709964696317911f, 0.0016619660891592503f, 0.0008509214385412633f, 0.0010773284593597054f, 0.001683531911112368f, 0.0015491083031520247f, 0.0012890072539448738f, 0.0009952406398952007f, 0.0018833836074918509f, 0.0019629995804280043f, 0.0017871529562398791f, 0.001789397792890668f, 0.0021814012434333563f, 0.0017183211166411638f, 0.0012478624703362584f, 0.0010819960152730346f, 0.0012766894651576877f, 0.0013806159840896726f, 0.0014900729293003678f, 0.0009766098810359836f, 0.002233645413070917f, 0.0014124957378953695f, 0.0011489728931337595f, 0.0014125948073342443f, 0.0010175765492022038f, 0.0014414131874218583f, 0.0021984197665005922f, 0.0011213633697479963f, 0.0019035121658816934f, 0.0009581185295246542f, 0.0015411011409014463f, 0.0010460116900503635f, 0.0011753998696804047f, 0.0018495253752917051f, 0.0024665642995387316f, 0.0011338799959048629f, 0.0014527657767757773f, 0.0014324079966172576f, 0.0009535558056086302f, 0.0008525971788913012f, 0.001328548532910645f, 0.0010688429465517402f, 0.000957177602685988f, 0.0017306384397670627f, 0.0022894474677741528f, 0.0008708018576726317f, 0.0012517309514805675f, 0.0012192706344649196f, 0.0015163933858275414f, 0.000982877565547824f, 0.0008719295728951693f, 0.001611817511729896f, 0.002120822900906205f, 0.0015106718055903912f, 0.0017252806574106216f, 0.0015588069800287485f, 0.001472074305638671f, 0.0013570598093792796f, 0.0012333451304584742f, 0.0010690242052078247f, 0.0014255237765610218f, 0.0014289693208411336f, 0.0013455795124173164f, 0.001456654630601406f, 0.0011681171599775553f, 0.001369129284285009f, 0.001688641496002674f, 0.0009337132796645164f, 0.0017524410504847765f, 0.0009370339685119689f, 0.0008887371513992548f, 0.001031773746944964f, 0.0014552592765539885f, 0.0011257895966991782f, 0.0013329958310350776f, 0.0020844521932303905f, 0.0018666120013222098f, 0.0014720183098688722f, 0.0011217230930924416f, 0.001750840456224978f, 0.0016348795033991337f, 0.0013324131723493338f, 0.0016119559295475483f, 0.0014785437379032373f, 0.0029613215010613203f, 0.002875995123758912f, 0.001727295108139515f, 0.002408628584817052f, 0.0021548508666455746f, 0.0015184866497293115f, 0.0016556745395064354f, 0.0013921242207288742f, 0.001730112126097083f, 0.00156537932343781f, 0.0017210101941600442f, 0.0015331401955336332f, 0.001258095377124846f, 0.001842631259933114f, 0.001305650221183896f, 0.0017980291740968823f, 0.001388176460750401f, 0.001471406314522028f, 0.0017391011351719499f, 0.0012893618550151587f, 0.0018227500841021538f, 0.0008639192674309015f, 0.0011128326877951622f, 0.0013900207122787833f, 0.0011570731876417994f, 0.0019188192673027515f, 0.0014558382099494338f, 0.0016730696661397815f, 0.001778587931767106f, 0.0013146038400009274f, 0.0008841667440719903f, 0.0009009263594634831f, 0.0012728197034448385f, 0.001094442792236805f, 0.0013996970374137163f, 0.0025540166534483433f, 0.0014234490226954222f, 0.0015056964475661516f, 0.0016050096601247787f, 0.0014698741724714637f, 0.0010255001252517104f, 0.0013794798869639635f, 0.0013027646346017718f, 0.001303375232964754f, 0.0021622367203235626f, 0.0014263985212892294f, 0.0016528688138350844f, 0.0016724079614505172f, 0.0012378180399537086f, 0.001080522546544671f, 0.0018879505805671215f, 0.0020731703843921423f, 0.002345956629142165f, 0.0013752372469753027f, 0.0016713266959413886f, 0.00122062920127064f, 0.0008880658424459398f, 0.0019341257866472006f, 0.0015397449024021626f, 0.0009431677754037082f, 0.0012124660424888134f, 0.0011577270925045013f, 0.0009144344949163496f, 0.0016270881751552224f, 0.0022077469620853662f, 0.0016211296897381544f, 0.0008374137105420232f, 0.0017288957023993134f, 0.0007940679206512868f, 0.0020301369950175285f, 0.0016450004186481237f, 0.0012857745168730617f, 0.0016897673485800624f, 0.0014909098390489817f, 0.0011850573355332017f, 0.0015263112727552652f, 0.0007318209391087294f, 0.0012962371110916138f, 0.0013694923836737871f, 0.0013987194979563355f, 0.0015857222024351358f, 0.001337424386292696f, 0.0014333550352603197f, 0.0011613977840170264f, 0.0011092463973909616f, 0.0013313739327713847f, 0.0010243193246424198f, 0.00207283953204751f, 0.001408234005793929f, 0.0016470811096951365f, 0.0013763833558186889f, 0.0014464093837887049f, 0.0016308107879012823f, 0.0012707395944744349f, 0.001432043849490583f, 0.0022145488765090704f, 0.0011869969312101603f, 0.0017100003315135837f, 0.0013850679388269782f, 0.0010418634628877044f, 0.0019061974016949534f, 0.0012270434526726604f, 0.0010333667742088437f, 0.001457872916944325f, 0.0014598129782825708f, 0.0017857863567769527f, 0.0011142740258947015f, 0.002519180066883564f, 0.001445079455152154f, 0.001051698811352253f, 0.0018045221222564578f, 0.0014002014650031924f, 0.0007245857268571854f, 0.0008352496661245823f, 0.0013425274519249797f, 0.001015246263705194f, 0.0014550008345395327f, 0.0013170201564207673f, 0.0017193765379488468f, 0.001536441850475967f, 0.00188886106479913f, 0.0008293266291730106f, 0.0013265147572383285f, 0.0008397561032325029f, 0.0029098400846123695f, 0.0013416619040071964f, 0.0009313878836110234f, 0.0017198727000504732f, 0.0013289289781823754f, 0.0009259021608158946f, 0.0011908799642696977f, 0.001164957182481885f, 0.0017634436953812838f, 0.0017668927321210504f, 0.0020272093825042248f, 0.0010495221940800548f, 0.0010575351770967245f, 0.0013292406219989061f, 0.0016119966749101877f, 0.0009954022243618965f, 0.0014893743209540844f, 0.0013508338015526533f, 0.0014736676821485162f, 0.0016942886868491769f, 0.0026028589345514774f, 0.0008964590961113572f, 0.0018834093352779746f, 0.0017060079844668508f, 0.002073881682008505f, 0.0011846849229186773f, 0.0008968745241872966f, 0.0014794570161029696f, 0.0009220427600666881f, 0.0017126110615208745f, 0.0011941612465307117f, 0.0010177813237532973f, 0.0016442089108750224f, 0.0013854524586349726f, 0.0022023471537977457f, 0.0019462615018710494f, 0.0013433843851089478f, 0.0010161914397031069f, 0.001590993837453425f, 0.0014629290672019124f, 0.0014462423278018832f, 0.0020014941692352295f, 0.0012211194261908531f, 0.0017107668099924922f, 0.0009541948675177991f, 0.001535243820399046f);
static const ai_layer_format_type conv2d_84_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_85_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_85_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_85_pad_before_t_in_0_shape_h_const_u32 = 2;

static const ai_u16 conv2d_85_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_85_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_85_t_in_0_shape_ch_const_u16 = 480;
static const ai_u16 conv2d_85_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_85_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_85_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_85_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_85_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_85_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_85_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.009907983243465424f, 0.013274943456053734f, 0.018064085394144058f, 0.010455701500177383f, 0.02154902182519436f, 0.014322294853627682f, 0.01186060719192028f, 0.01166026946157217f, 0.017255499958992004f, 0.01610734686255455f, 0.00733069097623229f, 0.011834582313895226f, 0.0140143483877182f, 0.007353262975811958f, 0.005446200259029865f, 0.01572529971599579f, 0.008236932568252087f, 0.01808325946331024f, 0.011902186088263988f, 0.006239567417651415f, 0.010152176022529602f, 0.014988727867603302f, 0.014337779022753239f, 0.018692025914788246f, 0.008542937226593494f, 0.014525449834764004f, 0.010521477088332176f, 0.00932942982763052f, 0.012693031691014767f, 0.014671582728624344f, 0.015349315479397774f, 0.027842262759804726f, 0.013822576962411404f, 0.006550896447151899f, 0.013732095248997211f, 0.03096708655357361f, 0.013553360477089882f, 0.016892623156309128f, 0.037045810371637344f, 0.010179748758673668f, 0.012022053822875023f, 0.015520550310611725f, 0.006948806345462799f, 0.011949748732149601f, 0.015434841625392437f, 0.0243680402636528f, 0.006757088005542755f, 0.011681501753628254f, 0.009135880507528782f, 0.012204939499497414f, 0.006184148136526346f, 0.014514066278934479f, 0.012589477002620697f, 0.015096282586455345f, 0.011359008960425854f, 0.019721725955605507f, 0.009207481518387794f, 0.013984033837914467f, 0.014981338754296303f, 0.008346396498382092f, 0.01015099324285984f, 0.012817483395338058f, 0.01615852490067482f, 0.00724783493205905f, 0.01053700502961874f, 0.007580042351037264f, 0.013714481145143509f, 0.015237555839121342f, 0.02335004135966301f, 0.007650382816791534f, 0.004560970701277256f, 0.014807160012423992f, 0.015933595597743988f, 0.010052925907075405f, 0.004727429710328579f, 0.013179362751543522f, 0.015652859583497047f, 0.0139557383954525f, 0.023053940385580063f, 0.010976203717291355f, 0.018502091988921165f, 0.008133317343890667f, 0.038653209805488586f, 0.027137232944369316f, 0.005925155710428953f, 0.017060816287994385f, 0.0040985532104969025f, 0.011694299057126045f, 0.007809300906956196f, 0.01267106831073761f, 0.017217885702848434f, 0.00770615367218852f, 0.009095624089241028f, 0.016574077308177948f, 0.005531173665076494f, 0.021218666806817055f, 0.012552582658827305f, 0.007431969046592712f, 0.009348687715828419f, 0.03932372108101845f, 0.018319012597203255f, 0.009022905491292477f, 0.012470394372940063f, 0.04739612340927124f, 0.0160750113427639f, 0.008454973809421062f, 0.01136036030948162f, 0.006960161961615086f, 0.01166897639632225f, 0.010956145823001862f, 0.01251082494854927f, 0.015584079548716545f, 0.005246320273727179f, 0.018222231417894363f, 0.00697401724755764f, 0.011538071557879448f, 0.022823737934231758f, 0.018536681309342384f, 0.010037083178758621f, 0.027059519663453102f, 0.017868956550955772f, 0.012088957242667675f, 0.016744714230298996f, 0.010098704136908054f, 0.012965483590960503f, 0.021678123623132706f, 0.027762290090322495f, 0.016365770250558853f, 0.018656374886631966f, 0.00939643569290638f, 0.007975214160978794f, 0.006066292058676481f, 0.010983447544276714f, 0.01577387936413288f, 0.007373769301921129f, 0.014707738533616066f, 0.008426032029092312f, 0.01908385008573532f, 0.01302512176334858f, 0.017650488764047623f, 0.02067573368549347f, 0.012332173064351082f, 0.00895461905747652f, 0.008459102362394333f, 0.027296047657728195f, 0.019259249791502953f, 0.013426333665847778f, 0.007560761179775f, 0.021843140944838524f, 0.012668591924011707f, 0.016564400866627693f, 0.005259792786091566f, 0.005452910903841257f, 0.005150842014700174f, 0.007398227695375681f, 0.01083644013851881f, 0.02180635742843151f, 0.009015267714858055f, 0.010062356479465961f, 0.020839611068367958f, 0.008568703196942806f, 0.005221141502261162f, 0.0082795275375247f, 0.012201654724776745f, 0.01635928452014923f, 0.020312894135713577f, 0.00941536482423544f, 0.014826327562332153f, 0.009076935239136219f, 0.009526700712740421f, 0.009282493963837624f, 0.02264152280986309f, 0.010379791259765625f, 0.006822892930358648f, 0.004561919718980789f, 0.012486506253480911f, 0.020414575934410095f, 0.016536034643650055f, 0.01373239140957594f, 0.00924062728881836f, 0.011557184159755707f, 0.019809896126389503f, 0.013565647415816784f, 0.008188300766050816f, 0.011138754896819592f, 0.016019364818930626f, 0.011618097312748432f, 0.016658423468470573f, 0.013022694736719131f, 0.0089088911190629f, 0.017490295693278313f, 0.011671597138047218f, 0.01174767967313528f, 0.007552963215857744f, 0.007549132686108351f, 0.009658552706241608f, 0.011933770962059498f, 0.016396023333072662f, 0.008518952876329422f, 0.020248934626579285f, 0.01079900749027729f, 0.006619702558964491f, 0.009471219964325428f, 0.004470059182494879f, 0.011052393354475498f, 0.02433611825108528f, 0.011036469601094723f, 0.008315570652484894f, 0.013806459493935108f, 0.018359828740358353f, 0.01644526980817318f, 0.009530104696750641f, 0.011577922850847244f, 0.008328328840434551f, 0.009953423403203487f, 0.010682078078389168f, 0.0057537974789738655f, 0.01489118579775095f, 0.015340810641646385f, 0.00970480777323246f, 0.013785077258944511f, 0.012185030616819859f, 0.022733112797141075f, 0.007863679900765419f, 0.01134943775832653f, 0.013456054031848907f, 0.019718769937753677f, 0.009122689254581928f, 0.02542414888739586f, 0.009557254612445831f, 0.008763132616877556f, 0.01150102075189352f, 0.019742855802178383f, 0.04033893719315529f, 0.009856943972408772f, 0.011326020583510399f, 0.008209101855754852f, 0.008309370838105679f, 0.00621053809300065f, 0.012442974373698235f, 0.0092158829793334f, 0.008664466440677643f, 0.015044339001178741f, 0.0066437008790671825f, 0.015401467680931091f, 0.011546884663403034f, 0.00926867127418518f, 0.014108485542237759f, 0.02486513741314411f, 0.011364521458745003f, 0.013886533677577972f, 0.01736651547253132f, 0.006544066593050957f, 0.02214295044541359f, 0.007872361689805984f, 0.013875944539904594f, 0.009313637390732765f, 0.017731884494423866f, 0.01864273101091385f, 0.01072088535875082f, 0.0046788351610302925f, 0.014983375556766987f, 0.007598846219480038f, 0.016397541388869286f, 0.021687911823391914f, 0.027113385498523712f, 0.01274064276367426f, 0.007623102981597185f, 0.02232021652162075f, 0.009726503863930702f, 0.006683865562081337f, 0.013832850381731987f, 0.02109122835099697f, 0.009002787992358208f, 0.02255309745669365f, 0.017206838354468346f, 0.017178691923618317f, 0.020270131528377533f, 0.008380847051739693f, 0.007711181882768869f, 0.009836294688284397f, 0.010393957607448101f, 0.01192421279847622f, 0.00957371387630701f, 0.022262951359152794f, 0.024534525349736214f, 0.003853896167129278f, 0.006678663194179535f, 0.007704064715653658f, 0.008715762756764889f, 0.01941790245473385f, 0.01669687032699585f, 0.013925567269325256f, 0.01811322197318077f, 0.009851666167378426f, 0.026211390271782875f, 0.028696522116661072f, 0.02350992150604725f, 0.00867512822151184f, 0.014583011157810688f, 0.012663230299949646f, 0.009727959521114826f, 0.006067947018891573f, 0.00881073996424675f, 0.008992199786007404f, 0.009181028231978416f, 0.004978301003575325f, 0.00756197702139616f, 0.006153689231723547f, 0.011405166238546371f, 0.006208263337612152f, 0.004669653717428446f, 0.007340223994106054f, 0.007202046923339367f, 0.008036743849515915f, 0.01729048229753971f, 0.0074187214486300945f, 0.010458005592226982f, 0.012677984312176704f, 0.017492827028036118f, 0.007743136957287788f, 0.016202695667743683f, 0.008196180686354637f, 0.010530268773436546f, 0.008481147699058056f, 0.0065605188719928265f, 0.015142582356929779f, 0.010362394154071808f, 0.005672132130712271f, 0.015982206910848618f, 0.009002579376101494f, 0.017047708854079247f, 0.012417083606123924f, 0.00704934261739254f, 0.010037333704531193f, 0.012826376594603062f, 0.011115748435258865f, 0.015842143446207047f, 0.009321854449808598f, 0.00485741114243865f, 0.020520541816949844f, 0.02666347101330757f, 0.04388221353292465f, 0.01963440701365471f, 0.011696445755660534f, 0.0058644842356443405f, 0.019423356279730797f, 0.0054381429217755795f, 0.006185682024806738f, 0.005545257590711117f, 0.01019240077584982f, 0.013437307439744473f, 0.006941094063222408f, 0.016125690191984177f, 0.005739017389714718f, 0.01849900744855404f, 0.004489018581807613f, 0.01682606339454651f, 0.0068496158346533775f, 0.02227080427110195f, 0.011236798949539661f, 0.011086538434028625f, 0.006442287005484104f, 0.011717628687620163f, 0.011919806711375713f, 0.014422113075852394f, 0.027335936203598976f, 0.010005357675254345f, 0.005106460768729448f, 0.010418146848678589f, 0.018661705777049065f, 0.008390448056161404f, 0.025912733748555183f, 0.008019819855690002f, 0.005797248799353838f, 0.009264802560210228f, 0.033711522817611694f, 0.007806289009749889f, 0.04952669516205788f, 0.010450761765241623f, 0.01671011932194233f, 0.024731190875172615f, 0.011148840188980103f, 0.013993954285979271f, 0.013049661181867123f, 0.016601935029029846f, 0.01909087784588337f, 0.012671374715864658f, 0.01448152121156454f, 0.010573454201221466f, 0.011667733080685139f, 0.01666409708559513f, 0.01203168649226427f, 0.016635006293654442f, 0.013205395080149174f, 0.013737854547798634f, 0.017151523381471634f, 0.009001929312944412f, 0.022230692207813263f, 0.0058631813153624535f, 0.012471112422645092f, 0.011113581247627735f, 0.013715933077037334f, 0.006603152025490999f, 0.015731343999505043f, 0.005638289265334606f, 0.01073266752064228f, 0.00801023468375206f, 0.013342603109776974f, 0.011368270963430405f, 0.011967144906520844f, 0.01669313572347164f, 0.010419774800539017f, 0.018594594672322273f, 0.0227259062230587f, 0.014334643259644508f, 0.023042947053909302f, 0.0063549671322107315f, 0.01129845529794693f, 0.025938251987099648f, 0.009328932501375675f, 0.015580913051962852f, 0.03749782219529152f, 0.028216321021318436f, 0.012440570630133152f, 0.025492584332823753f, 0.005585360806435347f, 0.013981042429804802f, 0.005420870613306761f, 0.00566021678969264f, 0.01008023414760828f, 0.02626775950193405f, 0.009813284501433372f, 0.021181566640734673f, 0.002732116961851716f, 0.01197239849716425f, 0.011432895436882973f, 0.006926890928298235f, 0.014120033010840416f, 0.020156973972916603f, 0.02285330556333065f, 0.017713753506541252f, 0.0100936284288764f, 0.016465555876493454f, 0.008125741966068745f, 0.023686474189162254f, 0.030769145116209984f, 0.013771302998065948f, 0.009123326279222965f, 0.026442011818289757f, 0.011793422512710094f, 0.00743313180282712f, 0.01447464432567358f, 0.014403493143618107f, 0.004709512460976839f, 0.015052826143801212f, 0.009357521310448647f, 0.010761735960841179f, 0.011680278927087784f, 0.02108938805758953f, 0.012277113273739815f, 0.013024442829191685f, 0.019580068066716194f, 0.007087506353855133f, 0.012793469242751598f, 0.013677164912223816f, 0.015757471323013306f, 0.01498372107744217f, 0.010978933423757553f, 0.009030701592564583f, 0.015314954333007336f, 0.02552506886422634f, 0.01462520007044077f, 0.011640021577477455f, 0.005174506921321154f, 0.009248429909348488f, 0.010906870476901531f, 0.018704885616898537f, 0.016116773709654808f, 0.013439321890473366f);
static const ai_u16 conv2d_85_t_out_0_shape_w_const_u16 = 2;
static const ai_u16 conv2d_85_t_out_0_shape_h_const_u16 = 2;

static const ai_u16 conv2d_86_t_in_0_shape_w_const_u16 = 2;
static const ai_u16 conv2d_86_t_in_0_shape_h_const_u16 = 2;
static const ai_u16 conv2d_86_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_86_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_86_t_in_0_shape_ch_const_u16 = 480;
static const ai_u16 conv2d_86_t_out_0_shape_ch_const_u16 = 80;
static const ai_i8 conv2d_86_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_86_t_out_0_fmt_zero_const_s8 = 5;
static const ai_float conv2d_86_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_86_t_out_0_fmt_scale_const_f32 = 0.056112151592969894f;
static const ai_float conv2d_86_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.002396004507318139f, 0.0017154498491436243f, 0.002119005424901843f, 0.001297180773690343f, 0.0008578991983085871f, 0.0015677910996600986f, 0.001196612953208387f, 0.0016343037132173777f, 0.0012288702419027686f, 0.0014987423783168197f, 0.002100392011925578f, 0.0021783264819532633f, 0.0017832692246884108f, 0.0016377271385863423f, 0.002645672531798482f, 0.002750958548858762f, 0.002072998322546482f, 0.002177848480641842f, 0.002508174628019333f, 0.0015216174069792032f, 0.0016845397185534239f, 0.0017003548564389348f, 0.0028735282830893993f, 0.0014168620109558105f, 0.0027200253680348396f, 0.002511243335902691f, 0.001614599023014307f, 0.00324360397644341f, 0.0016913650324568152f, 0.002806832315400243f, 0.0019292457727715373f, 0.0016140660736709833f, 0.0015813643112778664f, 0.001478048274293542f, 0.002461876254528761f, 0.0026314810384064913f, 0.0024402241688221693f, 0.0017884747358039021f, 0.0023227587807923555f, 0.0018827921012416482f, 0.0024189145769923925f, 0.0018783370032906532f, 0.0012746179709210992f, 0.002493171487003565f, 0.001923202071338892f, 0.002015212783589959f, 0.0022526264656335115f, 0.002676107455044985f, 0.0016651375917717814f, 0.002763469237834215f, 0.001985289854928851f, 0.0021538464352488518f, 0.0016621880931779742f, 0.0014392223674803972f, 0.00099953671451658f, 0.0018061499577015638f, 0.002182414522394538f, 0.002521268092095852f, 0.0018875892274081707f, 0.00209024571813643f, 0.0018051924416795373f, 0.002162263263016939f, 0.0015133380657061934f, 0.002319721272215247f, 0.002160858828574419f, 0.0019052921561524272f, 0.0019287430914118886f, 0.001960130874067545f, 0.002089601708576083f, 0.0017991125350818038f, 0.003881761571392417f, 0.0013917889446020126f, 0.0015508781652897596f, 0.0035045314580202103f, 0.001531045651063323f, 0.003407486015930772f, 0.0022754159290343523f, 0.0015488044591620564f, 0.0016487601678818464f, 0.0018148564267903566f);
static const ai_layer_format_type conv2d_86_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_u16 conv2d_88_t_in_0_shape_w_const_u16 = 2;
static const ai_u16 conv2d_88_t_in_0_shape_h_const_u16 = 2;
static const ai_u16 conv2d_88_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_88_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_88_t_in_0_shape_ch_const_u16 = 80;
static const ai_u16 conv2d_88_t_out_0_shape_ch_const_u16 = 480;
static const ai_i8 conv2d_88_t_in_0_fmt_zero_const_s8 = 5;
static const ai_i8 conv2d_88_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_88_t_in_0_fmt_scale_const_f32 = 0.056112151592969894f;
static const ai_float conv2d_88_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_88_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0009877869160845876f, 0.0011364861857146025f, 0.0013097033370286226f, 0.0008027734002098441f, 0.0009072047541849315f, 0.0011077946983277798f, 0.0014670953387394547f, 0.0009825477609410882f, 0.001139783300459385f, 0.0009532622643746436f, 0.001343427924439311f, 0.0007885824306868017f, 0.0007937777554616332f, 0.0009934459812939167f, 0.0009677186608314514f, 0.0013146301498636603f, 0.0009537978330627084f, 0.0014214121038094163f, 0.0010573886102065444f, 0.0011406339472159743f, 0.0008404577965848148f, 0.0008767053950577974f, 0.003674581181257963f, 0.0010705776512622833f, 0.0014229576336219907f, 0.0010682615684345365f, 0.0009050252847373486f, 0.0011437213979661465f, 0.001072967192158103f, 0.0008062302367761731f, 0.0011406037956476212f, 0.0010665246518328786f, 0.0011512093478813767f, 0.000860731874126941f, 0.0010936205508187413f, 0.0010609481250867248f, 0.0010527617996558547f, 0.0008377595222555101f, 0.0015472983941435814f, 0.0013054425362497568f, 0.001163896406069398f, 0.0010470713023096323f, 0.001108901691623032f, 0.0009638620540499687f, 0.0009853322990238667f, 0.0010190819157287478f, 0.0011450707679614425f, 0.0012180391931906343f, 0.0012597846798598766f, 0.0009954571723937988f, 0.001051645609550178f, 0.001081485184840858f, 0.0009177718893624842f, 0.001113476580940187f, 0.0011362475343048573f, 0.0012524857884272933f, 0.0011838899226859212f, 0.0011493401834741235f, 0.0013003256171941757f, 0.0008688822272233665f, 0.0015075079863891006f, 0.0012300318339839578f, 0.0024454800877720118f, 0.001285175676457584f, 0.001300694071687758f, 0.0008576543186791241f, 0.0017398889176547527f, 0.0007944686803966761f, 0.001569128129631281f, 0.0012048757635056973f, 0.0014887561555951834f, 0.001199683640152216f, 0.0015685090329498053f, 0.0009902140591293573f, 0.0012349644675850868f, 0.0011411196319386363f, 0.0020710653625428677f, 0.0019058374455198646f, 0.0011525485897436738f, 0.0007075216271914542f, 0.001222762162797153f, 0.0008338023326359689f, 0.0011588858906179667f, 0.0006924463668838143f, 0.0013520614011213183f, 0.0010950687574222684f, 0.0008780562202446163f, 0.0009840497514232993f, 0.0013356147101148963f, 0.002735485788434744f, 0.0011186962947249413f, 0.0007233838550746441f, 0.0017956709489226341f, 0.0016750985523685813f, 0.0016816413262858987f, 0.001085544703528285f, 0.0011097305687144399f, 0.0009382038842886686f, 0.0007017431198619306f, 0.0010614156490191817f, 0.0008285405347123742f, 0.0011120487470179796f, 0.0009602928184904158f, 0.001471889205276966f, 0.0011228038929402828f, 0.0010878528701141477f, 0.0009255818440578878f, 0.0014519585529342294f, 0.0015053185634315014f, 0.0010837797308340669f, 0.0007520467042922974f, 0.0027721747756004333f, 0.0014013642212375998f, 0.0006889374344609678f, 0.0010355740087106824f, 0.001353390864096582f, 0.001233119168318808f, 0.0010197494411841035f, 0.0009825379820540547f, 0.0011094753863289952f, 0.0009170787525363266f, 0.0010477182222530246f, 0.0008005748386494815f, 0.0009654539171606302f, 0.0009320337558165193f, 0.001013746252283454f, 0.0011557372054085135f, 0.0008070192416198552f, 0.001541599747724831f, 0.0012748648878186941f, 0.0008810932631604373f, 0.0007960882503539324f, 0.0010124852415174246f, 0.0009361052070744336f, 0.0010000570910051465f, 0.0008467948064208031f, 0.0011978870024904609f, 0.0006994554423727095f, 0.0011416964698582888f, 0.0008686573128215969f, 0.0007116471533663571f, 0.001002128585241735f, 0.0011365279788151383f, 0.0010012038983404636f, 0.0012968688970431685f, 0.000789063866250217f, 0.0013778364518657327f, 0.0015980728203430772f, 0.001352094579488039f, 0.0006007238407619298f, 0.0008326134993694723f, 0.0010552708990871906f, 0.0006701119709759951f, 0.0013950401917099953f, 0.0009653741144575179f, 0.0009578974568285048f, 0.0012926298659294844f, 0.001046298653818667f, 0.0014445375418290496f, 0.0007840054458938539f, 0.0013256380334496498f, 0.0008163070888258517f, 0.0006146738305687904f, 0.001565840793773532f, 0.0006695713382214308f, 0.0011557204416021705f, 0.0006657859194092453f, 0.0009293646435253322f, 0.0009272726601921022f, 0.0006726834108121693f, 0.0012121921172365546f, 0.0014122327556833625f, 0.000817730266135186f, 0.0007239279220812023f, 0.001336845220066607f, 0.001069813733920455f, 0.0011381800286471844f, 0.0008940667612478137f, 0.0007878094911575317f, 0.0008967028115876019f, 0.0009854664094746113f, 0.0009081871248781681f, 0.0009767896262928843f, 0.0009510766831226647f, 0.0014536892995238304f, 0.0009878085693344474f, 0.0008348856936208904f, 0.0009352143970318139f, 0.0008377372287213802f, 0.0008918680832721293f, 0.0009960499592125416f, 0.0010464724618941545f, 0.0012556994333863258f, 0.0014283022610470653f, 0.0014630454825237393f, 0.001188617548905313f, 0.001036941190250218f, 0.001393546350300312f, 0.0006688751163892448f, 0.0010930069256573915f, 0.0011508860625326633f, 0.0008314125007018447f, 0.0005518604884855449f, 0.000939901452511549f, 0.0007550728623755276f, 0.0015265346737578511f, 0.0016195598291233182f, 0.0006660606013610959f, 0.0016632417682558298f, 0.0010409662500023842f, 0.0010748363565653563f, 0.001106868265196681f, 0.0015545962378382683f, 0.0016103563830256462f, 0.0010038323234766722f, 0.001279435004107654f, 0.0015218784101307392f, 0.000401709956349805f, 0.0007526949048042297f, 0.000802934227976948f, 0.0009581915801391006f, 0.001178253791294992f, 0.0009678729111328721f, 0.000952353875618428f, 0.0009385171579197049f, 0.000796484702732414f, 0.0007809647941030562f, 0.0009167498792521656f, 0.0014557706890627742f, 0.001136957434937358f, 0.0014611828373745084f, 0.0013350577792152762f, 0.0007238651742227376f, 0.0013727744808420539f, 0.0007730548968538642f, 0.0011344208614900708f, 0.0008987309993244708f, 0.002271209843456745f, 0.0012061798479408026f, 0.000988312647677958f, 0.0008071562624536455f, 0.001086692325770855f, 0.0012443636078387499f, 0.0009687312413007021f, 0.0010628742165863514f, 0.001475982484407723f, 0.0012961040483787656f, 0.0012713159667328f, 0.0009458853746764362f, 0.00134093698579818f, 0.0009523981134407222f, 0.000696890871040523f, 0.001174872275441885f, 0.0010701925493776798f, 0.0010445869993418455f, 0.0010076778708025813f, 0.0011663571931421757f, 0.0013194462517276406f, 0.001040658913552761f, 0.0010243651922792196f, 0.0007317569688893855f, 0.0007211075280793011f, 0.0010835051070898771f, 0.0009276484488509595f, 0.001271063694730401f, 0.0007019726326689124f, 0.0007603430422022939f, 0.0012129079550504684f, 0.001037298934534192f, 0.001160109299235046f, 0.0006680262158624828f, 0.0011622137390077114f, 0.001103415503166616f, 0.0011294930009171367f, 0.001136007602326572f, 0.0011585707543417811f, 0.0011293190764263272f, 0.0011212542885914445f, 0.0008984747109934688f, 0.000562128727324307f, 0.0009328249143436551f, 0.0016583793330937624f, 0.0013235462829470634f, 0.0007830913527868688f, 0.0008196847629733384f, 0.0014123665168881416f, 0.0012028486235067248f, 0.0007207893067970872f, 0.0010957204503938556f, 0.0011653797701001167f, 0.000879993021953851f, 0.0015310735907405615f, 0.0010983311804011464f, 0.0008913202909752727f, 0.0010378415463492274f, 0.0011453833431005478f, 0.0006896437844261527f, 0.0005464365240186453f, 0.0008334881858900189f, 0.0012813593493774533f, 0.0011322286445647478f, 0.0008239502203650773f, 0.0012120509054511786f, 0.0009873401140794158f, 0.0011189450742676854f, 0.0008567131590098143f, 0.0013524364912882447f, 0.0012650039279833436f, 0.0009901280282065272f, 0.0013052445137873292f, 0.0010535852052271366f, 0.0008081739069893956f, 0.0011777254985645413f, 0.0007185137365013361f, 0.0011900279205292463f, 0.0008845482370816171f, 0.0005998521228320897f, 0.0017486490542069077f, 0.0009032056550495327f, 0.0018750502495095134f, 0.0010608277516439557f, 0.000823429087176919f, 0.001112749450840056f, 0.0008273128187283874f, 0.001267334446310997f, 0.0012548939557746053f, 0.0014056444633752108f, 0.0009466835763305426f, 0.0008390472503378987f, 0.0011747600510716438f, 0.0013272804208099842f, 0.001248869113624096f, 0.0010610497556626797f, 0.0016132041346281767f, 0.0009993304265663028f, 0.0012492710957303643f, 0.000983696198090911f, 0.000978551572188735f, 0.0015795475337654352f, 0.0005607103230431676f, 0.0014801069628447294f, 0.0011585960164666176f, 0.0008262830087915063f, 0.0009005516767501831f, 0.0010518007911741734f, 0.001074090600013733f, 0.0013987877173349261f, 0.0009203206864185631f, 0.0010788033250719309f, 0.0009050712105818093f, 0.0018352990737184882f, 0.0010979340877383947f, 0.0007981149246916175f, 0.0007919821073301136f, 0.0008961477433331311f, 0.001093030092306435f, 0.0011311726411804557f, 0.001353370025753975f, 0.0008403175161220133f, 0.0009738609660416842f, 0.0010082286316901445f, 0.001103189424611628f, 0.0009896799456328154f, 0.0009120506001636386f, 0.0010264890734106302f, 0.0013370031956583261f, 0.0010730910580605268f, 0.001462262705899775f, 0.0012869681231677532f, 0.0012275459012016654f, 0.0010585562558844686f, 0.0014148805057629943f, 0.0008460042881779373f, 0.0009016935364343226f, 0.0009825545130297542f, 0.0014282967895269394f, 0.0013679014518857002f, 0.000928946421481669f, 0.0011167548364028335f, 0.001737523707561195f, 0.000844679307192564f, 0.000868461502250284f, 0.001521592610515654f, 0.0010892703430727124f, 0.0010503687663003802f, 0.0008453056216239929f, 0.0012198196491226554f, 0.0008311245474033058f, 0.0008166756015270948f, 0.0005827195709571242f, 0.0006657004123553634f, 0.0015850331401452422f, 0.001018524169921875f, 0.0011256583966314793f, 0.0009228226263076067f, 0.0010363562032580376f, 0.0010143269319087267f, 0.0010678755352273583f, 0.0013310888316482306f, 0.0009878125274553895f, 0.0007476003374904394f, 0.0004786422068718821f, 0.0009170811390504241f, 0.0009786818409338593f, 0.0014882819959893823f, 0.0011782246874645352f, 0.0007752012461423874f, 0.0013292408548295498f, 0.000770106038544327f, 0.0011094859801232815f, 0.0009301907848566771f, 0.0012030841317027807f, 0.001523675979115069f, 0.0009643154335208237f, 0.0012799100950360298f, 0.0008745417580939829f, 0.0012037790147587657f, 0.0008928244933485985f, 0.000980830634944141f, 0.0008614605758339167f, 0.0017515980871394277f, 0.0009609057451598346f, 0.0009447680204175413f, 0.0009939331794157624f, 0.0008040970424190164f, 0.0009943058248609304f, 0.0013014524010941386f, 0.003172144992277026f, 0.001706532435491681f, 0.0019381088204681873f, 0.0011230737436562777f, 0.0013945141108706594f, 0.001387262367643416f, 0.002534101717174053f, 0.0008406664128415287f, 0.00102736777625978f, 0.0008553236839361489f, 0.0008291833801195025f, 0.0010858208406716585f, 0.0011602749582380056f, 0.0016071783611550927f, 0.0013270159251987934f, 0.0011202660389244556f, 0.0009254183969460428f, 0.0011805767426267266f, 0.0013894346775487065f, 0.0013240557163953781f, 0.0009640906355343759f, 0.0012500048615038395f, 0.001243265112861991f, 0.0012460054131224751f, 0.0014423730317503214f, 0.0010608535958454013f, 0.0012947235954925418f, 0.0012091195676475763f, 0.0011966103920713067f, 0.001335248234681785f, 0.0014578221598640084f, 0.0007699147099629045f, 0.0008889597957022488f, 0.0006680141668766737f, 0.0009866992477327585f, 0.0018534836126491427f, 0.0011659810552373528f, 0.0011569912312552333f, 0.001372879953123629f, 0.0018289313884451985f, 0.0011036526411771774f, 0.001141962013207376f, 0.0011612331727519631f, 0.0008103064610622823f, 0.00124230922665447f, 0.0009460524888709188f, 0.0008868862641975284f, 0.0016153556061908603f, 0.0015202094800770283f, 0.0012699207291007042f, 0.0010401244508102536f, 0.001605317578651011f, 0.0007700029527768493f);
static const ai_layer_format_type conv2d_88_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_i8 conv2d_89_pad_before_v_pad_constant_value_const_s8[] = LITE_ARRAY_VALUES(-128);
static const ai_i16 conv2d_89_pad_before_t_in_0_fmt_bitsize_const_s16 = 8;
static const ai_u32 conv2d_89_pad_before_t_in_0_shape_h_const_u32 = 2;

static const ai_u16 conv2d_89_t_in_0_shape_w_const_u16 = 4;
static const ai_u16 conv2d_89_t_in_0_shape_h_const_u16 = 4;
static const ai_u16 conv2d_89_t_in_0_shape_ch_const_u16 = 480;
static const ai_u16 conv2d_89_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_89_l_stride_0_const_u16 = 1;
static const ai_i8 conv2d_89_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_89_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_89_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_89_t_out_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_89_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.009710944257676601f, 0.00623892480507493f, 0.005310174077749252f, 0.005603252910077572f, 0.01355965156108141f, 0.006453620735555887f, 0.011744026094675064f, 0.010660583153367043f, 0.011267664842307568f, 0.013911837711930275f, 0.005789850372821093f, 0.012203967198729515f, 0.036244381219148636f, 0.009396142326295376f, 0.007449035067111254f, 0.005241628270596266f, 0.012650898657739162f, 0.00917506031692028f, 0.006529154255986214f, 0.00630872743204236f, 0.014714065007865429f, 0.014604938216507435f, 0.005002251360565424f, 0.014106997288763523f, 0.006656228099018335f, 0.005272047594189644f, 0.012877455912530422f, 0.01202305220067501f, 0.006475320551544428f, 0.01746530458331108f, 0.012883733958005905f, 0.005387276876717806f, 0.009686123579740524f, 0.012587868608534336f, 0.012188096530735493f, 0.011055147275328636f, 0.025889378041028976f, 0.004164938349276781f, 0.006712071597576141f, 0.008133581839501858f, 0.011239564046263695f, 0.01373120117932558f, 0.013675382360816002f, 0.016357284039258957f, 0.007678360678255558f, 0.02002747729420662f, 0.005548010114580393f, 0.004622605629265308f, 0.00564968166872859f, 0.0061447047628462315f, 0.010977126657962799f, 0.015485437586903572f, 0.005529423244297504f, 0.005430882330983877f, 0.005258440040051937f, 0.02098586969077587f, 0.005092745181173086f, 0.010640867054462433f, 0.006416128017008305f, 0.020104778930544853f, 0.00556990597397089f, 0.010347657836973667f, 0.004900532774627209f, 0.008237896487116814f, 0.013359852135181427f, 0.013180429115891457f, 0.004229020792990923f, 0.024121591821312904f, 0.007192410062998533f, 0.006017089355736971f, 0.004200580529868603f, 0.012583284638822079f, 0.010755566880106926f, 0.01043674349784851f, 0.005424048285931349f, 0.006770296487957239f, 0.007199685089290142f, 0.010484145022928715f, 0.008750193752348423f, 0.008910014294087887f, 0.008817566558718681f, 0.0099131865426898f, 0.013246424496173859f, 0.00590699166059494f, 0.004632056225091219f, 0.0053236535750329494f, 0.010472902096807957f, 0.012029948644340038f, 0.00509986188262701f, 0.010507160797715187f, 0.005610145628452301f, 0.016969481483101845f, 0.005942315794527531f, 0.004571803379803896f, 0.0062944660894572735f, 0.007902479730546474f, 0.006086613982915878f, 0.017032839357852936f, 0.00806350912898779f, 0.012271548621356487f, 0.005156089551746845f, 0.009260456077754498f, 0.021471191197633743f, 0.010603724047541618f, 0.0148785300552845f, 0.008225065656006336f, 0.01125374250113964f, 0.010912043042480946f, 0.007460873108357191f, 0.005789141170680523f, 0.025649365037679672f, 0.004505293909460306f, 0.01017261203378439f, 0.014292499050498009f, 0.00828141625970602f, 0.008900408633053303f, 0.011872824281454086f, 0.01250722911208868f, 0.005142275243997574f, 0.011775334365665913f, 0.03004133328795433f, 0.006545044481754303f, 0.014925344847142696f, 0.010517029091715813f, 0.009502102620899677f, 0.015721172094345093f, 0.006074083503335714f, 0.01146532129496336f, 0.011338197626173496f, 0.007611089386045933f, 0.010588620789349079f, 0.00707203196361661f, 0.0178068894892931f, 0.011670593172311783f, 0.0051170713268220425f, 0.005625341087579727f, 0.009908829815685749f, 0.009933865629136562f, 0.00813745055347681f, 0.006812358275055885f, 0.016487151384353638f, 0.006498728878796101f, 0.004251379519701004f, 0.012730546295642853f, 0.008501826785504818f, 0.010570350103080273f, 0.005524265114217997f, 0.0036954390816390514f, 0.00601748563349247f, 0.009920832701027393f, 0.011959045194089413f, 0.009613149799406528f, 0.02763284556567669f, 0.004074549302458763f, 0.007162326481193304f, 0.010755867697298527f, 0.00805245153605938f, 0.004085566382855177f, 0.005386849399656057f, 0.019015993922948837f, 0.007958760485053062f, 0.0336255319416523f, 0.015530779026448727f, 0.004940323065966368f, 0.010215003043413162f, 0.00715095829218626f, 0.01944989524781704f, 0.010785802267491817f, 0.008723975159227848f, 0.00807404424995184f, 0.007353330962359905f, 0.014582953415811062f, 0.009698842652142048f, 0.014298430643975735f, 0.006261318456381559f, 0.005585178732872009f, 0.0055581447668373585f, 0.013727826997637749f, 0.008401699364185333f, 0.007184121757745743f, 0.025523493066430092f, 0.011279520578682423f, 0.008079282008111477f, 0.017260698601603508f, 0.0056649018079042435f, 0.026182161644101143f, 0.005612507462501526f, 0.02164197526872158f, 0.021499237045645714f, 0.01024740468710661f, 0.007103276904672384f, 0.010331754572689533f, 0.005828022491186857f, 0.01173319574445486f, 0.005343262571841478f, 0.009653435088694096f, 0.008551625534892082f, 0.011476589366793633f, 0.020148077979683876f, 0.013854523189365864f, 0.011584114283323288f, 0.013401266187429428f, 0.028538644313812256f, 0.014419290237128735f, 0.012122640386223793f, 0.005763849709182978f, 0.005260738078504801f, 0.0233624204993248f, 0.006307939998805523f, 0.006452728062868118f, 0.007495481055229902f, 0.010226297192275524f, 0.00959086325019598f, 0.010918647050857544f, 0.0058886390179395676f, 0.004292754456400871f, 0.007096105255186558f, 0.015980424359440804f, 0.024947570636868477f, 0.01789131574332714f, 0.015041307546198368f, 0.008586176671087742f, 0.007577904034405947f, 0.006251854356378317f, 0.011051327921450138f, 0.014703606255352497f, 0.00990443117916584f, 0.022133367136120796f, 0.010541761294007301f, 0.006391921080648899f, 0.007792954798787832f, 0.004204775672405958f, 0.02454843744635582f, 0.0072911023162305355f, 0.00852580089122057f, 0.008331572636961937f, 0.01856057532131672f, 0.00861082598567009f, 0.006940806750208139f, 0.010653221979737282f, 0.016356613487005234f, 0.021186888217926025f, 0.018618784844875336f, 0.013145233504474163f, 0.0065583656542003155f, 0.00997205637395382f, 0.006255635060369968f, 0.011748605407774448f, 0.025393066927790642f, 0.00520083773881197f, 0.007272503338754177f, 0.01525843609124422f, 0.01716144196689129f, 0.008163325488567352f, 0.01419960055500269f, 0.027562391012907028f, 0.014093196950852871f, 0.010633635334670544f, 0.0070745027624070644f, 0.012037888169288635f, 0.009366900660097599f, 0.016955005005002022f, 0.004888376221060753f, 0.03425100818276405f, 0.005783799570053816f, 0.04166029021143913f, 0.023012304678559303f, 0.0054098330438137054f, 0.005937471985816956f, 0.005593320820480585f, 0.01370291318744421f, 0.008801881223917007f, 0.010478650219738483f, 0.007375669665634632f, 0.003391681471839547f, 0.006258357781916857f, 0.005826851818710566f, 0.009310368448495865f, 0.008805384859442711f, 0.0306687094271183f, 0.010745683684945107f, 0.007351664360612631f, 0.006598972249776125f, 0.012139353901147842f, 0.009899194352328777f, 0.004497745539993048f, 0.006824049167335033f, 0.020393311977386475f, 0.005665117874741554f, 0.005607848521322012f, 0.01586996018886566f, 0.008610413409769535f, 0.006063738372176886f, 0.005701557267457247f, 0.0100865438580513f, 0.0058340285904705524f, 0.022878220304846764f, 0.026416132226586342f, 0.007352164015173912f, 0.010429889895021915f, 0.007712917868047953f, 0.011362291872501373f, 0.00988494511693716f, 0.025368640199303627f, 0.007388900499790907f, 0.014663999900221825f, 0.0037111404817551374f, 0.003904587822034955f, 0.012383599765598774f, 0.01024478767067194f, 0.00784110277891159f, 0.00691496767103672f, 0.00588051974773407f, 0.02355360798537731f, 0.005319735035300255f, 0.01720230095088482f, 0.019526472315192223f, 0.005628191865980625f, 0.017512429505586624f, 0.005869126878678799f, 0.006858405657112598f, 0.010660390369594097f, 0.004793880507349968f, 0.025974366813898087f, 0.0065758866257965565f, 0.008173859678208828f, 0.006983496714383364f, 0.01779964566230774f, 0.006865045055747032f, 0.006552245933562517f, 0.009988282807171345f, 0.006391939241439104f, 0.008439232595264912f, 0.00639384938403964f, 0.01027619931846857f, 0.00803112331777811f, 0.00979303102940321f, 0.014143355190753937f, 0.004338860046118498f, 0.02424304373562336f, 0.008115543983876705f, 0.010089821182191372f, 0.012703532353043556f, 0.006714378949254751f, 0.012433459050953388f, 0.0069399019703269005f, 0.007121371105313301f, 0.009847591631114483f, 0.009441963396966457f, 0.0059190476313233376f, 0.005664350930601358f, 0.005863542202860117f, 0.007881168276071548f, 0.025603244081139565f, 0.024607161059975624f, 0.005950538441538811f, 0.006867513991892338f, 0.007650693878531456f, 0.006858018692582846f, 0.008644862100481987f, 0.008985103107988834f, 0.026319799944758415f, 0.013113252818584442f, 0.014394435100257397f, 0.009393286891281605f, 0.006234405096620321f, 0.007958858273923397f, 0.005223714746534824f, 0.008104736916720867f, 0.004864981397986412f, 0.00863061472773552f, 0.011739833280444145f, 0.007134861312806606f, 0.008081733249127865f, 0.013106505386531353f, 0.004843211267143488f, 0.006569312885403633f, 0.008383423089981079f, 0.010114767588675022f, 0.012126515619456768f, 0.0103345587849617f, 0.008060300722718239f, 0.008593013510107994f, 0.00958659965544939f, 0.012058957479894161f, 0.009672815911471844f, 0.00926734134554863f, 0.008877898566424847f, 0.00700580608099699f, 0.03168223425745964f, 0.025925811380147934f, 0.006869640666991472f, 0.008992798626422882f, 0.005375824403017759f, 0.011887770146131516f, 0.014416535384953022f, 0.007288574241101742f, 0.0068751671351492405f, 0.007959097623825073f, 0.009136947803199291f, 0.009147659875452518f, 0.05074867606163025f, 0.010395297780632973f, 0.01749403215944767f, 0.007083158008754253f, 0.010709156282246113f, 0.004331176169216633f, 0.0035875665489584208f, 0.02197210304439068f, 0.010414067655801773f, 0.010123121552169323f, 0.005774336867034435f, 0.012355039827525616f, 0.006764338817447424f, 0.01242510974407196f, 0.013060243800282478f, 0.006089387461543083f, 0.01283574104309082f, 0.0067030382342636585f, 0.007457844913005829f, 0.012017331086099148f, 0.010523454286158085f, 0.006349916569888592f, 0.014273717068135738f, 0.0076775578781962395f, 0.006912387907505035f, 0.009695803746581078f, 0.005223949905484915f, 0.0080427760258317f, 0.007557726930826902f, 0.00683051161468029f, 0.010089782997965813f, 0.006297571584582329f, 0.006677321158349514f, 0.021894974634051323f, 0.011024688370525837f, 0.006617988459765911f, 0.018767990171909332f, 0.010205193422734737f, 0.015434905886650085f, 0.005290632601827383f, 0.009371270425617695f, 0.008626013062894344f, 0.01443642657250166f, 0.010995465330779552f, 0.008323566056787968f, 0.005215926095843315f, 0.010791723616421223f, 0.007964530028402805f, 0.00744060380384326f, 0.01298624835908413f, 0.006621724460273981f, 0.00862202513962984f, 0.008937177248299122f, 0.005487354937940836f, 0.006991228554397821f, 0.007266163360327482f, 0.006028397940099239f, 0.007290223613381386f, 0.01179941650480032f, 0.03270943835377693f, 0.005308999214321375f, 0.005183517001569271f, 0.004976438358426094f, 0.010458109900355339f, 0.007027117535471916f, 0.011343201622366905f, 0.01063445396721363f, 0.007324753329157829f, 0.007430025842040777f, 0.00969889760017395f, 0.008958819322288036f, 0.014476657845079899f, 0.009865246713161469f, 0.005934542510658503f, 0.003911782056093216f, 0.009422763250768185f, 0.011898526921868324f, 0.011100091971457005f, 0.017346417531371117f);
static const ai_u16 conv2d_89_t_out_0_shape_w_const_u16 = 2;
static const ai_u16 conv2d_89_t_out_0_shape_h_const_u16 = 2;

static const ai_u16 conv2d_90_t_in_0_shape_w_const_u16 = 2;
static const ai_u16 conv2d_90_t_in_0_shape_h_const_u16 = 2;
static const ai_u16 conv2d_90_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_90_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_90_t_in_0_shape_ch_const_u16 = 480;
static const ai_u16 conv2d_90_t_out_0_shape_ch_const_u16 = 160;
static const ai_i8 conv2d_90_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 conv2d_90_t_out_0_fmt_zero_const_s8 = 5;
static const ai_float conv2d_90_t_in_0_fmt_scale_const_f32 = 0.0235294122248888f;
static const ai_float conv2d_90_t_out_0_fmt_scale_const_f32 = 0.05142867565155029f;
static const ai_float conv2d_90_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.0017715131398290396f, 0.0009406510507687926f, 0.0006726411520503461f, 0.0008799341740086675f, 0.0012732860632240772f, 0.0005733594880439341f, 0.0009633243316784501f, 0.0006378976977430284f, 0.00082477304385975f, 0.001511955400928855f, 0.0011987702455371618f, 0.0010207929881289601f, 0.0010072715813294053f, 0.0005777154583483934f, 0.0006195581518113613f, 0.0010747455526143312f, 0.0007019223412498832f, 0.0010485026286914945f, 0.0007352920365519822f, 0.0012347216252237558f, 0.0015478201676160097f, 0.0006280088564381003f, 0.0008378365891985595f, 0.000594533805269748f, 0.0008005273994058371f, 0.0006140781915746629f, 0.0011961895506829023f, 0.0014497507363557816f, 0.001580410753376782f, 0.0019439964089542627f, 0.0014966552844271064f, 0.0010385928908362985f, 0.0005191672826185822f, 0.0011431626044213772f, 0.0007157180807553232f, 0.0011149417841807008f, 0.0006436773110181093f, 0.0007093016756698489f, 0.0005864439299330115f, 0.0014728658134117723f, 0.0007072347798384726f, 0.0005846344865858555f, 0.0006970959948375821f, 0.0007992714527063072f, 0.0017088394379243255f, 0.0008656652644276619f, 0.0011506458977237344f, 0.0007555844495072961f, 0.0007732422091066837f, 0.002417433075606823f, 0.0012449715286493301f, 0.0006166978273540735f, 0.001185555593110621f, 0.0011615378316491842f, 0.00040918169543147087f, 0.0007246099994517863f, 0.0008130231872200966f, 0.001044873846694827f, 0.0007886930834501982f, 0.001375847146846354f, 0.0007315138354897499f, 0.0008168822969309986f, 0.0006991947302594781f, 0.0006070961826480925f, 0.0005803637322969735f, 0.0008705201325938106f, 0.0007284779567271471f, 0.0009034608956426382f, 0.0009215878089889884f, 0.0007297555566765368f, 0.0005431074532680213f, 0.0012185394298285246f, 0.0008683933992870152f, 0.0007997312350198627f, 0.0016288288170471787f, 0.0007097321795299649f, 0.0009966082870960236f, 0.0007165476563386619f, 0.0024942243471741676f, 0.0013939659111201763f, 0.0006345644360408187f, 0.0009058451978489757f, 0.0011460675159469247f, 0.0013640429824590683f, 0.0006402566796168685f, 0.0005624127225019038f, 0.0012546307407319546f, 0.0018812644993886352f, 0.0006847863551229239f, 0.0010587455471977592f, 0.0007809054804965854f, 0.000778299116063863f, 0.0005320091149769723f, 0.0006378785474225879f, 0.0008330552373081446f, 0.00128097680862993f, 0.0005990001955069602f, 0.0010107350535690784f, 0.000588705122936517f, 0.0017687567742541432f, 0.0007597025251016021f, 0.0010093104792758822f, 0.0006349316681735218f, 0.0009209830895997584f, 0.0011239820159971714f, 0.0008116254466585815f, 0.0005461259861476719f, 0.0004683845618274063f, 0.0005633862456306815f, 0.0008240718161687255f, 0.0012316625798121095f, 0.000839332293253392f, 0.0011159919667989016f, 0.001936471089720726f, 0.0011251287069171667f, 0.0008012469043023884f, 0.0008721143240109086f, 0.0010588793084025383f, 0.0006497396971099079f, 0.0015997840091586113f, 0.0008577249827794731f, 0.0008755537564866245f, 0.001050286926329136f, 0.0006280741072259843f, 0.0007319872383959591f, 0.0009191860444843769f, 0.0012246462283656001f, 0.0016766332555562258f, 0.0004830593825317919f, 0.0007319535361602902f, 0.0019150745356455445f, 0.001289229723624885f, 0.0008061029366217554f, 0.0007214350625872612f, 0.0005366669502109289f, 0.0008724016370251775f, 0.000788829755038023f, 0.000945321808103472f, 0.0011814468307420611f, 0.0006933278637006879f, 0.0014793440932407975f, 0.0007708363700658083f, 0.0008436825009994209f, 0.0006400208221748471f, 0.0006479620933532715f, 0.0007120932568795979f, 0.0006865637260489166f, 0.0006308198790065944f, 0.0025896229781210423f, 0.0007337496499530971f, 0.0022821782622486353f, 0.0015130448155105114f, 0.0010242863791063428f, 0.001250705448910594f, 0.000502976356074214f, 0.0007050593849271536f, 0.0007113692117854953f, 0.0007557585486210883f, 0.000983655801974237f, 0.0009043748141266406f);
static const ai_layer_format_type conv2d_90_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;

static const ai_u16 conv2d_91_t_in_0_shape_w_const_u16 = 2;
static const ai_u16 conv2d_91_t_in_0_shape_h_const_u16 = 2;
static const ai_u16 conv2d_91_l_stride_1_const_u16 = 1;
static const ai_u16 conv2d_91_l_stride_0_const_u16 = 1;
static const ai_u16 conv2d_91_t_in_0_shape_ch_const_u16 = 160;
static const ai_u16 conv2d_91_t_out_0_shape_ch_const_u16 = 1280;
static const ai_i8 conv2d_91_t_in_0_fmt_zero_const_s8 = 5;
static const ai_i8 conv2d_91_t_out_0_fmt_zero_const_s8 = -128;
static const ai_float conv2d_91_t_in_0_fmt_scale_const_f32 = 0.05142867565155029f;
static const ai_float conv2d_91_t_out_0_fmt_scale_const_f32 = 0.013553140684962273f;
static const ai_float conv2d_91_t_weight_0_fmt_scale_const_f32[] = LITE_ARRAY_VALUES(0.00042715269955806434f, 0.0005225383210927248f, 0.0003395997337065637f, 0.0003646909026429057f, 0.00030433584470301867f, 0.00047595473006367683f, 0.0004463272634893656f, 0.0004682439030148089f, 0.0003226699773222208f, 0.0003844908205792308f, 0.00037468268419615924f, 0.0002997984120156616f, 0.0004910691059194505f, 0.0005102772847749293f, 0.0008009260054677725f, 0.00035836221650242805f, 0.0003718479711096734f, 0.0003668189456220716f, 0.00043753712088800967f, 0.00041891325963661075f, 0.0003964523784816265f, 0.0005036162910982966f, 0.0004875183221884072f, 0.00036143267061561346f, 0.00040035604615695775f, 0.0004878824111074209f, 0.00041990421595983207f, 0.00038544824928976595f, 0.00041927635902538896f, 0.0005796384066343307f, 0.0004404533829074353f, 0.0004887237446382642f, 0.0005120389978401363f, 0.00047126278514042497f, 0.00033754660398699343f, 0.00031060868059284985f, 0.00046762992860749364f, 0.0005466474685817957f, 0.0007736615953035653f, 0.00038813662831671536f, 0.00048078291001729667f, 0.0005074469372630119f, 0.0004561844398267567f, 0.00037568467087112367f, 0.0005645035416819155f, 0.00034119453630410135f, 0.00039976098923943937f, 0.00035098398802801967f, 0.00032931938767433167f, 0.0005797111080028117f, 0.0005047140875831246f, 0.00046598888002336025f, 0.0003815964446403086f, 0.0006418803241103888f, 0.000505503558088094f, 0.0004036154132336378f, 0.0003601027128752321f, 0.0005081731360405684f, 0.00044310910743661225f, 0.0006112566334195435f, 0.00041570814209990203f, 0.0004719091230072081f, 0.0004138964868616313f, 0.000308954156935215f, 0.0005274341092444956f, 0.0004993719630874693f, 0.000495590444188565f, 0.0004435635928530246f, 0.0003677677013911307f, 0.00046061500324867666f, 0.0004267739423085004f, 0.0006140826735645533f, 0.0005750006530433893f, 0.00043393822852522135f, 0.00039451027987524867f, 0.0006200403440743685f, 0.00042957166442647576f, 0.0004768487415276468f, 0.0004954551695846021f, 0.0004443945363163948f, 0.0005817434866912663f, 0.000562708533834666f, 0.0004331275122240186f, 0.00047099977382458746f, 0.0002620834275148809f, 0.0003511067770887166f, 0.0006756691727787256f, 0.00038598032551817596f, 0.0006038116407580674f, 0.0005101017304696143f, 0.000507170392666012f, 0.0004860134795308113f, 0.00047690424253232777f, 0.00039451514021493495f, 0.0004400661855470389f, 0.0004948923597112298f, 0.00043144478695467114f, 0.0003892368986271322f, 0.00036278332117944956f, 0.00039942481089383364f, 0.0005293952999636531f, 0.0004172415065113455f, 0.00039351388113573194f, 0.0006569649558514357f, 0.0005019857198931277f, 0.0004508289566729218f, 0.0004641778359655291f, 0.00047123315744102f, 0.0005380460643209517f, 0.00045899415272288024f, 0.0004828904347959906f, 0.00039677676977589726f, 0.0004948583664372563f, 0.0002908808528445661f, 0.000534930033609271f, 0.000616231351159513f, 0.000495229905936867f, 0.0004397628363221884f, 0.0005886692670173943f, 0.000363081693649292f, 0.0006041947635821998f, 0.0005939822876825929f, 0.0005628401413559914f, 0.0006690780282951891f, 0.000446248275693506f, 0.0003553233982529491f, 0.0005120841087773442f, 0.00040882299072109163f, 0.0006214131717570126f, 0.0004899230552837253f, 0.000423603254603222f, 0.0003640915674623102f, 0.000422398850787431f, 0.0004177673254162073f, 0.00040610163705423474f, 0.00037887372309342027f, 0.0003923348558600992f, 0.0005202051834203303f, 0.000317999511025846f, 0.0002945562591776252f, 0.00045121184666641057f, 0.0003799920086748898f, 0.0003670888254418969f, 0.0004511054721660912f, 0.00039020433905534446f, 0.0003932515683118254f, 0.00037730997428297997f, 0.00048463098937645555f, 0.00039545647450722754f, 0.0005029166932217777f, 0.0007189352181740105f, 0.00036521648871712387f, 0.0005072898929938674f, 0.00030862470157444477f, 0.00044176907977089286f, 0.000529260141775012f, 0.0004507811972871423f, 0.0003441316948737949f, 0.000326586450682953f, 0.0005159928114153445f, 0.00045912640052847564f, 0.0004928560229018331f, 0.0003604451776482165f, 0.00035716293496079743f, 0.00036122932215221226f, 0.00042374958866275847f, 0.0003733293851837516f, 0.00047403082135133445f, 0.0005513480282388628f, 0.0005729463300667703f, 0.0005974185769446194f, 0.00044819939648732543f, 0.0006597846513614058f, 0.0004305674519855529f, 0.000435892230598256f, 0.0005381330847740173f, 0.0005101732676848769f, 0.0003882807504851371f, 0.0004800109891220927f, 0.00038032224983908236f, 0.00035287137143313885f, 0.00038526361458934844f, 0.0005047843442298472f, 0.0005009590531699359f, 0.0004909446579404175f, 0.0009251504670828581f, 0.0005149803473614156f, 0.0004107954737264663f, 0.000452838750788942f, 0.0006156964809633791f, 0.00043140252819284797f, 0.0006609027623198926f, 0.0004193640488665551f, 0.0005953263607807457f, 0.0006648063426837325f, 0.0004242966533638537f, 0.0004901308566331863f, 0.00039622734766453505f, 0.0004903438384644687f, 0.0003910269879270345f, 0.00048796419287100434f, 0.0003939257294405252f, 0.0004409407265484333f, 0.0004289380449336022f, 0.0005035775247961283f, 0.0004474545712582767f, 0.0005029004532843828f, 0.000445621000835672f, 0.00033639470348134637f, 0.0004723373567685485f, 0.0003783334977924824f, 0.00040411841473542154f, 0.00045479313121177256f, 0.00037672516191378236f, 0.0005504926666617393f, 0.0004962144303135574f, 0.0007598469965159893f, 0.0004276145191397518f, 0.00027709428104572f, 0.0007346157799474895f, 0.0005670949467457831f, 0.0004250213096383959f, 0.0005514843505807221f, 0.0004081084334757179f, 0.0004613624478224665f, 0.0004140028904657811f, 0.00041918453644029796f, 0.0004319828876759857f, 0.0004638683167286217f, 0.0005194328841753304f, 0.0005579954595305026f, 0.0005750143900513649f, 0.00047862055362202227f, 0.0004858426400460303f, 0.00045184357441030443f, 0.0005401491071097553f, 0.0003661118389572948f, 0.00043170872959308326f, 0.0002795094915200025f, 0.0005258794408291578f, 0.0003984860668424517f, 0.000688315776642412f, 0.00043273085611872375f, 0.0004708447668235749f, 0.0005547007312998176f, 0.00043959886534139514f, 0.00037835678085684776f, 0.0002766480029094964f, 0.0005243014311417937f, 0.00044912160956300795f, 0.0006699254736304283f, 0.000593188451603055f, 0.00035934988409280777f, 0.0004367940709926188f, 0.0003975554427597672f, 0.00043982575880363584f, 0.0005832570022903383f, 0.0004353336989879608f, 0.0005118310218676925f, 0.00046402434236370027f, 0.0005434410995803773f, 0.000344307190971449f, 0.00052734644850716f, 0.000440664793131873f, 0.00038400100311264396f, 0.00036015716614201665f, 0.0005005101556889713f, 0.000412538560340181f, 0.0005586178158409894f, 0.00035966740688309073f, 0.0004386340151540935f, 0.00036890266346745193f, 0.00042341649532318115f, 0.00036795411142520607f, 0.00044623814756050706f, 0.0004151704488322139f, 0.0006158322212286294f, 0.000500585010740906f, 0.0003730015887413174f, 0.0004415434377733618f, 0.00035149577888660133f, 0.000496434629894793f, 0.0005075503722764552f, 0.0004315965052228421f, 0.000467941805254668f, 0.0004777013964485377f, 0.00036812006146647036f, 0.0005366930854506791f, 0.0005914795910939574f, 0.0004399433091748506f, 0.0007755671394988894f, 0.00042579142609611154f, 0.0005680622416548431f, 0.0004407615924719721f, 0.00048101512948051095f, 0.00044743582839146256f, 0.00045359940850175917f, 0.00038386383675970137f, 0.0004442801873665303f, 0.0004613468481693417f, 0.0004713554517365992f, 0.0004698720877058804f, 0.000389817520044744f, 0.0004160258686169982f, 0.0003569723921827972f, 0.0005824881955049932f, 0.0005390906007960439f, 0.000472657528007403f, 0.00041856238385662436f, 0.00040958510362543166f, 0.0003641947405412793f, 0.0005011723260395229f, 0.0005069932085461915f, 0.0005049924366176128f, 0.0003018878633156419f, 0.0004888977855443954f, 0.0003783869033213705f, 0.0006311347242444754f, 0.0005619869334623218f, 0.00041575595969334245f, 0.0005007553845643997f, 0.0005015161586925387f, 0.00039099552668631077f, 0.0003484260814730078f, 0.0003176569880452007f, 0.00042431012843735516f, 0.00040309157338924706f, 0.0004425113438628614f, 0.0004234112857375294f, 0.0003956854925490916f, 0.0005394943873398006f, 0.0005415923660621047f, 0.000561108288820833f, 0.00036323248059488833f, 0.0003396191168576479f, 0.0003166577371302992f, 0.0005991382640786469f, 0.00043961816118098795f, 0.0006642800290137529f, 0.0006623613880947232f, 0.00034393047099001706f, 0.0006523280753754079f, 0.0004305274924263358f, 0.0004872333665844053f, 0.0004411524278111756f, 0.0006807615282014012f, 0.0003475061967037618f, 0.0005038892850279808f, 0.0008129904163070023f, 0.0006413630326278508f, 0.0005401781527325511f, 0.00047966314014047384f, 0.0005073971697129309f, 0.0005097592365927994f, 0.0003663486859295517f, 0.000319260434480384f, 0.00042074802331626415f, 0.00047518807696178555f, 0.00043098023161292076f, 0.00041419811896048486f, 0.0005727388197556138f, 0.0006128055392764509f, 0.0003142647328786552f, 0.0005059331888332963f, 0.0005133678787387908f, 0.0003828116168733686f, 0.0006475568516179919f, 0.0004294320533517748f, 0.00039961194852367043f, 0.00037632891326211393f, 0.0004608571471180767f, 0.0004386840737424791f, 0.00038481736555695534f, 0.0003172829747200012f, 0.0004421396879479289f, 0.0004319475556258112f, 0.0005515863304026425f, 0.00048158777644857764f, 0.00041783961933106184f, 0.0005683001363649964f, 0.00025583134265616536f, 0.00042072474025189877f, 0.0003927444340661168f, 0.00037234678165987134f, 0.00047548222937621176f, 0.0003988626995123923f, 0.0004791291430592537f, 0.0004436318122316152f, 0.00047281739534810185f, 0.00044000172056257725f, 0.00040293566416949034f, 0.0005763961817137897f, 0.000488992256578058f, 0.00036587376962415874f, 0.0004183364799246192f, 0.0004623797140084207f, 0.0005040031392127275f, 0.0003255139454267919f, 0.0003435878606978804f, 0.0005808654241263866f, 0.0003799204423557967f, 0.0004889300325885415f, 0.0002392158639850095f, 0.000284957408439368f, 0.0005703775095753372f, 0.0005022925324738026f, 0.0004899385385215282f, 0.0005476595251820982f, 0.0005649749655276537f, 0.0005041133263148367f, 0.0003908551298081875f, 0.0004984042607247829f, 0.0007309019565582275f, 0.0007610026514157653f, 0.0005901276017539203f, 0.0005358916823752224f, 0.00041144955321215093f, 0.0004979125806130469f, 0.000499002926517278f, 0.0004748643550556153f, 0.000579287763684988f, 0.00046893642866052687f, 0.0006419846322387457f, 0.0004963793326169252f, 0.0004146707651671022f, 0.00047529765288345516f, 0.0005073142237961292f, 0.00037061035982333124f, 0.00045783541281707585f, 0.0005897164810448885f, 0.00042721041245386004f, 0.00043556359014473855f, 0.0004978199722245336f, 0.0004737235140055418f, 0.0004442774807102978f, 0.000619799830019474f, 0.00047565679415129125f, 0.00047293328680098057f, 0.000463864766061306f, 0.0005899721290916204f, 0.0005250116810202599f, 0.0004019328625872731f, 0.0006583207286894321f, 0.0004716369730886072f, 0.00042146380292251706f, 0.0004729923093691468f, 0.0005239835591055453f, 0.0003522821352817118f, 0.0003737016231752932f, 0.0004343637847341597f, 0.0005434025079011917f, 0.0006310198223218322f, 0.0004627973248716444f, 0.0004894114681519568f, 0.0004029374395031482f, 0.0005610922817140818f, 0.0003980367037001997f, 0.0004967607092112303f, 0.0004979194491170347f, 0.0004511674342211336f, 0.00034482066985219717f, 0.000651966140139848f, 0.0003675078914966434f, 0.00038493378087878227f, 0.00042698910692706704f, 0.000447409285698086f, 0.00039636684232391417f, 0.000534416816662997f, 0.0004993677139282227f, 0.0002538576954975724f, 0.00045871222391724586f, 0.00047722828458063304f, 0.0004630022740457207f, 0.0005142547306604683f, 0.00044680197606794536f, 0.0005690664984285831f, 0.000606234185397625f, 0.0005872149486094713f, 0.0005872230394743383f, 0.0004342194297350943f, 0.0005422433023341f, 0.0004782654868904501f, 0.0003713961341418326f, 0.00039763396489433944f, 0.00034720386611297727f, 0.0004412697162479162f, 0.00044443836668506265f, 0.00044209754560142756f, 0.0004753721586894244f, 0.0005870023742318153f, 0.0004015199083369225f, 0.0005067420424893498f, 0.00031509913969784975f, 0.00043893532711081207f, 0.0004970004083588719f, 0.0004973873728886247f, 0.00046086040674708784f, 0.0005789878196083009f, 0.0004088549467269331f, 0.00044872763101011515f, 0.0003659244976006448f, 0.000467644480522722f, 0.0005384859396144748f, 0.0006568956305272877f, 0.0004607659357134253f, 0.0003445552138146013f, 0.0004876543825957924f, 0.0003851625951938331f, 0.0003681029484141618f, 0.0005038267117924988f, 0.0004400828911457211f, 0.00025947412359528244f, 0.00036155301495455205f, 0.0006642848020419478f, 0.00043234057375229895f, 0.0005142794689163566f, 0.00042532116640359163f, 0.0004758724826388061f, 0.0006211412837728858f, 0.0005367014091461897f, 0.000636853335890919f, 0.00047228526091203094f, 0.0005172001547180116f, 0.000603586551733315f, 0.0005454125348478556f, 0.0004306795890443027f, 0.00037363547016866505f, 0.0004084613756276667f, 0.000875511032063514f, 0.00044961136882193387f, 0.0004601155815180391f, 0.000498628884088248f, 0.0005738908657804132f, 0.00041874690214172006f, 0.0005587245104834437f, 0.00040450930828228593f, 0.00048446745495311916f, 0.0004905402893200517f, 0.0004487541154958308f, 0.0004500867216847837f, 0.0006001569563522935f, 0.00039791615563444793f, 0.000502313079778105f, 0.00036263433867134154f, 0.0005499753169715405f, 0.0004912168951705098f, 0.0003970281395595521f, 0.0005669594975188375f, 0.00045127468183636665f, 0.0004415159346535802f, 0.0005398826324380934f, 0.00038473779568448663f, 0.00040643286774866283f, 0.0004268227785360068f, 0.0005218891892582178f, 0.00041990698082372546f, 0.0007017134921625257f, 0.0003631153085734695f, 0.0005295169539749622f, 0.0004977429052814841f, 0.0004818542511202395f, 0.0005132380174472928f, 0.00030769576551392674f, 0.0003593279980123043f, 0.00031890967511571944f, 0.000530052580870688f, 0.00035096771898679435f, 0.00029126147273927927f, 0.0004759876464959234f, 0.00041257604607380927f, 0.00042559727444313467f, 0.0003147187235299498f, 0.0004376998695079237f, 0.0003664602409116924f, 0.0004971188027411699f, 0.00042441231198608875f, 0.0006670920993201435f, 0.00051091582281515f, 0.00042078344267793f, 0.0004088805872015655f, 0.0005077020614407957f, 0.0004550106532406062f, 0.00040031387470662594f, 0.00047513333265669644f, 0.0005973720108158886f, 0.0002974511298816651f, 0.0004524129908531904f, 0.00038325978675857186f, 0.0005283058271743357f, 0.0005147427436895669f, 0.0006846156320534647f, 0.00041127813165076077f, 0.0004323893808759749f, 0.000552720099221915f, 0.0004290847573429346f, 0.0005850844900123775f, 0.0005230510723777115f, 0.00029164590523578227f, 0.0004272621881682426f, 0.0004434219154063612f, 0.0003463333414401859f, 0.0003316829679533839f, 0.00034683840931393206f, 0.0005494887009263039f, 0.00039531535003334284f, 0.0004216739325784147f, 0.0005254073184914887f, 0.0005076281959190965f, 0.00040190311847254634f, 0.0003190297866240144f, 0.00048633714322932065f, 0.0003705769486259669f, 0.00040752996574155986f, 0.0005107184406369925f, 0.00039719499181956053f, 0.0006420150166377425f, 0.0004221101989969611f, 0.00042206948273815215f, 0.0003863335296045989f, 0.0003870062646456063f, 0.0005017386865802109f, 0.0004906250396743417f, 0.000284541369182989f, 0.000341074715834111f, 0.0004405488434713334f, 0.0004655563388951123f, 0.0004068662237841636f, 0.000358286255504936f, 0.0003663079405669123f, 0.0006070630042813718f, 0.00039400506648235023f, 0.00045616045827046037f, 0.00037809499190188944f, 0.0004483216325752437f, 0.0003655891341622919f, 0.0003114728897344321f, 0.000388082058634609f, 0.0004932094016112387f, 0.00032760825706645846f, 0.0003843007143586874f, 0.00039399450179189444f, 0.00043399803689680994f, 0.00048818191862665117f, 0.000578585546463728f, 0.0004871489654760808f, 0.00045513297664001584f, 0.00032307871151715517f, 0.0007676658569835126f, 0.00038622901774942875f, 0.0005477579543367028f, 0.0004657665267586708f, 0.00039003719575703144f, 0.00045630967360921204f, 0.0006532869883812964f, 0.0004902742221020162f, 0.0007016758318059146f, 0.00047647301107645035f, 0.0004985342384316027f, 0.00033392224577255547f, 0.00041361775947734714f, 0.00042689606198109686f, 0.00033646958763711154f, 0.0005761155625805259f, 0.0004772196407429874f, 0.0006292907637543976f, 0.0005530869239009917f, 0.0004700081772170961f, 0.0004169016901869327f, 0.0004173888883087784f, 0.0006229694117791951f, 0.0003361389972269535f, 0.0003379260597284883f, 0.00040687195723876357f, 0.0005187991773709655f, 0.00034218112705275416f, 0.00037975015584379435f, 0.0003092157421633601f, 0.0005593564128503203f, 0.00036858170642517507f, 0.0005888718878850341f, 0.00038231429061852396f, 0.00046615037717856467f, 0.000506509211845696f, 0.00044882355723530054f, 0.0004558615619316697f, 0.0003045671619474888f, 0.0003942126641049981f, 0.00042837392538785934f, 0.00039382019895128906f, 0.0006606376264244318f, 0.0005273733986541629f, 0.00039409787859767675f, 0.0005199990700930357f, 0.0005873254267498851f, 0.00036642755731008947f, 0.00043198117054998875f, 0.000359587196726352f, 0.000628249195870012f, 0.0005532249924726784f, 0.0006578632746823132f, 0.00033041214919649065f, 0.00042447700980119407f, 0.00045591031084768474f, 0.0007959668291732669f, 0.00037441434687934816f, 0.0004275025276001543f, 0.00045160602894611657f, 0.00032360642217099667f, 0.00040200797957368195f, 0.00044187516323290765f, 0.0003550405672285706f, 0.0004410599358379841f, 0.0005512309144251049f, 0.0004503222880885005f, 0.0005853974726051092f, 0.00043206423288211226f, 0.0003902872558683157f, 0.00038474530447274446f, 0.0005700819310732186f, 0.00035578233655542135f, 0.0004125538980588317f, 0.0003945073112845421f, 0.0005577938281930983f, 0.00031677671358920634f, 0.0005296385497786105f, 0.0004165638529229909f, 0.0005298425094224513f, 0.0004732728993985802f, 0.0004756764101330191f, 0.0004918111953884363f, 0.0003575199516490102f, 0.0003744947025552392f, 0.00041797550511546433f, 0.000621895887888968f, 0.0006764346035197377f, 0.000532951729837805f, 0.00038947389111854136f, 0.00033797038486227393f, 0.00049681676318869f, 0.0003695411724038422f, 0.00041734843398444355f, 0.00048703645006753504f, 0.0006510383682325482f, 0.0004458102921489626f, 0.00038372582639567554f, 0.000466206663986668f, 0.0004466840182431042f, 0.00045178690925240517f, 0.0006897709681652486f, 0.0005618655704893172f, 0.0005759753985330462f, 0.0004996522329747677f, 0.0005248589441180229f, 0.00046126884990371764f, 0.0004701652214862406f, 0.0007155057392083108f, 0.0004752874083351344f, 0.0003963389026466757f, 0.0005756565369665623f, 0.000695022230502218f, 0.0003406230534892529f, 0.0004699107666965574f, 0.0004534308682195842f, 0.00058373948559165f, 0.00042496604146435857f, 0.0004041410284116864f, 0.0005035424837842584f, 0.0004734898393508047f, 0.0006409472553059459f, 0.00040671334136277437f, 0.0005001162644475698f, 0.0004357780562713742f, 0.0004122214450035244f, 0.0004454839800018817f, 0.0006140340701676905f, 0.0004579112573992461f, 0.0005567903281189501f, 0.0004045180685352534f, 0.0005685516516678035f, 0.0003063504700548947f, 0.0004941367078572512f, 0.0005470013129524887f, 0.0005090548074804246f, 0.00048645990318618715f, 0.000351080612745136f, 0.0003069415397476405f, 0.0005425341078080237f, 0.00042816426139324903f, 0.00029069717857055366f, 0.0005990542704239488f, 0.00046843732707202435f, 0.00047499139327555895f, 0.00039107471820898354f, 0.0005323879886418581f, 0.0003878138377331197f, 0.0005006720894016325f, 0.0003223548410460353f, 0.0005661919713020325f, 0.00046801945427432656f, 0.0005679804598912597f, 0.0005182755994610488f, 0.0004258856934029609f, 0.0004272504011169076f, 0.000433360633905977f, 0.0004675217205658555f, 0.00034850920201279223f, 0.00035304136690683663f, 0.0003878435236401856f, 0.0004457197501324117f, 0.0005175926489755511f, 0.0005923319840803742f, 0.0004370379028841853f, 0.0004545822157524526f, 0.0003437432460486889f, 0.000606766901910305f, 0.0005473206401802599f, 0.0007610725006088614f, 0.0006768144085071981f, 0.0005048267194069922f, 0.0005536213866434991f, 0.0005573718808591366f, 0.0005319592892192304f, 0.000561149325221777f, 0.0005004623089917004f, 0.000426234066253528f, 0.0005122938891872764f, 0.000513562757987529f, 0.0004881329950876534f, 0.00048470593173988163f, 0.0006720125093124807f, 0.0004799611051566899f, 0.00040918870945461094f, 0.0005056610680185258f, 0.0004438704636413604f, 0.0005857969517819583f, 0.0006531403050757945f, 0.0004664654843509197f, 0.0005251487018540502f, 0.0003851363144349307f, 0.0005748294061049819f, 0.0003789353941101581f, 0.000469942664494738f, 0.0005876262439414859f, 0.0005099948612041771f, 0.00044889620039612055f, 0.0004267852345947176f, 0.0008299443870782852f, 0.0004509538412094116f, 0.0003767981834243983f, 0.0007121001253835857f, 0.0005157153937034309f, 0.0003824399318546057f, 0.0003826049214694649f, 0.0004219625552650541f, 0.0005796089535579085f, 0.0004978086799383163f, 0.0004718912532553077f, 0.0004118740907870233f, 0.0006704804254695773f, 0.0003117854648735374f, 0.00039817759534344077f, 0.00047622862621210515f, 0.0005530717899091542f, 0.00044379584142006934f, 0.0008964499575085938f, 0.000440005911514163f, 0.00044715346302837133f, 0.00040748785249888897f, 0.0006475233240053058f, 0.0006567337550222874f, 0.00038316854625009f, 0.00034972006687894464f, 0.0004980102530680597f, 0.0006587016396224499f, 0.0003910028899554163f, 0.0004472391156014055f, 0.0004082011291757226f, 0.00044734266703017056f, 0.0004655345401261002f, 0.00036023653228767216f, 0.0005232652765698731f, 0.000766118464525789f, 0.0004478306509554386f, 0.00046688172733411193f, 0.0004898817278444767f, 0.00029841356445103884f, 0.00026062288088724017f, 0.0005081009585410357f, 0.0005041712429374456f, 0.0005286057130433619f, 0.0004977852804586291f, 0.0004349203663878143f, 0.00048296237946487963f, 0.0003864858590532094f, 0.00046392466174438596f, 0.00044535851338878274f, 0.00048627523938193917f, 0.0004709715722128749f, 0.000469837716082111f, 0.0006193344597704709f, 0.00041343289194628596f, 0.000632463488727808f, 0.0005763944354839623f, 0.0004360006423667073f, 0.0005724585498683155f, 0.0003792638599406928f, 0.00045792708988301456f, 0.0004785527416970581f, 0.00037313817301765084f, 0.0004779753217007965f, 0.0005351710715331137f, 0.00038349287933669984f, 0.0005101479473523796f, 0.0005144871538504958f, 0.0005397769855335355f, 0.0005035012727603316f, 0.00046420670696534216f, 0.000389849825296551f, 0.00041123360279016197f, 0.0005458321538753808f, 0.000544797396287322f, 0.00038715096889063716f, 0.0005304788937792182f, 0.00044995686039328575f, 0.00048568780766800046f, 0.0004591559700202197f, 0.00038127656443975866f, 0.00041766639333218336f, 0.0005431206664070487f, 0.0005057642702013254f, 0.00038596641388721764f, 0.0003750296018552035f, 0.0004566502757370472f, 0.0005522727151401341f, 0.00044348445953801274f, 0.0004802294715773314f, 0.0003905644698534161f, 0.00043834131793119013f, 0.00039835862116888165f, 0.0005453962367027998f, 0.0003891188825946301f, 0.0003601142961997539f, 0.0004829576355405152f, 0.0003871890075970441f, 0.0004761474556289613f, 0.000382382539100945f, 0.0006001713918522f, 0.0005043066339567304f, 0.00040028843795880675f, 0.0005056567024439573f, 0.00047278532292693853f, 0.00040769000770524144f, 0.0004963850951753557f, 0.0004553670296445489f, 0.0005863956175744534f, 0.000392138899769634f, 0.00044154468923807144f, 0.0003250688605476171f, 0.00048650644021108747f, 0.0004930611466988921f, 0.0004010694974567741f, 0.0006349153118208051f, 0.0005608749343082309f, 0.0004744134203065187f, 0.0005768740666098893f, 0.00034762584255076945f, 0.0004711703513748944f, 0.00033371904282830656f, 0.00041413609869778156f, 0.0004881745553575456f, 0.0003919542068615556f, 0.0002950164780486375f, 0.000438727845903486f, 0.0004020997148472816f, 0.0006426329491659999f, 0.00042123772436752915f, 0.00042619771556928754f, 0.00046096977894194424f, 0.0005946962628513575f, 0.0004226563323754817f, 0.00027617349405772984f, 0.0006891158409416676f, 0.0004450361884664744f, 0.00040142866782844067f, 0.0003730502212420106f, 0.0004519635112956166f, 0.00024118242436088622f, 0.00036590415402315557f, 0.0003431555815041065f, 0.0002909154864028096f, 0.00036421266850084066f, 0.0004339441074989736f, 0.0005271190893836319f, 0.00042874287464655936f, 0.000388051790650934f, 0.0009009228670038283f, 0.0005474347271956503f, 0.0005139624699950218f, 0.000404283928219229f, 0.00046740041580051184f, 0.0004790810926351696f, 0.00036418455420061946f, 0.00046347774332389235f, 0.00037327123573049903f, 0.0005368882557377219f, 0.0005049987812526524f, 0.0004619326791726053f, 0.0005483600543811917f, 0.0004771020612679422f, 0.0005499232211150229f, 0.000509843637701124f, 0.0003892523527611047f, 0.00034280229010619223f, 0.00038684418541379273f, 0.0005047753220424056f, 0.0005476564401760697f, 0.0007355404086410999f, 0.000489245168864727f, 0.0004954513278789818f, 0.0005239613819867373f, 0.0004963977844454348f, 0.0006912095122970641f, 0.0004696339019574225f, 0.00043411730439402163f, 0.0005741564673371613f, 0.0005296536255627871f, 0.0003758109814953059f, 0.00047370343236252666f, 0.0004132665926590562f, 0.0004419808101374656f, 0.00044455856550484896f, 0.0005823801620863378f, 0.00041175633668899536f, 0.0005027067963965237f, 0.0003925322089344263f, 0.0003355529624968767f, 0.00038272386882454157f, 0.0002514345687814057f, 0.000468366575660184f, 0.0004001414927188307f, 0.0006855610990896821f, 0.0004713721282314509f, 0.000690726563334465f, 0.0006628418923355639f, 0.0005111239152029157f, 0.00041195141966454685f, 0.0006382956635206938f, 0.0005144299939274788f, 0.00033004730357788503f, 0.0005218544974923134f, 0.0005878869560547173f, 0.00042504374869167805f, 0.00038057577330619097f, 0.00047490448923781514f, 0.00045238525490276515f, 0.0005313833826221526f, 0.00037452715332619846f, 0.0004901428474113345f, 0.0004555776540655643f, 0.00041694138781167567f, 0.0006112098344601691f, 0.00046135878073982894f, 0.0003928139922209084f, 0.00033695052843540907f, 0.000573511584661901f, 0.00034647408756427467f, 0.0005849209846928716f, 0.000376273033907637f, 0.00044600627734325826f, 0.00046943037887103856f, 0.0005164974136278033f, 0.0005058079259470105f, 0.0004176962829660624f, 0.0004512286395765841f, 0.0005477056256495416f, 0.00042362610111013055f, 0.00043894906411878765f, 0.0003826078900601715f, 0.0005527429166249931f, 0.000530596764292568f, 0.00045159467845223844f, 0.0004521882510744035f, 0.0004118848009966314f, 0.0003940992464777082f, 0.0002840991073753685f, 0.00048505549784749746f, 0.0005091993371024728f, 0.0003761073458008468f, 0.0005010519525967538f, 0.00043892371468245983f, 0.00038505482370965183f, 0.0004225292650517076f, 0.0004042784566991031f, 0.0004228183825034648f, 0.0003607978578656912f, 0.0006358203245326877f, 0.0004959821235388517f, 0.00043245768756605685f, 0.00043738787644542754f, 0.0003761797270271927f, 0.0006115032592788339f, 0.0004936847253702581f, 0.00039286454557441175f, 0.0005483904969878495f, 0.00042261710041202605f, 0.0004695022071246058f, 0.0005087414174340665f, 0.0004142995458096266f, 0.0005096127279102802f, 0.00049460434820503f, 0.0004820648464374244f, 0.0004363762272987515f, 0.00042626092908903956f, 0.00035315557033754885f, 0.00040623368113301694f, 0.0006447809282690287f, 0.00038427035906352103f, 0.0006485410267487168f, 0.0004224550793878734f, 0.0003926134086214006f, 0.0004362925246823579f, 0.0004525084514170885f, 0.0004952094750478864f, 0.0005267132655717432f, 0.0003870461951009929f, 0.0005839859368279576f, 0.00033511375659145415f, 0.0004230074118822813f, 0.0004922095686197281f, 0.0004068648850079626f, 0.0005144625320099294f, 0.0006853860686533153f, 0.0003488542861305177f, 0.0003952999541070312f, 0.000483338808408007f, 0.0004356029094196856f, 0.0005859445664100349f, 0.0003410175850149244f, 0.0004731013614218682f, 0.0004144801932852715f, 0.0005903103738091886f, 0.00043117572204209864f, 0.0002801123191602528f, 0.0004904309753328562f, 0.00038327567745000124f, 0.00037164409877732396f, 0.0004859602195210755f, 0.0005428442382253706f, 0.0004755269910674542f, 0.00033946006442420185f, 0.0005450388416647911f, 0.0004685067688114941f, 0.000526272167917341f, 0.0004960033693350852f, 0.0006261623348109424f, 0.0004191540938336402f, 0.0005733980215154588f, 0.0002914460201282054f, 0.0006885816110298038f, 0.0006155148730613291f, 0.0004446442180778831f, 0.000563447829335928f, 0.0004386389919091016f, 0.0004683911975007504f, 0.00048702640924602747f, 0.0004979665973223746f, 0.0003941894683521241f, 0.0005029917228966951f, 0.0005778573104180396f, 0.0005055901710875332f, 0.000521879002917558f, 0.000613223819527775f, 0.000329646747559309f, 0.0005231942050158978f, 0.00032512302277609706f, 0.0005290533881634474f, 0.0004829531826544553f, 0.0004251656064298004f, 0.00037748171598650515f, 0.0004422882047947496f, 0.0006212979205884039f, 0.00041739136213436723f, 0.0004760451556649059f, 0.0005464946734718978f, 0.00043389215716160834f, 0.00047074840404093266f, 0.0005232407711446285f, 0.00043791302596218884f, 0.0004914064193144441f, 0.0004093954630661756f, 0.0004500389040913433f, 0.0004413298156578094f, 0.0004430428962223232f, 0.00047437180182896554f, 0.00039578997530043125f, 0.00043341994751244783f, 0.00032525835558772087f, 0.0002500268747098744f, 0.0004198392271064222f, 0.0005177356069907546f, 0.0004498582275118679f, 0.0005147424526512623f, 0.0004900910425931215f, 0.000509329023770988f, 0.0004308836942072958f, 0.00043392073712311685f, 0.0005810823058709502f, 0.0003304160782136023f, 0.00042357356869615614f, 0.0004971853923052549f, 0.0004477245092857629f, 0.0006300545064732432f, 0.0004044606175739318f, 0.000539930711966008f, 0.0005913382046855986f, 0.0003845321771223098f, 0.0005607652710750699f, 0.0004430516273714602f, 0.000420751137426123f, 0.0005831193411722779f, 0.0003783704014495015f, 0.00039956567343324423f, 0.00045048081665299833f, 0.0005261553451418877f, 0.0005438850494101644f, 0.0005795076722279191f, 0.0005504590226337314f, 0.00042160338489338756f, 0.0005237326840870082f, 0.00035229380591772497f, 0.00042188377119600773f, 0.00035939604276791215f, 0.00041480216896161437f, 0.0005362287629395723f, 0.00035808226675726473f, 0.0007079290226101875f, 0.000373669492546469f, 0.0005619714502245188f, 0.0004719882272183895f, 0.00040939313475973904f, 0.0004887759569101036f, 0.0004662852152250707f, 0.00043774774530902505f, 0.00038720431621186435f, 0.0004931265139020979f, 0.00031616693013347685f, 0.00037404592148959637f, 0.0005125069292262197f, 0.0003871462249662727f, 0.00046730684698559344f, 0.0004379501915536821f, 0.000459415401564911f, 0.0004565862473100424f, 0.00033610695390962064f, 0.00047573185293003917f, 0.0004874411679338664f, 0.0004092940653208643f, 0.00040734701906330884f, 0.00036571803502738476f, 0.000541241024620831f, 0.0003764161956496537f, 0.00032984698191285133f, 0.0004278687993064523f, 0.00045630449312739074f, 0.0005880303215235472f, 0.0005324960802681744f, 0.0003908966318704188f, 0.00034403707832098007f, 0.0006384883308783174f, 0.0004238158871885389f, 0.0003922100877389312f, 0.0005840451340191066f, 0.00039284012746065855f, 0.00032165274024009705f, 0.00039634021231904626f, 0.0008308414253406227f, 0.00043953873682767153f, 0.00042147221392951906f, 0.00039481231942772865f, 0.0004716057737823576f);
static const ai_layer_format_type conv2d_91_l_out_ch_format_const_layer_format_type = AI_LAYER_FORMAT_CHANNEL_LAST_VALID;


static const ai_i8 gemm_93_t_in_0_fmt_zero_const_s8 = -128;
static const ai_i8 gemm_93_t_out_0_fmt_zero_const_s8 = 66;
static const ai_u16 gemm_93_t_in_0_shape_ch_const_u16 = 1280;
static const ai_u16 gemm_93_t_out_0_shape_ch_const_u16 = 27;
static const ai_u32 gemm_93_t_out_0_shape_h_w_prod_const_u32 = 1;
static const ai_float gemm_93_t_in_0_fmt_scale_const_f32 = 0.007074117660522461f;
static const ai_float gemm_93_t_out_0_fmt_scale_const_f32 = 0.18785002827644348f;
static const ai_float gemm_93_t_weight_0_fmt_scale_const_f32 = 0.005668801721185446f;










STAI_API_ENTRY
stai_return_code stai_student_c1j_ptq_int8_run(
  stai_network* network,
  const stai_run_mode mode)
{
   STAI_UNUSED(mode)
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_ACTIVATIONS) != STAI_FLAG_ACTIVATIONS,
        STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_INPUTS) != STAI_FLAG_INPUTS,
                  STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_OUTPUTS) != STAI_FLAG_OUTPUTS,
                  STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)

  _STAI_SET_ERROR(net_ctx, (net_ctx->_flags & STAI_FLAG_WEIGHTS) != STAI_FLAG_WEIGHTS,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)


  /* LITE_KERNEL_SECTION BEGIN conv2d_0 */
  {
      const ai_i8* conv2d_0_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_inputs[0] + 0);
    const ai_i8* conv2d_0_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 0);
    const ai_i32* conv2d_0_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 144);
    ai_i8* conv2d_0_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 39872);
    ai_i16* conv2d_0_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 61584);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(0, 1, {(stai_ptr) conv2d_0_t_in_0_ptr_const_s8});
    
  forward_lite_conv2d_sssa8_ch(conv2d_0_t_in_0_ptr_const_s8, conv2d_0_t_in_0_shape_w_const_u16, conv2d_0_t_in_0_shape_h_const_u16, conv2d_0_t_in_0_shape_ch_const_u16, conv2d_0_t_weight_0_ptr_const_s8, conv2d_0_t_out_0_shape_ch_const_u16, conv2d_0_t_weight_0_shape_w_const_u16, conv2d_0_t_weight_0_shape_h_const_u16, conv2d_0_l_stride_1_const_u16, conv2d_0_l_stride_0_const_u16, conv2d_0_l_pad_W_0_const_s32, conv2d_0_l_pad_H_0_const_s32, conv2d_0_t_weight_1_ptr_const_s32, conv2d_0_t_in_0_fmt_zero_const_s8, conv2d_0_t_out_0_fmt_zero_const_s8, conv2d_0_t_in_0_fmt_scale_const_f32, conv2d_0_t_out_0_fmt_scale_const_f32, conv2d_0_t_weight_0_fmt_scale_const_f32, conv2d_0_l_out_ch_format_const_layer_format_type, conv2d_0_t_out_0_ptr_s8, conv2d_0_t_out_0_shape_w_const_u16, conv2d_0_t_out_0_shape_h_const_u16, 1, 548, conv2d_0_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(0, 1, {(stai_ptr) conv2d_0_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_0 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_1_pad_before */
  {
      const ai_ptr conv2d_1_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 39872);
    ai_ptr conv2d_1_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 37760);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(1, 1, {(stai_ptr) conv2d_1_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_1_pad_before_t_in_0_ptr_const_ptr, conv2d_1_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_1_pad_before_v_pad_constant_value_const_s8), conv2d_1_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_1_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(512), (ai_i32)(544), (ai_i32)(544), (ai_i32)(16), (ai_i32)(16));
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(1, 1, {(stai_ptr) conv2d_1_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_1_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_1 */
  {
      const ai_i8* conv2d_1_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 37760);
    const ai_i8* conv2d_1_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 208);
    const ai_i32* conv2d_1_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 352);
    ai_i8* conv2d_1_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 37248);
    ai_i16* conv2d_1_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 61584);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(1, 1, {(stai_ptr) conv2d_1_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_1_t_in_0_ptr_const_s8, conv2d_1_t_in_0_shape_w_const_u16, conv2d_1_t_in_0_shape_h_const_u16, conv2d_1_t_in_0_shape_ch_const_u16, conv2d_1_t_weight_0_ptr_const_s8, conv2d_1_l_stride_1_const_u16, conv2d_1_l_stride_0_const_u16, conv2d_1_t_weight_1_ptr_const_s32, conv2d_1_t_in_0_fmt_zero_const_s8, conv2d_1_t_out_0_fmt_zero_const_s8, conv2d_1_t_in_0_fmt_scale_const_f32, conv2d_1_t_out_0_fmt_scale_const_f32, conv2d_1_t_weight_0_fmt_scale_const_f32, conv2d_1_t_out_0_ptr_s8, conv2d_1_t_out_0_shape_w_const_u16, conv2d_1_t_out_0_shape_h_const_u16, 0, 593, conv2d_1_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(1, 1, {(stai_ptr) conv2d_1_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_1 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_2 */
  {
      const ai_i8* conv2d_2_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 37248);
    const ai_i8* conv2d_2_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 416);
    const ai_i32* conv2d_2_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 544);
    ai_i8* conv2d_2_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 61584);
    ai_i16* conv2d_2_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 37104);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(2, 1, {(stai_ptr) conv2d_2_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_2_t_in_0_ptr_const_s8, conv2d_2_t_in_0_shape_w_const_u16, conv2d_2_t_in_0_shape_h_const_u16, conv2d_2_l_stride_1_const_u16, conv2d_2_l_stride_0_const_u16, conv2d_2_t_in_0_shape_ch_const_u16, conv2d_2_t_weight_0_ptr_const_s8, conv2d_2_t_out_0_shape_ch_const_u16, conv2d_2_t_weight_1_ptr_const_s32, conv2d_2_t_in_0_fmt_zero_const_s8, conv2d_2_t_out_0_fmt_zero_const_s8, conv2d_2_t_in_0_fmt_scale_const_f32, conv2d_2_t_out_0_fmt_scale_const_f32, conv2d_2_t_weight_0_fmt_scale_const_f32, conv2d_2_l_out_ch_format_const_layer_format_type, conv2d_2_t_out_0_ptr_s8, 1, 144, conv2d_2_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(2, 1, {(stai_ptr) conv2d_2_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_2 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_3 */
  {
      const ai_i8* conv2d_3_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 61584);
    const ai_i8* conv2d_3_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 576);
    const ai_i32* conv2d_3_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 960);
    ai_i8* conv2d_3_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7104);
    ai_i16* conv2d_3_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 60232);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(3, 1, {(stai_ptr) conv2d_3_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_3_t_in_0_ptr_const_s8, conv2d_3_t_in_0_shape_w_const_u16, conv2d_3_t_in_0_shape_h_const_u16, conv2d_3_l_stride_1_const_u16, conv2d_3_l_stride_0_const_u16, conv2d_3_t_in_0_shape_ch_const_u16, conv2d_3_t_weight_0_ptr_const_s8, conv2d_3_t_out_0_shape_ch_const_u16, conv2d_3_t_weight_1_ptr_const_s32, conv2d_3_t_in_0_fmt_zero_const_s8, conv2d_3_t_out_0_fmt_zero_const_s8, conv2d_3_t_in_0_fmt_scale_const_f32, conv2d_3_t_out_0_fmt_scale_const_f32, conv2d_3_t_weight_0_fmt_scale_const_f32, conv2d_3_l_out_ch_format_const_layer_format_type, conv2d_3_t_out_0_ptr_s8, 1, 512, conv2d_3_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(3, 1, {(stai_ptr) conv2d_3_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_3 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_4_pad_before */
  {
      const ai_ptr conv2d_4_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 7104);
    ai_ptr conv2d_4_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 768);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(4, 1, {(stai_ptr) conv2d_4_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_4_pad_before_t_in_0_ptr_const_ptr, conv2d_4_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_4_pad_before_v_pad_constant_value_const_s8), conv2d_4_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_4_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1536), (ai_i32)(0), (ai_i32)(3264), (ai_i32)(0), (ai_i32)(96));
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(4, 1, {(stai_ptr) conv2d_4_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_4_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_4 */
  {
      const ai_i8* conv2d_4_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 768);
    const ai_i8* conv2d_4_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1152);
    const ai_i32* conv2d_4_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 1584);
    ai_i8* conv2d_4_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    ai_i16* conv2d_4_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 67996);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(4, 1, {(stai_ptr) conv2d_4_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_4_t_in_0_ptr_const_s8, conv2d_4_t_in_0_shape_w_const_u16, conv2d_4_t_in_0_shape_h_const_u16, conv2d_4_t_in_0_shape_ch_const_u16, conv2d_4_t_weight_0_ptr_const_s8, conv2d_4_l_stride_1_const_u16, conv2d_4_l_stride_0_const_u16, conv2d_4_t_weight_1_ptr_const_s32, conv2d_4_t_in_0_fmt_zero_const_s8, conv2d_4_t_out_0_fmt_zero_const_s8, conv2d_4_t_in_0_fmt_scale_const_f32, conv2d_4_t_out_0_fmt_scale_const_f32, conv2d_4_t_weight_0_fmt_scale_const_f32, conv2d_4_t_out_0_ptr_s8, conv2d_4_t_out_0_shape_w_const_u16, conv2d_4_t_out_0_shape_h_const_u16, 0, 1777, conv2d_4_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(4, 1, {(stai_ptr) conv2d_4_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_4 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_5 */
  {
      const ai_i8* conv2d_5_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    const ai_i8* conv2d_5_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 1776);
    const ai_i32* conv2d_5_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 2544);
    ai_i8* conv2d_5_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 12640);
    ai_i16* conv2d_5_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 12288);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(5, 1, {(stai_ptr) conv2d_5_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_5_t_in_0_ptr_const_s8, conv2d_5_t_in_0_shape_w_const_u16, conv2d_5_t_in_0_shape_h_const_u16, conv2d_5_l_stride_1_const_u16, conv2d_5_l_stride_0_const_u16, conv2d_5_t_in_0_shape_ch_const_u16, conv2d_5_t_weight_0_ptr_const_s8, conv2d_5_t_out_0_shape_ch_const_u16, conv2d_5_t_weight_1_ptr_const_s32, conv2d_5_t_in_0_fmt_zero_const_s8, conv2d_5_t_out_0_fmt_zero_const_s8, conv2d_5_t_in_0_fmt_scale_const_f32, conv2d_5_t_out_0_fmt_scale_const_f32, conv2d_5_t_weight_0_fmt_scale_const_f32, conv2d_5_l_out_ch_format_const_layer_format_type, conv2d_5_t_out_0_ptr_s8, 1, 352, conv2d_5_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(5, 1, {(stai_ptr) conv2d_5_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_5 */
  /* LITE_KERNEL_SECTION BEGIN slice_7 */
  {
    
  forward_lite_slice_slice_7(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_7 */
  /* LITE_KERNEL_SECTION BEGIN concat_8 */
  {
    
  forward_lite_concat_concat_8(net_ctx);
  }
  /* LITE_KERNEL_SECTION END concat_8 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_9 */
  {
      const ai_i8* conv2d_9_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 3584);
    const ai_i8* conv2d_9_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 2608);
    const ai_i32* conv2d_9_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 4144);
    ai_i8* conv2d_9_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 31680);
    ai_i16* conv2d_9_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(9, 1, {(stai_ptr) conv2d_9_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_9_t_in_0_ptr_const_s8, conv2d_9_t_in_0_shape_w_const_u16, conv2d_9_t_in_0_shape_h_const_u16, conv2d_9_l_stride_1_const_u16, conv2d_9_l_stride_0_const_u16, conv2d_9_t_in_0_shape_ch_const_u16, conv2d_9_t_weight_0_ptr_const_s8, conv2d_9_t_out_0_shape_ch_const_u16, conv2d_9_t_weight_1_ptr_const_s32, conv2d_9_t_in_0_fmt_zero_const_s8, conv2d_9_t_out_0_fmt_zero_const_s8, conv2d_9_t_in_0_fmt_scale_const_f32, conv2d_9_t_out_0_fmt_scale_const_f32, conv2d_9_t_weight_0_fmt_scale_const_f32, conv2d_9_l_out_ch_format_const_layer_format_type, conv2d_9_t_out_0_ptr_s8, 1, 1024, conv2d_9_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(9, 1, {(stai_ptr) conv2d_9_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_9 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_10_pad_before */
  {
      const ai_ptr conv2d_10_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 31680);
    ai_ptr conv2d_10_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 25152);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(10, 1, {(stai_ptr) conv2d_10_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_10_pad_before_t_in_0_ptr_const_ptr, conv2d_10_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_10_pad_before_v_pad_constant_value_const_s8), conv2d_10_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_10_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1536), (ai_i32)(1728), (ai_i32)(1728), (ai_i32)(96), (ai_i32)(96));
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(10, 1, {(stai_ptr) conv2d_10_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_10_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_10 */
  {
      const ai_i8* conv2d_10_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 25152);
    const ai_i8* conv2d_10_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 4528);
    const ai_i32* conv2d_10_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 5392);
    ai_i8* conv2d_10_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 23616);
    ai_i16* conv2d_10_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(10, 1, {(stai_ptr) conv2d_10_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_10_t_in_0_ptr_const_s8, conv2d_10_t_in_0_shape_w_const_u16, conv2d_10_t_in_0_shape_h_const_u16, conv2d_10_t_in_0_shape_ch_const_u16, conv2d_10_t_weight_0_ptr_const_s8, conv2d_10_l_stride_1_const_u16, conv2d_10_l_stride_0_const_u16, conv2d_10_t_weight_1_ptr_const_s32, conv2d_10_t_in_0_fmt_zero_const_s8, conv2d_10_t_out_0_fmt_zero_const_s8, conv2d_10_t_in_0_fmt_scale_const_f32, conv2d_10_t_out_0_fmt_scale_const_f32, conv2d_10_t_weight_0_fmt_scale_const_f32, conv2d_10_t_out_0_ptr_s8, conv2d_10_t_out_0_shape_w_const_u16, conv2d_10_t_out_0_shape_h_const_u16, 0, 3553, conv2d_10_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(10, 1, {(stai_ptr) conv2d_10_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_10 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_11 */
  {
      const ai_i8* conv2d_11_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 23616);
    const ai_i8* conv2d_11_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 5776);
    const ai_i32* conv2d_11_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 7312);
    ai_i8* conv2d_11_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7680);
    ai_i16* conv2d_11_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(11, 1, {(stai_ptr) conv2d_11_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_11_t_in_0_ptr_const_s8, conv2d_11_t_in_0_shape_w_const_u16, conv2d_11_t_in_0_shape_h_const_u16, conv2d_11_l_stride_1_const_u16, conv2d_11_l_stride_0_const_u16, conv2d_11_t_in_0_shape_ch_const_u16, conv2d_11_t_weight_0_ptr_const_s8, conv2d_11_t_out_0_shape_ch_const_u16, conv2d_11_t_weight_1_ptr_const_s32, conv2d_11_t_in_0_fmt_zero_const_s8, conv2d_11_t_out_0_fmt_zero_const_s8, conv2d_11_t_in_0_fmt_scale_const_f32, conv2d_11_t_out_0_fmt_scale_const_f32, conv2d_11_t_weight_0_fmt_scale_const_f32, conv2d_11_l_out_ch_format_const_layer_format_type, conv2d_11_t_out_0_ptr_s8, 1, 544, conv2d_11_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(11, 1, {(stai_ptr) conv2d_11_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_11 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_12 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_12(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_12 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_13 */
  {
      const ai_i8* conv2d_13_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 16736);
    const ai_i8* conv2d_13_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 7376);
    const ai_i32* conv2d_13_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 8912);
    ai_i8* conv2d_13_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 31680);
    ai_i16* conv2d_13_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(13, 1, {(stai_ptr) conv2d_13_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_13_t_in_0_ptr_const_s8, conv2d_13_t_in_0_shape_w_const_u16, conv2d_13_t_in_0_shape_h_const_u16, conv2d_13_l_stride_1_const_u16, conv2d_13_l_stride_0_const_u16, conv2d_13_t_in_0_shape_ch_const_u16, conv2d_13_t_weight_0_ptr_const_s8, conv2d_13_t_out_0_shape_ch_const_u16, conv2d_13_t_weight_1_ptr_const_s32, conv2d_13_t_in_0_fmt_zero_const_s8, conv2d_13_t_out_0_fmt_zero_const_s8, conv2d_13_t_in_0_fmt_scale_const_f32, conv2d_13_t_out_0_fmt_scale_const_f32, conv2d_13_t_weight_0_fmt_scale_const_f32, conv2d_13_l_out_ch_format_const_layer_format_type, conv2d_13_t_out_0_ptr_s8, 1, 1024, conv2d_13_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(13, 1, {(stai_ptr) conv2d_13_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_13 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_14_pad_before */
  {
      const ai_ptr conv2d_14_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 31680);
    ai_ptr conv2d_14_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 25152);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(14, 1, {(stai_ptr) conv2d_14_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_14_pad_before_t_in_0_ptr_const_ptr, conv2d_14_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_14_pad_before_v_pad_constant_value_const_s8), conv2d_14_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_14_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1536), (ai_i32)(0), (ai_i32)(3456), (ai_i32)(0), (ai_i32)(192));
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(14, 1, {(stai_ptr) conv2d_14_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_14_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_14 */
  {
      const ai_i8* conv2d_14_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 25152);
    const ai_i8* conv2d_14_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 9296);
    const ai_i32* conv2d_14_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 10160);
    ai_i8* conv2d_14_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 3556);
    ai_i16* conv2d_14_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(14, 1, {(stai_ptr) conv2d_14_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_14_t_in_0_ptr_const_s8, conv2d_14_t_in_0_shape_w_const_u16, conv2d_14_t_in_0_shape_h_const_u16, conv2d_14_t_in_0_shape_ch_const_u16, conv2d_14_t_weight_0_ptr_const_s8, conv2d_14_l_stride_1_const_u16, conv2d_14_l_stride_0_const_u16, conv2d_14_t_weight_1_ptr_const_s32, conv2d_14_t_in_0_fmt_zero_const_s8, conv2d_14_t_out_0_fmt_zero_const_s8, conv2d_14_t_in_0_fmt_scale_const_f32, conv2d_14_t_out_0_fmt_scale_const_f32, conv2d_14_t_weight_0_fmt_scale_const_f32, conv2d_14_t_out_0_ptr_s8, conv2d_14_t_out_0_shape_w_const_u16, conv2d_14_t_out_0_shape_h_const_u16, 0, 3553, conv2d_14_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(14, 1, {(stai_ptr) conv2d_14_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_14 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_15 */
  {
      const ai_i8* conv2d_15_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 3556);
    const ai_i8* conv2d_15_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 10544);
    const ai_i32* conv2d_15_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 12080);
    ai_i8* conv2d_15_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 544);
    ai_i16* conv2d_15_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(15, 1, {(stai_ptr) conv2d_15_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_15_t_in_0_ptr_const_s8, conv2d_15_t_in_0_shape_w_const_u16, conv2d_15_t_in_0_shape_h_const_u16, conv2d_15_l_stride_1_const_u16, conv2d_15_l_stride_0_const_u16, conv2d_15_t_in_0_shape_ch_const_u16, conv2d_15_t_weight_0_ptr_const_s8, conv2d_15_t_out_0_shape_ch_const_u16, conv2d_15_t_weight_1_ptr_const_s32, conv2d_15_t_in_0_fmt_zero_const_s8, conv2d_15_t_out_0_fmt_zero_const_s8, conv2d_15_t_in_0_fmt_scale_const_f32, conv2d_15_t_out_0_fmt_scale_const_f32, conv2d_15_t_weight_0_fmt_scale_const_f32, conv2d_15_l_out_ch_format_const_layer_format_type, conv2d_15_t_out_0_ptr_s8, 1, 544, conv2d_15_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(15, 1, {(stai_ptr) conv2d_15_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_15 */
  /* LITE_KERNEL_SECTION BEGIN slice_17 */
  {
    
  forward_lite_slice_slice_17(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_17 */
  /* LITE_KERNEL_SECTION BEGIN concat_18 */
  {
    
  forward_lite_concat_concat_18(net_ctx);
  }
  /* LITE_KERNEL_SECTION END concat_18 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_19 */
  {
      const ai_i8* conv2d_19_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 2464);
    const ai_i8* conv2d_19_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 12144);
    const ai_i32* conv2d_19_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 13680);
    ai_i8* conv2d_19_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 4512);
    ai_i16* conv2d_19_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 3488);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(19, 1, {(stai_ptr) conv2d_19_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_19_t_in_0_ptr_const_s8, conv2d_19_t_in_0_shape_w_const_u16, conv2d_19_t_in_0_shape_h_const_u16, conv2d_19_l_stride_1_const_u16, conv2d_19_l_stride_0_const_u16, conv2d_19_t_in_0_shape_ch_const_u16, conv2d_19_t_weight_0_ptr_const_s8, conv2d_19_t_out_0_shape_ch_const_u16, conv2d_19_t_weight_1_ptr_const_s32, conv2d_19_t_in_0_fmt_zero_const_s8, conv2d_19_t_out_0_fmt_zero_const_s8, conv2d_19_t_in_0_fmt_scale_const_f32, conv2d_19_t_out_0_fmt_scale_const_f32, conv2d_19_t_weight_0_fmt_scale_const_f32, conv2d_19_l_out_ch_format_const_layer_format_type, conv2d_19_t_out_0_ptr_s8, 1, 1024, conv2d_19_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(19, 1, {(stai_ptr) conv2d_19_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_19 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_20_pad_before */
  {
      const ai_ptr conv2d_20_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 4512);
    ai_ptr conv2d_20_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 16736);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(20, 1, {(stai_ptr) conv2d_20_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_20_pad_before_t_in_0_ptr_const_ptr, conv2d_20_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_20_pad_before_v_pad_constant_value_const_s8), conv2d_20_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_20_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(768), (ai_i32)(960), (ai_i32)(960), (ai_i32)(96), (ai_i32)(96));
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(20, 1, {(stai_ptr) conv2d_20_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_20_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_20 */
  {
      const ai_i8* conv2d_20_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 16736);
    const ai_i8* conv2d_20_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 14064);
    const ai_i32* conv2d_20_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 14928);
    ai_i8* conv2d_20_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 26336);
    ai_i16* conv2d_20_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 3488);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(20, 1, {(stai_ptr) conv2d_20_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_20_t_in_0_ptr_const_s8, conv2d_20_t_in_0_shape_w_const_u16, conv2d_20_t_in_0_shape_h_const_u16, conv2d_20_t_in_0_shape_ch_const_u16, conv2d_20_t_weight_0_ptr_const_s8, conv2d_20_l_stride_1_const_u16, conv2d_20_l_stride_0_const_u16, conv2d_20_t_weight_1_ptr_const_s32, conv2d_20_t_in_0_fmt_zero_const_s8, conv2d_20_t_out_0_fmt_zero_const_s8, conv2d_20_t_in_0_fmt_scale_const_f32, conv2d_20_t_out_0_fmt_scale_const_f32, conv2d_20_t_weight_0_fmt_scale_const_f32, conv2d_20_t_out_0_ptr_s8, conv2d_20_t_out_0_shape_w_const_u16, conv2d_20_t_out_0_shape_h_const_u16, 0, 3553, conv2d_20_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(20, 1, {(stai_ptr) conv2d_20_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_20 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_21 */
  {
      const ai_i8* conv2d_21_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 26336);
    const ai_i8* conv2d_21_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 15312);
    const ai_i32* conv2d_21_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 16848);
    ai_i8* conv2d_21_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 3488);
    ai_i16* conv2d_21_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(21, 1, {(stai_ptr) conv2d_21_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_21_t_in_0_ptr_const_s8, conv2d_21_t_in_0_shape_w_const_u16, conv2d_21_t_in_0_shape_h_const_u16, conv2d_21_l_stride_1_const_u16, conv2d_21_l_stride_0_const_u16, conv2d_21_t_in_0_shape_ch_const_u16, conv2d_21_t_weight_0_ptr_const_s8, conv2d_21_t_out_0_shape_ch_const_u16, conv2d_21_t_weight_1_ptr_const_s32, conv2d_21_t_in_0_fmt_zero_const_s8, conv2d_21_t_out_0_fmt_zero_const_s8, conv2d_21_t_in_0_fmt_scale_const_f32, conv2d_21_t_out_0_fmt_scale_const_f32, conv2d_21_t_weight_0_fmt_scale_const_f32, conv2d_21_l_out_ch_format_const_layer_format_type, conv2d_21_t_out_0_ptr_s8, 1, 544, conv2d_21_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(21, 1, {(stai_ptr) conv2d_21_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_21 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_22 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_22(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_22 */
  /* LITE_KERNEL_SECTION BEGIN slice_24 */
  {
    
  forward_lite_slice_slice_24(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_24 */
  /* LITE_KERNEL_SECTION BEGIN concat_25 */
  {
    
  forward_lite_concat_concat_25(net_ctx);
  }
  /* LITE_KERNEL_SECTION END concat_25 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_26 */
  {
      const ai_i8* conv2d_26_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 2464);
    const ai_i8* conv2d_26_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 16912);
    const ai_i32* conv2d_26_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 18448);
    ai_i8* conv2d_26_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 5536);
    ai_i16* conv2d_26_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 3488);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(26, 1, {(stai_ptr) conv2d_26_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_26_t_in_0_ptr_const_s8, conv2d_26_t_in_0_shape_w_const_u16, conv2d_26_t_in_0_shape_h_const_u16, conv2d_26_l_stride_1_const_u16, conv2d_26_l_stride_0_const_u16, conv2d_26_t_in_0_shape_ch_const_u16, conv2d_26_t_weight_0_ptr_const_s8, conv2d_26_t_out_0_shape_ch_const_u16, conv2d_26_t_weight_1_ptr_const_s32, conv2d_26_t_in_0_fmt_zero_const_s8, conv2d_26_t_out_0_fmt_zero_const_s8, conv2d_26_t_in_0_fmt_scale_const_f32, conv2d_26_t_out_0_fmt_scale_const_f32, conv2d_26_t_weight_0_fmt_scale_const_f32, conv2d_26_l_out_ch_format_const_layer_format_type, conv2d_26_t_out_0_ptr_s8, 1, 1024, conv2d_26_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(26, 1, {(stai_ptr) conv2d_26_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_26 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_27_pad_before */
  {
      const ai_ptr conv2d_27_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 5536);
    ai_ptr conv2d_27_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 16736);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(27, 1, {(stai_ptr) conv2d_27_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_27_pad_before_t_in_0_ptr_const_ptr, conv2d_27_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_27_pad_before_v_pad_constant_value_const_s8), conv2d_27_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_27_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(768), (ai_i32)(960), (ai_i32)(960), (ai_i32)(96), (ai_i32)(96));
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(27, 1, {(stai_ptr) conv2d_27_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_27_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_27 */
  {
      const ai_i8* conv2d_27_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 16736);
    const ai_i8* conv2d_27_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 18832);
    const ai_i32* conv2d_27_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 19696);
    ai_i8* conv2d_27_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 26336);
    ai_i16* conv2d_27_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 5536);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(27, 1, {(stai_ptr) conv2d_27_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_27_t_in_0_ptr_const_s8, conv2d_27_t_in_0_shape_w_const_u16, conv2d_27_t_in_0_shape_h_const_u16, conv2d_27_t_in_0_shape_ch_const_u16, conv2d_27_t_weight_0_ptr_const_s8, conv2d_27_l_stride_1_const_u16, conv2d_27_l_stride_0_const_u16, conv2d_27_t_weight_1_ptr_const_s32, conv2d_27_t_in_0_fmt_zero_const_s8, conv2d_27_t_out_0_fmt_zero_const_s8, conv2d_27_t_in_0_fmt_scale_const_f32, conv2d_27_t_out_0_fmt_scale_const_f32, conv2d_27_t_weight_0_fmt_scale_const_f32, conv2d_27_t_out_0_ptr_s8, conv2d_27_t_out_0_shape_w_const_u16, conv2d_27_t_out_0_shape_h_const_u16, 0, 3553, conv2d_27_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(27, 1, {(stai_ptr) conv2d_27_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_27 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_28 */
  {
      const ai_i8* conv2d_28_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 26336);
    const ai_i8* conv2d_28_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 20080);
    const ai_i32* conv2d_28_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 21616);
    ai_i8* conv2d_28_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 3488);
    ai_i16* conv2d_28_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 0);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(28, 1, {(stai_ptr) conv2d_28_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_28_t_in_0_ptr_const_s8, conv2d_28_t_in_0_shape_w_const_u16, conv2d_28_t_in_0_shape_h_const_u16, conv2d_28_l_stride_1_const_u16, conv2d_28_l_stride_0_const_u16, conv2d_28_t_in_0_shape_ch_const_u16, conv2d_28_t_weight_0_ptr_const_s8, conv2d_28_t_out_0_shape_ch_const_u16, conv2d_28_t_weight_1_ptr_const_s32, conv2d_28_t_in_0_fmt_zero_const_s8, conv2d_28_t_out_0_fmt_zero_const_s8, conv2d_28_t_in_0_fmt_scale_const_f32, conv2d_28_t_out_0_fmt_scale_const_f32, conv2d_28_t_weight_0_fmt_scale_const_f32, conv2d_28_l_out_ch_format_const_layer_format_type, conv2d_28_t_out_0_ptr_s8, 1, 544, conv2d_28_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(28, 1, {(stai_ptr) conv2d_28_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_28 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_29 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_29(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_29 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_30 */
  {
      const ai_i8* conv2d_30_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 5536);
    const ai_i8* conv2d_30_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 21680);
    const ai_i32* conv2d_30_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 23216);
    ai_i8* conv2d_30_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 16736);
    ai_i16* conv2d_30_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 1568);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(30, 1, {(stai_ptr) conv2d_30_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_30_t_in_0_ptr_const_s8, conv2d_30_t_in_0_shape_w_const_u16, conv2d_30_t_in_0_shape_h_const_u16, conv2d_30_l_stride_1_const_u16, conv2d_30_l_stride_0_const_u16, conv2d_30_t_in_0_shape_ch_const_u16, conv2d_30_t_weight_0_ptr_const_s8, conv2d_30_t_out_0_shape_ch_const_u16, conv2d_30_t_weight_1_ptr_const_s32, conv2d_30_t_in_0_fmt_zero_const_s8, conv2d_30_t_out_0_fmt_zero_const_s8, conv2d_30_t_in_0_fmt_scale_const_f32, conv2d_30_t_out_0_fmt_scale_const_f32, conv2d_30_t_weight_0_fmt_scale_const_f32, conv2d_30_l_out_ch_format_const_layer_format_type, conv2d_30_t_out_0_ptr_s8, 1, 1024, conv2d_30_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(30, 1, {(stai_ptr) conv2d_30_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_30 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_31_pad_before */
  {
      const ai_ptr conv2d_31_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 16736);
    ai_ptr conv2d_31_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 22880);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(31, 1, {(stai_ptr) conv2d_31_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_31_pad_before_t_in_0_ptr_const_ptr, conv2d_31_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_31_pad_before_v_pad_constant_value_const_s8), conv2d_31_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_31_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(768), (ai_i32)(0), (ai_i32)(1920), (ai_i32)(0), (ai_i32)(192));
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(31, 1, {(stai_ptr) conv2d_31_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_31_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_31 */
  {
      const ai_i8* conv2d_31_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 22880);
    const ai_i8* conv2d_31_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 23600);
    const ai_i32* conv2d_31_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 24464);
    ai_i8* conv2d_31_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 1568);
    ai_i16* conv2d_31_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 5536);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(31, 1, {(stai_ptr) conv2d_31_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_31_t_in_0_ptr_const_s8, conv2d_31_t_in_0_shape_w_const_u16, conv2d_31_t_in_0_shape_h_const_u16, conv2d_31_t_in_0_shape_ch_const_u16, conv2d_31_t_weight_0_ptr_const_s8, conv2d_31_l_stride_1_const_u16, conv2d_31_l_stride_0_const_u16, conv2d_31_t_weight_1_ptr_const_s32, conv2d_31_t_in_0_fmt_zero_const_s8, conv2d_31_t_out_0_fmt_zero_const_s8, conv2d_31_t_in_0_fmt_scale_const_f32, conv2d_31_t_out_0_fmt_scale_const_f32, conv2d_31_t_weight_0_fmt_scale_const_f32, conv2d_31_t_out_0_ptr_s8, conv2d_31_t_out_0_shape_w_const_u16, conv2d_31_t_out_0_shape_h_const_u16, 0, 3553, conv2d_31_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(31, 1, {(stai_ptr) conv2d_31_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_31 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_32 */
  {
      const ai_i8* conv2d_32_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 1568);
    const ai_i8* conv2d_32_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 24848);
    const ai_i32* conv2d_32_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 27920);
    ai_i8* conv2d_32_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 0);
    ai_i16* conv2d_32_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 3104);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(32, 1, {(stai_ptr) conv2d_32_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_32_t_in_0_ptr_const_s8, conv2d_32_t_in_0_shape_w_const_u16, conv2d_32_t_in_0_shape_h_const_u16, conv2d_32_l_stride_1_const_u16, conv2d_32_l_stride_0_const_u16, conv2d_32_t_in_0_shape_ch_const_u16, conv2d_32_t_weight_0_ptr_const_s8, conv2d_32_t_out_0_shape_ch_const_u16, conv2d_32_t_weight_1_ptr_const_s32, conv2d_32_t_in_0_fmt_zero_const_s8, conv2d_32_t_out_0_fmt_zero_const_s8, conv2d_32_t_in_0_fmt_scale_const_f32, conv2d_32_t_out_0_fmt_scale_const_f32, conv2d_32_t_weight_0_fmt_scale_const_f32, conv2d_32_l_out_ch_format_const_layer_format_type, conv2d_32_t_out_0_ptr_s8, 1, 704, conv2d_32_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(32, 1, {(stai_ptr) conv2d_32_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_32 */
  /* LITE_KERNEL_SECTION BEGIN slice_34 */
  {
    
  forward_lite_slice_slice_34(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_34 */
  /* LITE_KERNEL_SECTION BEGIN concat_35 */
  {
    
  forward_lite_concat_concat_35(net_ctx);
  }
  /* LITE_KERNEL_SECTION END concat_35 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_36 */
  {
      const ai_i8* conv2d_36_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 2016);
    const ai_i8* conv2d_36_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 28048);
    const ai_i32* conv2d_36_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 34192);
    ai_i8* conv2d_36_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7584);
    ai_i16* conv2d_36_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 5536);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(36, 1, {(stai_ptr) conv2d_36_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_36_t_in_0_ptr_const_s8, conv2d_36_t_in_0_shape_w_const_u16, conv2d_36_t_in_0_shape_h_const_u16, conv2d_36_l_stride_1_const_u16, conv2d_36_l_stride_0_const_u16, conv2d_36_t_in_0_shape_ch_const_u16, conv2d_36_t_weight_0_ptr_const_s8, conv2d_36_t_out_0_shape_ch_const_u16, conv2d_36_t_weight_1_ptr_const_s32, conv2d_36_t_in_0_fmt_zero_const_s8, conv2d_36_t_out_0_fmt_zero_const_s8, conv2d_36_t_in_0_fmt_scale_const_f32, conv2d_36_t_out_0_fmt_scale_const_f32, conv2d_36_t_weight_0_fmt_scale_const_f32, conv2d_36_l_out_ch_format_const_layer_format_type, conv2d_36_t_out_0_ptr_s8, 1, 2048, conv2d_36_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(36, 1, {(stai_ptr) conv2d_36_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_36 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_37_pad_before */
  {
      const ai_ptr conv2d_37_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 7584);
    ai_ptr conv2d_37_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 16736);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(37, 1, {(stai_ptr) conv2d_37_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_37_pad_before_t_in_0_ptr_const_ptr, conv2d_37_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_37_pad_before_v_pad_constant_value_const_s8), conv2d_37_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_37_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(768), (ai_i32)(1152), (ai_i32)(1152), (ai_i32)(192), (ai_i32)(192));
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(37, 1, {(stai_ptr) conv2d_37_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_37_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_37 */
  {
      const ai_i8* conv2d_37_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 16736);
    const ai_i8* conv2d_37_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 34960);
    const ai_i32* conv2d_37_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 36688);
    ai_i8* conv2d_37_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 5536);
    ai_i16* conv2d_37_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 23648);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(37, 1, {(stai_ptr) conv2d_37_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_37_t_in_0_ptr_const_s8, conv2d_37_t_in_0_shape_w_const_u16, conv2d_37_t_in_0_shape_h_const_u16, conv2d_37_t_in_0_shape_ch_const_u16, conv2d_37_t_weight_0_ptr_const_s8, conv2d_37_l_stride_1_const_u16, conv2d_37_l_stride_0_const_u16, conv2d_37_t_weight_1_ptr_const_s32, conv2d_37_t_in_0_fmt_zero_const_s8, conv2d_37_t_out_0_fmt_zero_const_s8, conv2d_37_t_in_0_fmt_scale_const_f32, conv2d_37_t_out_0_fmt_scale_const_f32, conv2d_37_t_weight_0_fmt_scale_const_f32, conv2d_37_t_out_0_ptr_s8, conv2d_37_t_out_0_shape_w_const_u16, conv2d_37_t_out_0_shape_h_const_u16, 0, 7105, conv2d_37_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(37, 1, {(stai_ptr) conv2d_37_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_37 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_38 */
  {
      const ai_i8* conv2d_38_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 5536);
    const ai_i8* conv2d_38_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 37456);
    const ai_i32* conv2d_38_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 43600);
    ai_i8* conv2d_38_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 3616);
    ai_i16* conv2d_38_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 2528);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(38, 1, {(stai_ptr) conv2d_38_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_38_t_in_0_ptr_const_s8, conv2d_38_t_in_0_shape_w_const_u16, conv2d_38_t_in_0_shape_h_const_u16, conv2d_38_l_stride_1_const_u16, conv2d_38_l_stride_0_const_u16, conv2d_38_t_in_0_shape_ch_const_u16, conv2d_38_t_weight_0_ptr_const_s8, conv2d_38_t_out_0_shape_ch_const_u16, conv2d_38_t_weight_1_ptr_const_s32, conv2d_38_t_in_0_fmt_zero_const_s8, conv2d_38_t_out_0_fmt_zero_const_s8, conv2d_38_t_in_0_fmt_scale_const_f32, conv2d_38_t_out_0_fmt_scale_const_f32, conv2d_38_t_weight_0_fmt_scale_const_f32, conv2d_38_l_out_ch_format_const_layer_format_type, conv2d_38_t_out_0_ptr_s8, 1, 1088, conv2d_38_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(38, 1, {(stai_ptr) conv2d_38_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_38 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_39 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_39(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_39 */
  /* LITE_KERNEL_SECTION BEGIN slice_41 */
  {
    
  forward_lite_slice_slice_41(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_41 */
  /* LITE_KERNEL_SECTION BEGIN concat_42 */
  {
    
  forward_lite_concat_concat_42(net_ctx);
  }
  /* LITE_KERNEL_SECTION END concat_42 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_43 */
  {
      const ai_i8* conv2d_43_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 2016);
    const ai_i8* conv2d_43_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 43728);
    const ai_i32* conv2d_43_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 49872);
    ai_i8* conv2d_43_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7584);
    ai_i16* conv2d_43_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 5536);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(43, 1, {(stai_ptr) conv2d_43_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_43_t_in_0_ptr_const_s8, conv2d_43_t_in_0_shape_w_const_u16, conv2d_43_t_in_0_shape_h_const_u16, conv2d_43_l_stride_1_const_u16, conv2d_43_l_stride_0_const_u16, conv2d_43_t_in_0_shape_ch_const_u16, conv2d_43_t_weight_0_ptr_const_s8, conv2d_43_t_out_0_shape_ch_const_u16, conv2d_43_t_weight_1_ptr_const_s32, conv2d_43_t_in_0_fmt_zero_const_s8, conv2d_43_t_out_0_fmt_zero_const_s8, conv2d_43_t_in_0_fmt_scale_const_f32, conv2d_43_t_out_0_fmt_scale_const_f32, conv2d_43_t_weight_0_fmt_scale_const_f32, conv2d_43_l_out_ch_format_const_layer_format_type, conv2d_43_t_out_0_ptr_s8, 1, 2048, conv2d_43_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(43, 1, {(stai_ptr) conv2d_43_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_43 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_44_pad_before */
  {
      const ai_ptr conv2d_44_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 7584);
    ai_ptr conv2d_44_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 16736);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(44, 1, {(stai_ptr) conv2d_44_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_44_pad_before_t_in_0_ptr_const_ptr, conv2d_44_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_44_pad_before_v_pad_constant_value_const_s8), conv2d_44_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_44_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(768), (ai_i32)(1152), (ai_i32)(1152), (ai_i32)(192), (ai_i32)(192));
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(44, 1, {(stai_ptr) conv2d_44_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_44_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_44 */
  {
      const ai_i8* conv2d_44_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 16736);
    const ai_i8* conv2d_44_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 50640);
    const ai_i32* conv2d_44_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 52368);
    ai_i8* conv2d_44_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 5536);
    ai_i16* conv2d_44_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 23648);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(44, 1, {(stai_ptr) conv2d_44_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_44_t_in_0_ptr_const_s8, conv2d_44_t_in_0_shape_w_const_u16, conv2d_44_t_in_0_shape_h_const_u16, conv2d_44_t_in_0_shape_ch_const_u16, conv2d_44_t_weight_0_ptr_const_s8, conv2d_44_l_stride_1_const_u16, conv2d_44_l_stride_0_const_u16, conv2d_44_t_weight_1_ptr_const_s32, conv2d_44_t_in_0_fmt_zero_const_s8, conv2d_44_t_out_0_fmt_zero_const_s8, conv2d_44_t_in_0_fmt_scale_const_f32, conv2d_44_t_out_0_fmt_scale_const_f32, conv2d_44_t_weight_0_fmt_scale_const_f32, conv2d_44_t_out_0_ptr_s8, conv2d_44_t_out_0_shape_w_const_u16, conv2d_44_t_out_0_shape_h_const_u16, 0, 7105, conv2d_44_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(44, 1, {(stai_ptr) conv2d_44_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_44 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_45 */
  {
      const ai_i8* conv2d_45_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 5536);
    const ai_i8* conv2d_45_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 53136);
    const ai_i32* conv2d_45_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 59280);
    ai_i8* conv2d_45_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8608);
    ai_i16* conv2d_45_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 3040);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(45, 1, {(stai_ptr) conv2d_45_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_45_t_in_0_ptr_const_s8, conv2d_45_t_in_0_shape_w_const_u16, conv2d_45_t_in_0_shape_h_const_u16, conv2d_45_l_stride_1_const_u16, conv2d_45_l_stride_0_const_u16, conv2d_45_t_in_0_shape_ch_const_u16, conv2d_45_t_weight_0_ptr_const_s8, conv2d_45_t_out_0_shape_ch_const_u16, conv2d_45_t_weight_1_ptr_const_s32, conv2d_45_t_in_0_fmt_zero_const_s8, conv2d_45_t_out_0_fmt_zero_const_s8, conv2d_45_t_in_0_fmt_scale_const_f32, conv2d_45_t_out_0_fmt_scale_const_f32, conv2d_45_t_weight_0_fmt_scale_const_f32, conv2d_45_l_out_ch_format_const_layer_format_type, conv2d_45_t_out_0_ptr_s8, 1, 1088, conv2d_45_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(45, 1, {(stai_ptr) conv2d_45_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_45 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_46 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_46(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_46 */
  /* LITE_KERNEL_SECTION BEGIN slice_48 */
  {
    
  forward_lite_slice_slice_48(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_48 */
  /* LITE_KERNEL_SECTION BEGIN concat_49 */
  {
    
  forward_lite_concat_concat_49(net_ctx);
  }
  /* LITE_KERNEL_SECTION END concat_49 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_50 */
  {
      const ai_i8* conv2d_50_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 2016);
    const ai_i8* conv2d_50_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 59408);
    const ai_i32* conv2d_50_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 65552);
    ai_i8* conv2d_50_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7584);
    ai_i16* conv2d_50_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 5536);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(50, 1, {(stai_ptr) conv2d_50_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_50_t_in_0_ptr_const_s8, conv2d_50_t_in_0_shape_w_const_u16, conv2d_50_t_in_0_shape_h_const_u16, conv2d_50_l_stride_1_const_u16, conv2d_50_l_stride_0_const_u16, conv2d_50_t_in_0_shape_ch_const_u16, conv2d_50_t_weight_0_ptr_const_s8, conv2d_50_t_out_0_shape_ch_const_u16, conv2d_50_t_weight_1_ptr_const_s32, conv2d_50_t_in_0_fmt_zero_const_s8, conv2d_50_t_out_0_fmt_zero_const_s8, conv2d_50_t_in_0_fmt_scale_const_f32, conv2d_50_t_out_0_fmt_scale_const_f32, conv2d_50_t_weight_0_fmt_scale_const_f32, conv2d_50_l_out_ch_format_const_layer_format_type, conv2d_50_t_out_0_ptr_s8, 1, 2048, conv2d_50_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(50, 1, {(stai_ptr) conv2d_50_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_50 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_51_pad_before */
  {
      const ai_ptr conv2d_51_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 7584);
    ai_ptr conv2d_51_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 16736);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(51, 1, {(stai_ptr) conv2d_51_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_51_pad_before_t_in_0_ptr_const_ptr, conv2d_51_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_51_pad_before_v_pad_constant_value_const_s8), conv2d_51_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_51_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(768), (ai_i32)(1152), (ai_i32)(1152), (ai_i32)(192), (ai_i32)(192));
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(51, 1, {(stai_ptr) conv2d_51_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_51_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_51 */
  {
      const ai_i8* conv2d_51_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 16736);
    const ai_i8* conv2d_51_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 66320);
    const ai_i32* conv2d_51_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 68048);
    ai_i8* conv2d_51_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 5536);
    ai_i16* conv2d_51_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 23648);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(51, 1, {(stai_ptr) conv2d_51_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_51_t_in_0_ptr_const_s8, conv2d_51_t_in_0_shape_w_const_u16, conv2d_51_t_in_0_shape_h_const_u16, conv2d_51_t_in_0_shape_ch_const_u16, conv2d_51_t_weight_0_ptr_const_s8, conv2d_51_l_stride_1_const_u16, conv2d_51_l_stride_0_const_u16, conv2d_51_t_weight_1_ptr_const_s32, conv2d_51_t_in_0_fmt_zero_const_s8, conv2d_51_t_out_0_fmt_zero_const_s8, conv2d_51_t_in_0_fmt_scale_const_f32, conv2d_51_t_out_0_fmt_scale_const_f32, conv2d_51_t_weight_0_fmt_scale_const_f32, conv2d_51_t_out_0_ptr_s8, conv2d_51_t_out_0_shape_w_const_u16, conv2d_51_t_out_0_shape_h_const_u16, 0, 7105, conv2d_51_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(51, 1, {(stai_ptr) conv2d_51_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_51 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_52 */
  {
      const ai_i8* conv2d_52_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 5536);
    const ai_i8* conv2d_52_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 68816);
    const ai_i32* conv2d_52_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 74960);
    ai_i8* conv2d_52_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 3552);
    ai_i16* conv2d_52_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 8608);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(52, 1, {(stai_ptr) conv2d_52_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_52_t_in_0_ptr_const_s8, conv2d_52_t_in_0_shape_w_const_u16, conv2d_52_t_in_0_shape_h_const_u16, conv2d_52_l_stride_1_const_u16, conv2d_52_l_stride_0_const_u16, conv2d_52_t_in_0_shape_ch_const_u16, conv2d_52_t_weight_0_ptr_const_s8, conv2d_52_t_out_0_shape_ch_const_u16, conv2d_52_t_weight_1_ptr_const_s32, conv2d_52_t_in_0_fmt_zero_const_s8, conv2d_52_t_out_0_fmt_zero_const_s8, conv2d_52_t_in_0_fmt_scale_const_f32, conv2d_52_t_out_0_fmt_scale_const_f32, conv2d_52_t_weight_0_fmt_scale_const_f32, conv2d_52_l_out_ch_format_const_layer_format_type, conv2d_52_t_out_0_ptr_s8, 1, 1088, conv2d_52_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(52, 1, {(stai_ptr) conv2d_52_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_52 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_53 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_53(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_53 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_54 */
  {
      const ai_i8* conv2d_54_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 5536);
    const ai_i8* conv2d_54_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 75088);
    const ai_i32* conv2d_54_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 81232);
    ai_i8* conv2d_54_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 8096);
    ai_i16* conv2d_54_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 6048);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(54, 1, {(stai_ptr) conv2d_54_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_54_t_in_0_ptr_const_s8, conv2d_54_t_in_0_shape_w_const_u16, conv2d_54_t_in_0_shape_h_const_u16, conv2d_54_l_stride_1_const_u16, conv2d_54_l_stride_0_const_u16, conv2d_54_t_in_0_shape_ch_const_u16, conv2d_54_t_weight_0_ptr_const_s8, conv2d_54_t_out_0_shape_ch_const_u16, conv2d_54_t_weight_1_ptr_const_s32, conv2d_54_t_in_0_fmt_zero_const_s8, conv2d_54_t_out_0_fmt_zero_const_s8, conv2d_54_t_in_0_fmt_scale_const_f32, conv2d_54_t_out_0_fmt_scale_const_f32, conv2d_54_t_weight_0_fmt_scale_const_f32, conv2d_54_l_out_ch_format_const_layer_format_type, conv2d_54_t_out_0_ptr_s8, 1, 2048, conv2d_54_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(54, 1, {(stai_ptr) conv2d_54_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_54 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_55_pad_before */
  {
      const ai_ptr conv2d_55_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 8096);
    ai_ptr conv2d_55_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 16736);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(55, 1, {(stai_ptr) conv2d_55_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_55_pad_before_t_in_0_ptr_const_ptr, conv2d_55_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_55_pad_before_v_pad_constant_value_const_s8), conv2d_55_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_55_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(768), (ai_i32)(1152), (ai_i32)(1152), (ai_i32)(192), (ai_i32)(192));
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(55, 1, {(stai_ptr) conv2d_55_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_55_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_55 */
  {
      const ai_i8* conv2d_55_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 16736);
    const ai_i8* conv2d_55_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 82000);
    const ai_i32* conv2d_55_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 83728);
    ai_i8* conv2d_55_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 5536);
    ai_i16* conv2d_55_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 23648);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(55, 1, {(stai_ptr) conv2d_55_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_55_t_in_0_ptr_const_s8, conv2d_55_t_in_0_shape_w_const_u16, conv2d_55_t_in_0_shape_h_const_u16, conv2d_55_t_in_0_shape_ch_const_u16, conv2d_55_t_weight_0_ptr_const_s8, conv2d_55_l_stride_1_const_u16, conv2d_55_l_stride_0_const_u16, conv2d_55_t_weight_1_ptr_const_s32, conv2d_55_t_in_0_fmt_zero_const_s8, conv2d_55_t_out_0_fmt_zero_const_s8, conv2d_55_t_in_0_fmt_scale_const_f32, conv2d_55_t_out_0_fmt_scale_const_f32, conv2d_55_t_weight_0_fmt_scale_const_f32, conv2d_55_t_out_0_ptr_s8, conv2d_55_t_out_0_shape_w_const_u16, conv2d_55_t_out_0_shape_h_const_u16, 0, 7105, conv2d_55_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(55, 1, {(stai_ptr) conv2d_55_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_55 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_56 */
  {
      const ai_i8* conv2d_56_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 5536);
    const ai_i8* conv2d_56_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 84496);
    const ai_i32* conv2d_56_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 93712);
    ai_i8* conv2d_56_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 1568);
    ai_i16* conv2d_56_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 8608);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(56, 1, {(stai_ptr) conv2d_56_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_56_t_in_0_ptr_const_s8, conv2d_56_t_in_0_shape_w_const_u16, conv2d_56_t_in_0_shape_h_const_u16, conv2d_56_l_stride_1_const_u16, conv2d_56_l_stride_0_const_u16, conv2d_56_t_in_0_shape_ch_const_u16, conv2d_56_t_weight_0_ptr_const_s8, conv2d_56_t_out_0_shape_ch_const_u16, conv2d_56_t_weight_1_ptr_const_s32, conv2d_56_t_in_0_fmt_zero_const_s8, conv2d_56_t_out_0_fmt_zero_const_s8, conv2d_56_t_in_0_fmt_scale_const_f32, conv2d_56_t_out_0_fmt_scale_const_f32, conv2d_56_t_weight_0_fmt_scale_const_f32, conv2d_56_l_out_ch_format_const_layer_format_type, conv2d_56_t_out_0_ptr_s8, 1, 1248, conv2d_56_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(56, 1, {(stai_ptr) conv2d_56_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_56 */
  /* LITE_KERNEL_SECTION BEGIN slice_58 */
  {
    
  forward_lite_slice_slice_58(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_58 */
  /* LITE_KERNEL_SECTION BEGIN concat_59 */
  {
    
  forward_lite_concat_concat_59(net_ctx);
  }
  /* LITE_KERNEL_SECTION END concat_59 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_60 */
  {
      const ai_i8* conv2d_60_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 5536);
    const ai_i8* conv2d_60_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 93904);
    const ai_i32* conv2d_60_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 107728);
    ai_i8* conv2d_60_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 16736);
    ai_i16* conv2d_60_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 6304);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(60, 1, {(stai_ptr) conv2d_60_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_60_t_in_0_ptr_const_s8, conv2d_60_t_in_0_shape_w_const_u16, conv2d_60_t_in_0_shape_h_const_u16, conv2d_60_l_stride_1_const_u16, conv2d_60_l_stride_0_const_u16, conv2d_60_t_in_0_shape_ch_const_u16, conv2d_60_t_weight_0_ptr_const_s8, conv2d_60_t_out_0_shape_ch_const_u16, conv2d_60_t_weight_1_ptr_const_s32, conv2d_60_t_in_0_fmt_zero_const_s8, conv2d_60_t_out_0_fmt_zero_const_s8, conv2d_60_t_in_0_fmt_scale_const_f32, conv2d_60_t_out_0_fmt_scale_const_f32, conv2d_60_t_weight_0_fmt_scale_const_f32, conv2d_60_l_out_ch_format_const_layer_format_type, conv2d_60_t_out_0_ptr_s8, 1, 3072, conv2d_60_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(60, 1, {(stai_ptr) conv2d_60_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_60 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_61_pad_before */
  {
      const ai_ptr conv2d_61_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 16736);
    ai_ptr conv2d_61_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 21344);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(61, 1, {(stai_ptr) conv2d_61_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_61_pad_before_t_in_0_ptr_const_ptr, conv2d_61_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_61_pad_before_v_pad_constant_value_const_s8), conv2d_61_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_61_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1152), (ai_i32)(1728), (ai_i32)(1728), (ai_i32)(288), (ai_i32)(288));
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(61, 1, {(stai_ptr) conv2d_61_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_61_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_61 */
  {
      const ai_i8* conv2d_61_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 21344);
    const ai_i8* conv2d_61_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 108880);
    const ai_i32* conv2d_61_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 111472);
    ai_i8* conv2d_61_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 6304);
    ai_i16* conv2d_61_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 31712);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(61, 1, {(stai_ptr) conv2d_61_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_61_t_in_0_ptr_const_s8, conv2d_61_t_in_0_shape_w_const_u16, conv2d_61_t_in_0_shape_h_const_u16, conv2d_61_t_in_0_shape_ch_const_u16, conv2d_61_t_weight_0_ptr_const_s8, conv2d_61_l_stride_1_const_u16, conv2d_61_l_stride_0_const_u16, conv2d_61_t_weight_1_ptr_const_s32, conv2d_61_t_in_0_fmt_zero_const_s8, conv2d_61_t_out_0_fmt_zero_const_s8, conv2d_61_t_in_0_fmt_scale_const_f32, conv2d_61_t_out_0_fmt_scale_const_f32, conv2d_61_t_weight_0_fmt_scale_const_f32, conv2d_61_t_out_0_ptr_s8, conv2d_61_t_out_0_shape_w_const_u16, conv2d_61_t_out_0_shape_h_const_u16, 0, 10657, conv2d_61_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(61, 1, {(stai_ptr) conv2d_61_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_61 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_62 */
  {
      const ai_i8* conv2d_62_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 6304);
    const ai_i8* conv2d_62_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 112624);
    const ai_i32* conv2d_62_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 126448);
    ai_i8* conv2d_62_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 3552);
    ai_i16* conv2d_62_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 10912);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(62, 1, {(stai_ptr) conv2d_62_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_62_t_in_0_ptr_const_s8, conv2d_62_t_in_0_shape_w_const_u16, conv2d_62_t_in_0_shape_h_const_u16, conv2d_62_l_stride_1_const_u16, conv2d_62_l_stride_0_const_u16, conv2d_62_t_in_0_shape_ch_const_u16, conv2d_62_t_weight_0_ptr_const_s8, conv2d_62_t_out_0_shape_ch_const_u16, conv2d_62_t_weight_1_ptr_const_s32, conv2d_62_t_in_0_fmt_zero_const_s8, conv2d_62_t_out_0_fmt_zero_const_s8, conv2d_62_t_in_0_fmt_scale_const_f32, conv2d_62_t_out_0_fmt_scale_const_f32, conv2d_62_t_weight_0_fmt_scale_const_f32, conv2d_62_l_out_ch_format_const_layer_format_type, conv2d_62_t_out_0_ptr_s8, 1, 1632, conv2d_62_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(62, 1, {(stai_ptr) conv2d_62_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_62 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_63 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_63(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_63 */
  /* LITE_KERNEL_SECTION BEGIN slice_65 */
  {
    
  forward_lite_slice_slice_65(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_65 */
  /* LITE_KERNEL_SECTION BEGIN concat_66 */
  {
    
  forward_lite_concat_concat_66(net_ctx);
  }
  /* LITE_KERNEL_SECTION END concat_66 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_67 */
  {
      const ai_i8* conv2d_67_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 5536);
    const ai_i8* conv2d_67_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 126640);
    const ai_i32* conv2d_67_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 140464);
    ai_i8* conv2d_67_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 16736);
    ai_i16* conv2d_67_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7072);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(67, 1, {(stai_ptr) conv2d_67_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_67_t_in_0_ptr_const_s8, conv2d_67_t_in_0_shape_w_const_u16, conv2d_67_t_in_0_shape_h_const_u16, conv2d_67_l_stride_1_const_u16, conv2d_67_l_stride_0_const_u16, conv2d_67_t_in_0_shape_ch_const_u16, conv2d_67_t_weight_0_ptr_const_s8, conv2d_67_t_out_0_shape_ch_const_u16, conv2d_67_t_weight_1_ptr_const_s32, conv2d_67_t_in_0_fmt_zero_const_s8, conv2d_67_t_out_0_fmt_zero_const_s8, conv2d_67_t_in_0_fmt_scale_const_f32, conv2d_67_t_out_0_fmt_scale_const_f32, conv2d_67_t_weight_0_fmt_scale_const_f32, conv2d_67_l_out_ch_format_const_layer_format_type, conv2d_67_t_out_0_ptr_s8, 1, 3072, conv2d_67_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(67, 1, {(stai_ptr) conv2d_67_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_67 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_68_pad_before */
  {
      const ai_ptr conv2d_68_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 16736);
    ai_ptr conv2d_68_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 21344);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(68, 1, {(stai_ptr) conv2d_68_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_68_pad_before_t_in_0_ptr_const_ptr, conv2d_68_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_68_pad_before_v_pad_constant_value_const_s8), conv2d_68_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_68_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1152), (ai_i32)(1728), (ai_i32)(1728), (ai_i32)(288), (ai_i32)(288));
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(68, 1, {(stai_ptr) conv2d_68_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_68_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_68 */
  {
      const ai_i8* conv2d_68_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 21344);
    const ai_i8* conv2d_68_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 141616);
    const ai_i32* conv2d_68_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 144208);
    ai_i8* conv2d_68_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7072);
    ai_i16* conv2d_68_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 31712);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(68, 1, {(stai_ptr) conv2d_68_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_68_t_in_0_ptr_const_s8, conv2d_68_t_in_0_shape_w_const_u16, conv2d_68_t_in_0_shape_h_const_u16, conv2d_68_t_in_0_shape_ch_const_u16, conv2d_68_t_weight_0_ptr_const_s8, conv2d_68_l_stride_1_const_u16, conv2d_68_l_stride_0_const_u16, conv2d_68_t_weight_1_ptr_const_s32, conv2d_68_t_in_0_fmt_zero_const_s8, conv2d_68_t_out_0_fmt_zero_const_s8, conv2d_68_t_in_0_fmt_scale_const_f32, conv2d_68_t_out_0_fmt_scale_const_f32, conv2d_68_t_weight_0_fmt_scale_const_f32, conv2d_68_t_out_0_ptr_s8, conv2d_68_t_out_0_shape_w_const_u16, conv2d_68_t_out_0_shape_h_const_u16, 0, 10657, conv2d_68_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(68, 1, {(stai_ptr) conv2d_68_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_68 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_69 */
  {
      const ai_i8* conv2d_69_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 7072);
    const ai_i8* conv2d_69_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 145360);
    const ai_i32* conv2d_69_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 159184);
    ai_i8* conv2d_69_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 3552);
    ai_i16* conv2d_69_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 16736);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(69, 1, {(stai_ptr) conv2d_69_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_69_t_in_0_ptr_const_s8, conv2d_69_t_in_0_shape_w_const_u16, conv2d_69_t_in_0_shape_h_const_u16, conv2d_69_l_stride_1_const_u16, conv2d_69_l_stride_0_const_u16, conv2d_69_t_in_0_shape_ch_const_u16, conv2d_69_t_weight_0_ptr_const_s8, conv2d_69_t_out_0_shape_ch_const_u16, conv2d_69_t_weight_1_ptr_const_s32, conv2d_69_t_in_0_fmt_zero_const_s8, conv2d_69_t_out_0_fmt_zero_const_s8, conv2d_69_t_in_0_fmt_scale_const_f32, conv2d_69_t_out_0_fmt_scale_const_f32, conv2d_69_t_weight_0_fmt_scale_const_f32, conv2d_69_l_out_ch_format_const_layer_format_type, conv2d_69_t_out_0_ptr_s8, 1, 1632, conv2d_69_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(69, 1, {(stai_ptr) conv2d_69_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_69 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_70 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_70(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_70 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_71 */
  {
      const ai_i8* conv2d_71_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 7072);
    const ai_i8* conv2d_71_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 159376);
    const ai_i32* conv2d_71_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 173200);
    ai_i8* conv2d_71_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 16736);
    ai_i16* conv2d_71_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7840);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(71, 1, {(stai_ptr) conv2d_71_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_71_t_in_0_ptr_const_s8, conv2d_71_t_in_0_shape_w_const_u16, conv2d_71_t_in_0_shape_h_const_u16, conv2d_71_l_stride_1_const_u16, conv2d_71_l_stride_0_const_u16, conv2d_71_t_in_0_shape_ch_const_u16, conv2d_71_t_weight_0_ptr_const_s8, conv2d_71_t_out_0_shape_ch_const_u16, conv2d_71_t_weight_1_ptr_const_s32, conv2d_71_t_in_0_fmt_zero_const_s8, conv2d_71_t_out_0_fmt_zero_const_s8, conv2d_71_t_in_0_fmt_scale_const_f32, conv2d_71_t_out_0_fmt_scale_const_f32, conv2d_71_t_weight_0_fmt_scale_const_f32, conv2d_71_l_out_ch_format_const_layer_format_type, conv2d_71_t_out_0_ptr_s8, 1, 3072, conv2d_71_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(71, 1, {(stai_ptr) conv2d_71_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_71 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_72_pad_before */
  {
      const ai_ptr conv2d_72_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 16736);
    ai_ptr conv2d_72_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 21344);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(72, 1, {(stai_ptr) conv2d_72_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_72_pad_before_t_in_0_ptr_const_ptr, conv2d_72_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_72_pad_before_v_pad_constant_value_const_s8), conv2d_72_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_72_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(1152), (ai_i32)(0), (ai_i32)(3456), (ai_i32)(0), (ai_i32)(576));
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(72, 1, {(stai_ptr) conv2d_72_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_72_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_72 */
  {
      const ai_i8* conv2d_72_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 21344);
    const ai_i8* conv2d_72_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 174352);
    const ai_i32* conv2d_72_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 176944);
    ai_i8* conv2d_72_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7072);
    ai_i16* conv2d_72_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 31712);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(72, 1, {(stai_ptr) conv2d_72_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_72_t_in_0_ptr_const_s8, conv2d_72_t_in_0_shape_w_const_u16, conv2d_72_t_in_0_shape_h_const_u16, conv2d_72_t_in_0_shape_ch_const_u16, conv2d_72_t_weight_0_ptr_const_s8, conv2d_72_l_stride_1_const_u16, conv2d_72_l_stride_0_const_u16, conv2d_72_t_weight_1_ptr_const_s32, conv2d_72_t_in_0_fmt_zero_const_s8, conv2d_72_t_out_0_fmt_zero_const_s8, conv2d_72_t_in_0_fmt_scale_const_f32, conv2d_72_t_out_0_fmt_scale_const_f32, conv2d_72_t_weight_0_fmt_scale_const_f32, conv2d_72_t_out_0_ptr_s8, conv2d_72_t_out_0_shape_w_const_u16, conv2d_72_t_out_0_shape_h_const_u16, 0, 10657, conv2d_72_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(72, 1, {(stai_ptr) conv2d_72_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_72 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_73 */
  {
      const ai_i8* conv2d_73_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 7072);
    const ai_i8* conv2d_73_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 178096);
    const ai_i32* conv2d_73_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 201136);
    ai_i8* conv2d_73_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 3552);
    ai_i16* conv2d_73_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 8224);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(73, 1, {(stai_ptr) conv2d_73_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_73_t_in_0_ptr_const_s8, conv2d_73_t_in_0_shape_w_const_u16, conv2d_73_t_in_0_shape_h_const_u16, conv2d_73_l_stride_1_const_u16, conv2d_73_l_stride_0_const_u16, conv2d_73_t_in_0_shape_ch_const_u16, conv2d_73_t_weight_0_ptr_const_s8, conv2d_73_t_out_0_shape_ch_const_u16, conv2d_73_t_weight_1_ptr_const_s32, conv2d_73_t_in_0_fmt_zero_const_s8, conv2d_73_t_out_0_fmt_zero_const_s8, conv2d_73_t_in_0_fmt_scale_const_f32, conv2d_73_t_out_0_fmt_scale_const_f32, conv2d_73_t_weight_0_fmt_scale_const_f32, conv2d_73_l_out_ch_format_const_layer_format_type, conv2d_73_t_out_0_ptr_s8, 1, 1952, conv2d_73_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(73, 1, {(stai_ptr) conv2d_73_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_73 */
  /* LITE_KERNEL_SECTION BEGIN slice_75 */
  {
    
  forward_lite_slice_slice_75(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_75 */
  /* LITE_KERNEL_SECTION BEGIN concat_76 */
  {
    
  forward_lite_concat_concat_76(net_ctx);
  }
  /* LITE_KERNEL_SECTION END concat_76 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_77 */
  {
      const ai_i8* conv2d_77_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 4152);
    const ai_i8* conv2d_77_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 201456);
    const ai_i32* conv2d_77_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 239856);
    ai_i8* conv2d_77_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 16736);
    ai_i16* conv2d_77_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7072);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(77, 1, {(stai_ptr) conv2d_77_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_77_t_in_0_ptr_const_s8, conv2d_77_t_in_0_shape_w_const_u16, conv2d_77_t_in_0_shape_h_const_u16, conv2d_77_l_stride_1_const_u16, conv2d_77_l_stride_0_const_u16, conv2d_77_t_in_0_shape_ch_const_u16, conv2d_77_t_weight_0_ptr_const_s8, conv2d_77_t_out_0_shape_ch_const_u16, conv2d_77_t_weight_1_ptr_const_s32, conv2d_77_t_in_0_fmt_zero_const_s8, conv2d_77_t_out_0_fmt_zero_const_s8, conv2d_77_t_in_0_fmt_scale_const_f32, conv2d_77_t_out_0_fmt_scale_const_f32, conv2d_77_t_weight_0_fmt_scale_const_f32, conv2d_77_l_out_ch_format_const_layer_format_type, conv2d_77_t_out_0_ptr_s8, 1, 5120, conv2d_77_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(77, 1, {(stai_ptr) conv2d_77_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_77 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_78_pad_before */
  {
      const ai_ptr conv2d_78_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 16736);
    ai_ptr conv2d_78_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 18656);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(78, 1, {(stai_ptr) conv2d_78_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_78_pad_before_t_in_0_ptr_const_ptr, conv2d_78_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_78_pad_before_v_pad_constant_value_const_s8), conv2d_78_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_78_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(960), (ai_i32)(1920), (ai_i32)(1920), (ai_i32)(480), (ai_i32)(480));
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(78, 1, {(stai_ptr) conv2d_78_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_78_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_78 */
  {
      const ai_i8* conv2d_78_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 18656);
    const ai_i8* conv2d_78_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 241776);
    const ai_i32* conv2d_78_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 246096);
    ai_i8* conv2d_78_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7072);
    ai_i16* conv2d_78_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 26336);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(78, 1, {(stai_ptr) conv2d_78_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_78_t_in_0_ptr_const_s8, conv2d_78_t_in_0_shape_w_const_u16, conv2d_78_t_in_0_shape_h_const_u16, conv2d_78_t_in_0_shape_ch_const_u16, conv2d_78_t_weight_0_ptr_const_s8, conv2d_78_l_stride_1_const_u16, conv2d_78_l_stride_0_const_u16, conv2d_78_t_weight_1_ptr_const_s32, conv2d_78_t_in_0_fmt_zero_const_s8, conv2d_78_t_out_0_fmt_zero_const_s8, conv2d_78_t_in_0_fmt_scale_const_f32, conv2d_78_t_out_0_fmt_scale_const_f32, conv2d_78_t_weight_0_fmt_scale_const_f32, conv2d_78_t_out_0_ptr_s8, conv2d_78_t_out_0_shape_w_const_u16, conv2d_78_t_out_0_shape_h_const_u16, 0, 17761, conv2d_78_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(78, 1, {(stai_ptr) conv2d_78_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_78 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_79 */
  {
      const ai_i8* conv2d_79_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 7072);
    const ai_i8* conv2d_79_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 248016);
    const ai_i32* conv2d_79_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 286416);
    ai_i8* conv2d_79_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 5536);
    ai_i16* conv2d_79_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 8992);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(79, 1, {(stai_ptr) conv2d_79_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_79_t_in_0_ptr_const_s8, conv2d_79_t_in_0_shape_w_const_u16, conv2d_79_t_in_0_shape_h_const_u16, conv2d_79_l_stride_1_const_u16, conv2d_79_l_stride_0_const_u16, conv2d_79_t_in_0_shape_ch_const_u16, conv2d_79_t_weight_0_ptr_const_s8, conv2d_79_t_out_0_shape_ch_const_u16, conv2d_79_t_weight_1_ptr_const_s32, conv2d_79_t_in_0_fmt_zero_const_s8, conv2d_79_t_out_0_fmt_zero_const_s8, conv2d_79_t_in_0_fmt_scale_const_f32, conv2d_79_t_out_0_fmt_scale_const_f32, conv2d_79_t_weight_0_fmt_scale_const_f32, conv2d_79_l_out_ch_format_const_layer_format_type, conv2d_79_t_out_0_ptr_s8, 1, 2720, conv2d_79_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(79, 1, {(stai_ptr) conv2d_79_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_79 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_80 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_80(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_80 */
  /* LITE_KERNEL_SECTION BEGIN slice_82 */
  {
    
  forward_lite_slice_slice_82(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_82 */
  /* LITE_KERNEL_SECTION BEGIN concat_83 */
  {
    
  forward_lite_concat_concat_83(net_ctx);
  }
  /* LITE_KERNEL_SECTION END concat_83 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_84 */
  {
      const ai_i8* conv2d_84_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 4152);
    const ai_i8* conv2d_84_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 286736);
    const ai_i32* conv2d_84_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 325136);
    ai_i8* conv2d_84_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 16736);
    ai_i16* conv2d_84_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7072);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(84, 1, {(stai_ptr) conv2d_84_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_84_t_in_0_ptr_const_s8, conv2d_84_t_in_0_shape_w_const_u16, conv2d_84_t_in_0_shape_h_const_u16, conv2d_84_l_stride_1_const_u16, conv2d_84_l_stride_0_const_u16, conv2d_84_t_in_0_shape_ch_const_u16, conv2d_84_t_weight_0_ptr_const_s8, conv2d_84_t_out_0_shape_ch_const_u16, conv2d_84_t_weight_1_ptr_const_s32, conv2d_84_t_in_0_fmt_zero_const_s8, conv2d_84_t_out_0_fmt_zero_const_s8, conv2d_84_t_in_0_fmt_scale_const_f32, conv2d_84_t_out_0_fmt_scale_const_f32, conv2d_84_t_weight_0_fmt_scale_const_f32, conv2d_84_l_out_ch_format_const_layer_format_type, conv2d_84_t_out_0_ptr_s8, 1, 5120, conv2d_84_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(84, 1, {(stai_ptr) conv2d_84_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_84 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_85_pad_before */
  {
      const ai_ptr conv2d_85_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 16736);
    ai_ptr conv2d_85_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 18656);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(85, 1, {(stai_ptr) conv2d_85_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_85_pad_before_t_in_0_ptr_const_ptr, conv2d_85_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_85_pad_before_v_pad_constant_value_const_s8), conv2d_85_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_85_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(960), (ai_i32)(1920), (ai_i32)(1920), (ai_i32)(480), (ai_i32)(480));
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(85, 1, {(stai_ptr) conv2d_85_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_85_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_85 */
  {
      const ai_i8* conv2d_85_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 18656);
    const ai_i8* conv2d_85_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 327056);
    const ai_i32* conv2d_85_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 331376);
    ai_i8* conv2d_85_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7072);
    ai_i16* conv2d_85_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 26336);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(85, 1, {(stai_ptr) conv2d_85_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_85_t_in_0_ptr_const_s8, conv2d_85_t_in_0_shape_w_const_u16, conv2d_85_t_in_0_shape_h_const_u16, conv2d_85_t_in_0_shape_ch_const_u16, conv2d_85_t_weight_0_ptr_const_s8, conv2d_85_l_stride_1_const_u16, conv2d_85_l_stride_0_const_u16, conv2d_85_t_weight_1_ptr_const_s32, conv2d_85_t_in_0_fmt_zero_const_s8, conv2d_85_t_out_0_fmt_zero_const_s8, conv2d_85_t_in_0_fmt_scale_const_f32, conv2d_85_t_out_0_fmt_scale_const_f32, conv2d_85_t_weight_0_fmt_scale_const_f32, conv2d_85_t_out_0_ptr_s8, conv2d_85_t_out_0_shape_w_const_u16, conv2d_85_t_out_0_shape_h_const_u16, 0, 17761, conv2d_85_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(85, 1, {(stai_ptr) conv2d_85_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_85 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_86 */
  {
      const ai_i8* conv2d_86_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 7072);
    const ai_i8* conv2d_86_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 333296);
    const ai_i32* conv2d_86_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 371696);
    ai_i8* conv2d_86_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 5536);
    ai_i16* conv2d_86_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 8992);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(86, 1, {(stai_ptr) conv2d_86_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_86_t_in_0_ptr_const_s8, conv2d_86_t_in_0_shape_w_const_u16, conv2d_86_t_in_0_shape_h_const_u16, conv2d_86_l_stride_1_const_u16, conv2d_86_l_stride_0_const_u16, conv2d_86_t_in_0_shape_ch_const_u16, conv2d_86_t_weight_0_ptr_const_s8, conv2d_86_t_out_0_shape_ch_const_u16, conv2d_86_t_weight_1_ptr_const_s32, conv2d_86_t_in_0_fmt_zero_const_s8, conv2d_86_t_out_0_fmt_zero_const_s8, conv2d_86_t_in_0_fmt_scale_const_f32, conv2d_86_t_out_0_fmt_scale_const_f32, conv2d_86_t_weight_0_fmt_scale_const_f32, conv2d_86_l_out_ch_format_const_layer_format_type, conv2d_86_t_out_0_ptr_s8, 1, 2720, conv2d_86_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(86, 1, {(stai_ptr) conv2d_86_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_86 */
  /* LITE_KERNEL_SECTION BEGIN eltwise_87 */
  {
    
  forward_lite_eltwise_integer_INT8_eltwise_87(net_ctx);
  }
  /* LITE_KERNEL_SECTION END eltwise_87 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_88 */
  {
      const ai_i8* conv2d_88_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 7072);
    const ai_i8* conv2d_88_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 372016);
    const ai_i32* conv2d_88_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 410416);
    ai_i8* conv2d_88_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 16736);
    ai_i16* conv2d_88_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7392);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(88, 1, {(stai_ptr) conv2d_88_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_88_t_in_0_ptr_const_s8, conv2d_88_t_in_0_shape_w_const_u16, conv2d_88_t_in_0_shape_h_const_u16, conv2d_88_l_stride_1_const_u16, conv2d_88_l_stride_0_const_u16, conv2d_88_t_in_0_shape_ch_const_u16, conv2d_88_t_weight_0_ptr_const_s8, conv2d_88_t_out_0_shape_ch_const_u16, conv2d_88_t_weight_1_ptr_const_s32, conv2d_88_t_in_0_fmt_zero_const_s8, conv2d_88_t_out_0_fmt_zero_const_s8, conv2d_88_t_in_0_fmt_scale_const_f32, conv2d_88_t_out_0_fmt_scale_const_f32, conv2d_88_t_weight_0_fmt_scale_const_f32, conv2d_88_l_out_ch_format_const_layer_format_type, conv2d_88_t_out_0_ptr_s8, 1, 5120, conv2d_88_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(88, 1, {(stai_ptr) conv2d_88_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_88 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_89_pad_before */
  {
      const ai_ptr conv2d_89_pad_before_t_in_0_ptr_const_ptr = (ai_ptr)(net_ctx->_activations[0] + 16736);
    ai_ptr conv2d_89_pad_before_t_out_0_ptr_ptr = (ai_ptr)(net_ctx->_activations[0] + 18656);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(89, 1, {(stai_ptr) conv2d_89_pad_before_t_in_0_ptr_const_ptr});
    
  forward_lite_pad_constant(conv2d_89_pad_before_t_in_0_ptr_const_ptr, conv2d_89_pad_before_t_out_0_ptr_ptr, (ai_handle)(conv2d_89_pad_before_v_pad_constant_value_const_s8), conv2d_89_pad_before_t_in_0_fmt_bitsize_const_s16, conv2d_89_pad_before_t_in_0_shape_h_const_u32, (ai_i32)(1), (ai_i32)(960), (ai_i32)(1920), (ai_i32)(1920), (ai_i32)(480), (ai_i32)(480));
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(89, 1, {(stai_ptr) conv2d_89_pad_before_t_out_0_ptr_ptr});
  }
  /* LITE_KERNEL_SECTION END conv2d_89_pad_before */
  /* LITE_KERNEL_SECTION BEGIN conv2d_89 */
  {
      const ai_i8* conv2d_89_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 18656);
    const ai_i8* conv2d_89_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 412336);
    const ai_i32* conv2d_89_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 416656);
    ai_i8* conv2d_89_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7072);
    ai_i16* conv2d_89_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 26336);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(89, 1, {(stai_ptr) conv2d_89_t_in_0_ptr_const_s8});
    
  forward_lite_dw_3x3_sssa8_ch(conv2d_89_t_in_0_ptr_const_s8, conv2d_89_t_in_0_shape_w_const_u16, conv2d_89_t_in_0_shape_h_const_u16, conv2d_89_t_in_0_shape_ch_const_u16, conv2d_89_t_weight_0_ptr_const_s8, conv2d_89_l_stride_1_const_u16, conv2d_89_l_stride_0_const_u16, conv2d_89_t_weight_1_ptr_const_s32, conv2d_89_t_in_0_fmt_zero_const_s8, conv2d_89_t_out_0_fmt_zero_const_s8, conv2d_89_t_in_0_fmt_scale_const_f32, conv2d_89_t_out_0_fmt_scale_const_f32, conv2d_89_t_weight_0_fmt_scale_const_f32, conv2d_89_t_out_0_ptr_s8, conv2d_89_t_out_0_shape_w_const_u16, conv2d_89_t_out_0_shape_h_const_u16, 0, 17761, conv2d_89_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(89, 1, {(stai_ptr) conv2d_89_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_89 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_90 */
  {
      const ai_i8* conv2d_90_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 7072);
    const ai_i8* conv2d_90_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 418576);
    const ai_i32* conv2d_90_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 495376);
    ai_i8* conv2d_90_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 3872);
    ai_i16* conv2d_90_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 8992);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(90, 1, {(stai_ptr) conv2d_90_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_90_t_in_0_ptr_const_s8, conv2d_90_t_in_0_shape_w_const_u16, conv2d_90_t_in_0_shape_h_const_u16, conv2d_90_l_stride_1_const_u16, conv2d_90_l_stride_0_const_u16, conv2d_90_t_in_0_shape_ch_const_u16, conv2d_90_t_weight_0_ptr_const_s8, conv2d_90_t_out_0_shape_ch_const_u16, conv2d_90_t_weight_1_ptr_const_s32, conv2d_90_t_in_0_fmt_zero_const_s8, conv2d_90_t_out_0_fmt_zero_const_s8, conv2d_90_t_in_0_fmt_scale_const_f32, conv2d_90_t_out_0_fmt_scale_const_f32, conv2d_90_t_weight_0_fmt_scale_const_f32, conv2d_90_l_out_ch_format_const_layer_format_type, conv2d_90_t_out_0_ptr_s8, 1, 3520, conv2d_90_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(90, 1, {(stai_ptr) conv2d_90_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_90 */
  /* LITE_KERNEL_SECTION BEGIN conv2d_91 */
  {
      const ai_i8* conv2d_91_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 3872);
    const ai_i8* conv2d_91_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 496016);
    const ai_i32* conv2d_91_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 700816);
    ai_i8* conv2d_91_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_activations[0] + 7072);
    ai_i16* conv2d_91_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 16736);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(91, 1, {(stai_ptr) conv2d_91_t_in_0_ptr_const_s8});
    
  forward_lite_pw_sssa8_ch(conv2d_91_t_in_0_ptr_const_s8, conv2d_91_t_in_0_shape_w_const_u16, conv2d_91_t_in_0_shape_h_const_u16, conv2d_91_l_stride_1_const_u16, conv2d_91_l_stride_0_const_u16, conv2d_91_t_in_0_shape_ch_const_u16, conv2d_91_t_weight_0_ptr_const_s8, conv2d_91_t_out_0_shape_ch_const_u16, conv2d_91_t_weight_1_ptr_const_s32, conv2d_91_t_in_0_fmt_zero_const_s8, conv2d_91_t_out_0_fmt_zero_const_s8, conv2d_91_t_in_0_fmt_scale_const_f32, conv2d_91_t_out_0_fmt_scale_const_f32, conv2d_91_t_weight_0_fmt_scale_const_f32, conv2d_91_l_out_ch_format_const_layer_format_type, conv2d_91_t_out_0_ptr_s8, 1, 13440, conv2d_91_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(91, 1, {(stai_ptr) conv2d_91_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END conv2d_91 */
  /* LITE_KERNEL_SECTION BEGIN pool_92 */
  {
    
  forward_lite_ap_integer_INT8_pool_92(net_ctx);
  }
  /* LITE_KERNEL_SECTION END pool_92 */
  /* LITE_KERNEL_SECTION BEGIN gemm_93 */
  {
      ai_i8* gemm_93_t_out_0_ptr_s8 = (ai_i8*)(net_ctx->_outputs[6] + 0);
    const ai_i8* gemm_93_t_in_0_ptr_const_s8 = (ai_i8*)(net_ctx->_activations[0] + 16736);
    const ai_i8* gemm_93_t_weight_0_ptr_const_s8 = (ai_i8*)(net_ctx->_weights[0] + 705936);
    const ai_i32* gemm_93_t_weight_1_ptr_const_s32 = (ai_i32*)(net_ctx->_weights[0] + 740496);
    ai_i16* gemm_93_t_scratch_0_ptr_s16 = (ai_i16*)(net_ctx->_activations[0] + 7072);
  
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB(93, 1, {(stai_ptr) gemm_93_t_in_0_ptr_const_s8});
    
  forward_lite_dense_is8os8ws8(gemm_93_t_out_0_ptr_s8, gemm_93_t_in_0_ptr_const_s8, gemm_93_t_weight_0_ptr_const_s8, gemm_93_t_weight_1_ptr_const_s32, gemm_93_t_in_0_fmt_zero_const_s8, gemm_93_t_out_0_fmt_zero_const_s8, gemm_93_t_in_0_shape_ch_const_u16, gemm_93_t_out_0_shape_ch_const_u16, gemm_93_t_out_0_shape_h_w_prod_const_u32, gemm_93_t_in_0_fmt_scale_const_f32, gemm_93_t_out_0_fmt_scale_const_f32, gemm_93_t_weight_0_fmt_scale_const_f32, gemm_93_t_scratch_0_ptr_s16);
    
  _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB(93, 1, {(stai_ptr) gemm_93_t_out_0_ptr_s8});
  }
  /* LITE_KERNEL_SECTION END gemm_93 */
  /* LITE_KERNEL_SECTION BEGIN slice_81 */
  {
    
  forward_lite_slice_slice_81(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_81 */
  /* LITE_KERNEL_SECTION BEGIN slice_74 */
  {
    
  forward_lite_slice_slice_74(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_74 */
  /* LITE_KERNEL_SECTION BEGIN slice_64 */
  {
    
  forward_lite_slice_slice_64(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_64 */
  /* LITE_KERNEL_SECTION BEGIN slice_57 */
  {
    
  forward_lite_slice_slice_57(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_57 */
  /* LITE_KERNEL_SECTION BEGIN slice_47 */
  {
    
  forward_lite_slice_slice_47(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_47 */
  /* LITE_KERNEL_SECTION BEGIN slice_40 */
  {
    
  forward_lite_slice_slice_40(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_40 */
  /* LITE_KERNEL_SECTION BEGIN slice_33 */
  {
    
  forward_lite_slice_slice_33(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_33 */
  /* LITE_KERNEL_SECTION BEGIN slice_23 */
  {
    
  forward_lite_slice_slice_23(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_23 */
  /* LITE_KERNEL_SECTION BEGIN slice_16 */
  {
    
  forward_lite_slice_slice_16(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_16 */
  /* LITE_KERNEL_SECTION BEGIN slice_6 */
  {
    
  forward_lite_slice_slice_6(net_ctx);
  }
  /* LITE_KERNEL_SECTION END slice_6 */
  return net_ctx->_return_code;
}

/*****************************************************************************/
/*  Getters APIs Section  */
STAI_API_ENTRY
stai_size stai_student_c1j_ptq_int8_get_context_size()
{
  return (stai_size)STAI_STUDENT_C1J_PTQ_INT8_CONTEXT_SIZE;
}

#if defined(HAVE_STUDENT_C1J_PTQ_INT8_INFO)
STAI_API_ENTRY
stai_return_code stai_student_c1j_ptq_int8_get_info(
  stai_network* network,
  stai_network_info* info)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, info==NULL, STAI_ERROR_NETWORK_INVALID_INFO, net_ctx->_return_code)

  // Copy of network info struct
  *info = g_student_c1j_ptq_int8_info;

  return STAI_SUCCESS;
}
#endif


STAI_API_ENTRY
stai_return_code stai_student_c1j_ptq_int8_get_activations(
  stai_network* network, stai_ptr* activations, stai_size* n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  _STAI_SET_ERROR(net_ctx, !n_activations, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_activations = STAI_STUDENT_C1J_PTQ_INT8_ACTIVATIONS_NUM;
for (stai_size idx=0; activations && (idx<STAI_STUDENT_C1J_PTQ_INT8_ACTIVATIONS_NUM); idx++) {
    // get address of the activations buffers
    activations[idx] = net_ctx->_activations[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_student_c1j_ptq_int8_get_weights(
  stai_network* network, stai_ptr* weights, stai_size* n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_weights, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_weights = STAI_STUDENT_C1J_PTQ_INT8_WEIGHTS_NUM;
for (stai_size idx=0; weights && (idx<STAI_STUDENT_C1J_PTQ_INT8_WEIGHTS_NUM); idx++) {
    // get address of the weights buffers
    weights[idx] = net_ctx->_weights[idx];
  }return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_student_c1j_ptq_int8_get_inputs(
  stai_network* network, stai_ptr* inputs, stai_size* n_inputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_inputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_inputs = STAI_STUDENT_C1J_PTQ_INT8_IN_NUM;
  for (stai_size idx=0; inputs && (idx<STAI_STUDENT_C1J_PTQ_INT8_IN_NUM); idx++) {
    inputs[idx] = net_ctx->_inputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_student_c1j_ptq_int8_get_outputs(
  stai_network* network, stai_ptr* outputs, stai_size* n_outputs)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_outputs, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  *n_outputs = STAI_STUDENT_C1J_PTQ_INT8_OUT_NUM;
  for (stai_size idx=0; outputs && (idx<STAI_STUDENT_C1J_PTQ_INT8_OUT_NUM); idx++) {
    outputs[idx] = net_ctx->_outputs[idx];
  }
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_student_c1j_ptq_int8_get_error(
  stai_network* network)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  /* return 1st generated error or STAI_SUCCESS if no errors so far */
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_student_c1j_ptq_int8_get_states(
  stai_network* network, stai_ptr* states, stai_size* n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !n_states, STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  /* get the number of internals states (supporting multi-heap also for internal states) */
  *n_states = STAI_STUDENT_C1J_PTQ_INT8_STATES_NUM;

  STAI_UNUSED(states)
return net_ctx->_return_code;
}


/*****************************************************************************/
/*  Setters APIs Section  */

STAI_API_ENTRY
stai_return_code stai_student_c1j_ptq_int8_set_activations(
  stai_network* network,
  const stai_ptr* activations,
  const stai_size n_activations)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _activations_alignment[] = STAI_STUDENT_C1J_PTQ_INT8_ACTIVATIONS_ALIGNMENTS;
  STAI_PRINT("  [stai_student_c1j_ptq_int8_set_activations] network(%p) activations[%d]: %p\n\n", net_ctx, n_activations, activations)
  _STAI_SET_ERROR(net_ctx, !activations,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_activations!=STAI_STUDENT_C1J_PTQ_INT8_ACTIVATIONS_NUM,
                  STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_NUM, net_ctx->_return_code)

  for (stai_size idx=0; activations && idx<STAI_STUDENT_C1J_PTQ_INT8_ACTIVATIONS_NUM; idx++) {
    STAI_PRINT("  activation[%d]: %p\n", idx, activations[idx])
    _STAI_SET_ERROR(net_ctx, activations[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_ACTIVATIONS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)activations[idx]) & (_activations_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_activations[idx] = activations[idx];
  }
  net_ctx->_inputs[0] = activations[0] + 56648;

  net_ctx->_inputs[1] = activations[0] + 60744;

  net_ctx->_inputs[2] = activations[0] + 56520;

  net_ctx->_inputs[3] = activations[0] + 61256;

  net_ctx->_inputs[4] = activations[0] + 56456;

  net_ctx->_inputs[5] = activations[0] + 61384;

  net_ctx->_inputs[6] = activations[0] + 56392;

  net_ctx->_inputs[7] = activations[0] + 61448;

  net_ctx->_inputs[8] = activations[0] + 56296;

  net_ctx->_inputs[9] = activations[0] + 61544;

  net_ctx->_inputs[10] = activations[0] + 56256;

  net_ctx->_outputs[0] = activations[0] + 2376;

  net_ctx->_outputs[1] = activations[0] + 1696;

  net_ctx->_outputs[2] = activations[0] + 540;

  net_ctx->_outputs[3] = activations[0] + 3552;

  net_ctx->_outputs[4] = activations[0] + 1632;

  net_ctx->_outputs[5] = activations[0] + 0;

  net_ctx->_outputs[6] = activations[0] + 512;

  net_ctx->_outputs[7] = activations[0] + 128;

  net_ctx->_outputs[8] = activations[0] + 2416;

  net_ctx->_outputs[9] = activations[0] + 2336;

  net_ctx->_outputs[10] = activations[0] + 1568;
_stai_student_c1j_ptq_int8_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_student_c1j_ptq_int8_set_weights(
  stai_network* network,
  const stai_ptr* weights,
  const stai_size n_weights)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
const uintptr_t _weights_alignment[] = STAI_STUDENT_C1J_PTQ_INT8_WEIGHTS_ALIGNMENTS;
  _STAI_SET_ERROR(net_ctx, !weights,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_weights!=STAI_STUDENT_C1J_PTQ_INT8_WEIGHTS_NUM,
                  STAI_ERROR_NETWORK_INVALID_WEIGHTS_NUM, net_ctx->_return_code)
  for (stai_size idx=0; weights && idx<STAI_STUDENT_C1J_PTQ_INT8_WEIGHTS_NUM; idx++) {
    STAI_PRINT("  weight[%d]: %p\n", idx, weights[idx])
    _STAI_SET_ERROR(net_ctx, weights[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_WEIGHTS_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)weights[idx]) & (_weights_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_weights[idx] = weights[idx];
  }_stai_student_c1j_ptq_int8_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_student_c1j_ptq_int8_set_inputs(
  stai_network* network,
  const stai_ptr* inputs,
  const stai_size n_inputs)
{
  const uintptr_t _inputs_alignment[] = STAI_STUDENT_C1J_PTQ_INT8_IN_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !inputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_inputs!=STAI_STUDENT_C1J_PTQ_INT8_IN_NUM,
                  STAI_ERROR_NETWORK_INVALID_IN_NUM, net_ctx->_return_code)

  for (stai_size idx=0; inputs && idx<STAI_STUDENT_C1J_PTQ_INT8_IN_NUM; idx++) {
    STAI_PRINT("  input[%d]: %p\n", idx, inputs[idx])
    _STAI_SET_ERROR(net_ctx, inputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_IN_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)inputs[idx]) & (_inputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_inputs[idx] = inputs[idx];
  }

  _stai_student_c1j_ptq_int8_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_student_c1j_ptq_int8_set_outputs(
  stai_network* network,
  const stai_ptr* outputs,
  const stai_size n_outputs)
{
  const uintptr_t _outputs_alignment[] = STAI_STUDENT_C1J_PTQ_INT8_OUT_ALIGNMENTS;
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  _STAI_SET_ERROR(net_ctx, !outputs,
                  STAI_ERROR_NETWORK_INVALID_API_ARGUMENTS, net_ctx->_return_code)
  _STAI_SET_ERROR(net_ctx, n_outputs!=STAI_STUDENT_C1J_PTQ_INT8_OUT_NUM,
                  STAI_ERROR_NETWORK_INVALID_OUT_NUM, net_ctx->_return_code)

  for (stai_size idx=0; outputs && idx<n_outputs; idx++) {
    STAI_PRINT("  output[%d]: %p\n", idx, outputs[idx])
    _STAI_SET_ERROR(net_ctx, outputs[idx]==NULL,
                    STAI_ERROR_NETWORK_INVALID_OUT_PTR, net_ctx->_return_code)
    _STAI_SET_ERROR(net_ctx, ((uintptr_t)outputs[idx]) & (_outputs_alignment[idx]-1),
                    STAI_ERROR_INVALID_BUFFER_ALIGNMENT, net_ctx->_return_code)
    net_ctx->_outputs[idx] = outputs[idx];
  }

  _stai_student_c1j_ptq_int8_check(net_ctx);
  return net_ctx->_return_code;
}


STAI_API_ENTRY
stai_return_code stai_student_c1j_ptq_int8_set_states(
  stai_network* network,
  const stai_ptr* states,
  const stai_size n_states)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)

  STAI_UNUSED(states)
  STAI_UNUSED(n_states)
_stai_student_c1j_ptq_int8_check(net_ctx);
  return net_ctx->_return_code;
}

STAI_API_ENTRY
stai_return_code stai_student_c1j_ptq_int8_set_callback(
  stai_network* network, const stai_event_cb cb, void* cb_cookie)
{
  _STAI_CONTEXT_ACQUIRE(net_ctx, network)
  STAI_PRINT("  set_callback %p cb %p cookie %p\n", net_ctx, cb, cb_cookie)
  // _STAI_SET_ERROR(net_ctx, cb==NULL, STAI_ERROR_NETWORK_INVALID_CALLBACK, net_ctx->_return_code)
  net_ctx->_callback = cb;
  net_ctx->_callback_cookie = cb_cookie;
  return net_ctx->_return_code;
}

#undef _STAI_SET_ERROR
#undef _STAI_CONTEXT_ALIGNMENT
#undef _STAI_CONTEXT_ACQUIRE
#undef _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_START_CB
#undef _STAI_STUDENT_C1J_PTQ_INT8_EVENT_NODE_STOP_CB
#undef _STAI_STUDENT_C1J_PTQ_INT8_MODEL_SIGNATURE
#undef _STAI_STUDENT_C1J_PTQ_INT8_DATETIME
#undef _STAI_STUDENT_C1J_PTQ_INT8_COMPILE_DATETIME

