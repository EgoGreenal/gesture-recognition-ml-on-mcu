/**
  ******************************************************************************
  * @file    student_c1j_ptq_int8.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-05-27T17:14:27+0200
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
#ifndef AI_STUDENT_C1J_PTQ_INT8_H
#define AI_STUDENT_C1J_PTQ_INT8_H

#include "student_c1j_ptq_int8_config.h"
#include "ai_platform.h"

/******************************************************************************/
#define AI_STUDENT_C1J_PTQ_INT8_MODEL_NAME          "student_c1j_ptq_int8"
#define AI_STUDENT_C1J_PTQ_INT8_ORIGIN_MODEL_NAME   "c1j_ptq_streaming_sm_int8"

/******************************************************************************/
#define AI_STUDENT_C1J_PTQ_INT8_ACTIVATIONS_ALIGNMENT   (4)
#define AI_STUDENT_C1J_PTQ_INT8_INPUTS_IN_ACTIVATIONS   (4)
#define AI_STUDENT_C1J_PTQ_INT8_OUTPUTS_IN_ACTIVATIONS  (4)

/******************************************************************************/
#define AI_STUDENT_C1J_PTQ_INT8_IN_NUM        (11)

AI_DEPRECATED
#define AI_STUDENT_C1J_PTQ_INT8_IN \
  ai_student_c1j_ptq_int8_inputs_get(AI_HANDLE_NULL, NULL)

#define AI_STUDENT_C1J_PTQ_INT8_IN_SIZE { \
  AI_STUDENT_C1J_PTQ_INT8_IN_1_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_IN_2_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_IN_3_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_IN_4_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_IN_5_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_IN_6_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_IN_7_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_IN_8_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_IN_9_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_IN_10_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_IN_11_SIZE, \
}
#define AI_STUDENT_C1J_PTQ_INT8_IN_SIZE_BYTES { \
  AI_STUDENT_C1J_PTQ_INT8_IN_1_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_IN_2_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_IN_3_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_IN_4_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_IN_5_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_IN_6_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_IN_7_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_IN_8_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_IN_9_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_IN_10_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_IN_11_SIZE_BYTES, \
}
#define AI_STUDENT_C1J_PTQ_INT8_IN_1_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_IN_1_HEIGHT      (64)
#define AI_STUDENT_C1J_PTQ_INT8_IN_1_WIDTH       (64)
#define AI_STUDENT_C1J_PTQ_INT8_IN_1_CHANNEL     (1)
#define AI_STUDENT_C1J_PTQ_INT8_IN_1_SIZE        (4096)
#define AI_STUDENT_C1J_PTQ_INT8_IN_1_SIZE_BYTES  (4096)
#define AI_STUDENT_C1J_PTQ_INT8_IN_2_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_IN_2_HEIGHT      (16)
#define AI_STUDENT_C1J_PTQ_INT8_IN_2_WIDTH       (16)
#define AI_STUDENT_C1J_PTQ_INT8_IN_2_CHANNEL     (2)
#define AI_STUDENT_C1J_PTQ_INT8_IN_2_SIZE        (512)
#define AI_STUDENT_C1J_PTQ_INT8_IN_2_SIZE_BYTES  (512)
#define AI_STUDENT_C1J_PTQ_INT8_IN_3_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_IN_3_HEIGHT      (8)
#define AI_STUDENT_C1J_PTQ_INT8_IN_3_WIDTH       (8)
#define AI_STUDENT_C1J_PTQ_INT8_IN_3_CHANNEL     (2)
#define AI_STUDENT_C1J_PTQ_INT8_IN_3_SIZE        (128)
#define AI_STUDENT_C1J_PTQ_INT8_IN_3_SIZE_BYTES  (128)
#define AI_STUDENT_C1J_PTQ_INT8_IN_4_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_IN_4_HEIGHT      (8)
#define AI_STUDENT_C1J_PTQ_INT8_IN_4_WIDTH       (8)
#define AI_STUDENT_C1J_PTQ_INT8_IN_4_CHANNEL     (2)
#define AI_STUDENT_C1J_PTQ_INT8_IN_4_SIZE        (128)
#define AI_STUDENT_C1J_PTQ_INT8_IN_4_SIZE_BYTES  (128)
#define AI_STUDENT_C1J_PTQ_INT8_IN_5_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_IN_5_HEIGHT      (4)
#define AI_STUDENT_C1J_PTQ_INT8_IN_5_WIDTH       (4)
#define AI_STUDENT_C1J_PTQ_INT8_IN_5_CHANNEL     (4)
#define AI_STUDENT_C1J_PTQ_INT8_IN_5_SIZE        (64)
#define AI_STUDENT_C1J_PTQ_INT8_IN_5_SIZE_BYTES  (64)
#define AI_STUDENT_C1J_PTQ_INT8_IN_6_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_IN_6_HEIGHT      (4)
#define AI_STUDENT_C1J_PTQ_INT8_IN_6_WIDTH       (4)
#define AI_STUDENT_C1J_PTQ_INT8_IN_6_CHANNEL     (4)
#define AI_STUDENT_C1J_PTQ_INT8_IN_6_SIZE        (64)
#define AI_STUDENT_C1J_PTQ_INT8_IN_6_SIZE_BYTES  (64)
#define AI_STUDENT_C1J_PTQ_INT8_IN_7_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_IN_7_HEIGHT      (4)
#define AI_STUDENT_C1J_PTQ_INT8_IN_7_WIDTH       (4)
#define AI_STUDENT_C1J_PTQ_INT8_IN_7_CHANNEL     (4)
#define AI_STUDENT_C1J_PTQ_INT8_IN_7_SIZE        (64)
#define AI_STUDENT_C1J_PTQ_INT8_IN_7_SIZE_BYTES  (64)
#define AI_STUDENT_C1J_PTQ_INT8_IN_8_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_IN_8_HEIGHT      (4)
#define AI_STUDENT_C1J_PTQ_INT8_IN_8_WIDTH       (4)
#define AI_STUDENT_C1J_PTQ_INT8_IN_8_CHANNEL     (6)
#define AI_STUDENT_C1J_PTQ_INT8_IN_8_SIZE        (96)
#define AI_STUDENT_C1J_PTQ_INT8_IN_8_SIZE_BYTES  (96)
#define AI_STUDENT_C1J_PTQ_INT8_IN_9_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_IN_9_HEIGHT      (4)
#define AI_STUDENT_C1J_PTQ_INT8_IN_9_WIDTH       (4)
#define AI_STUDENT_C1J_PTQ_INT8_IN_9_CHANNEL     (6)
#define AI_STUDENT_C1J_PTQ_INT8_IN_9_SIZE        (96)
#define AI_STUDENT_C1J_PTQ_INT8_IN_9_SIZE_BYTES  (96)
#define AI_STUDENT_C1J_PTQ_INT8_IN_10_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_IN_10_HEIGHT      (2)
#define AI_STUDENT_C1J_PTQ_INT8_IN_10_WIDTH       (2)
#define AI_STUDENT_C1J_PTQ_INT8_IN_10_CHANNEL     (10)
#define AI_STUDENT_C1J_PTQ_INT8_IN_10_SIZE        (40)
#define AI_STUDENT_C1J_PTQ_INT8_IN_10_SIZE_BYTES  (40)
#define AI_STUDENT_C1J_PTQ_INT8_IN_11_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_IN_11_HEIGHT      (2)
#define AI_STUDENT_C1J_PTQ_INT8_IN_11_WIDTH       (2)
#define AI_STUDENT_C1J_PTQ_INT8_IN_11_CHANNEL     (10)
#define AI_STUDENT_C1J_PTQ_INT8_IN_11_SIZE        (40)
#define AI_STUDENT_C1J_PTQ_INT8_IN_11_SIZE_BYTES  (40)

/******************************************************************************/
#define AI_STUDENT_C1J_PTQ_INT8_OUT_NUM       (11)

AI_DEPRECATED
#define AI_STUDENT_C1J_PTQ_INT8_OUT \
  ai_student_c1j_ptq_int8_outputs_get(AI_HANDLE_NULL, NULL)

#define AI_STUDENT_C1J_PTQ_INT8_OUT_SIZE { \
  AI_STUDENT_C1J_PTQ_INT8_OUT_1_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_2_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_3_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_4_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_5_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_6_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_7_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_8_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_9_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_10_SIZE, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_11_SIZE, \
}
#define AI_STUDENT_C1J_PTQ_INT8_OUT_SIZE_BYTES { \
  AI_STUDENT_C1J_PTQ_INT8_OUT_1_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_2_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_3_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_4_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_5_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_6_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_7_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_8_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_9_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_10_SIZE_BYTES, \
  AI_STUDENT_C1J_PTQ_INT8_OUT_11_SIZE_BYTES, \
}
#define AI_STUDENT_C1J_PTQ_INT8_OUT_1_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_1_HEIGHT      (2)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_1_WIDTH       (2)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_1_CHANNEL     (10)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_1_SIZE        (40)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_1_SIZE_BYTES  (40)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_2_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_2_HEIGHT      (4)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_2_WIDTH       (4)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_2_CHANNEL     (4)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_2_SIZE        (64)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_2_SIZE_BYTES  (64)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_3_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_3_HEIGHT      (16)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_3_WIDTH       (16)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_3_CHANNEL     (2)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_3_SIZE        (512)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_3_SIZE_BYTES  (512)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_4_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_4_HEIGHT      (4)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_4_WIDTH       (4)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_4_CHANNEL     (6)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_4_SIZE        (96)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_4_SIZE_BYTES  (96)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_5_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_5_HEIGHT      (4)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_5_WIDTH       (4)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_5_CHANNEL     (4)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_5_SIZE        (64)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_5_SIZE_BYTES  (64)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_6_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_6_HEIGHT      (8)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_6_WIDTH       (8)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_6_CHANNEL     (2)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_6_SIZE        (128)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_6_SIZE_BYTES  (128)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_7_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_7_CHANNEL     (27)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_7_SIZE        (27)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_7_SIZE_BYTES  (27)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_8_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_8_HEIGHT      (8)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_8_WIDTH       (8)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_8_CHANNEL     (2)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_8_SIZE        (128)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_8_SIZE_BYTES  (128)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_9_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_9_HEIGHT      (4)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_9_WIDTH       (4)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_9_CHANNEL     (6)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_9_SIZE        (96)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_9_SIZE_BYTES  (96)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_10_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_10_HEIGHT      (2)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_10_WIDTH       (2)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_10_CHANNEL     (10)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_10_SIZE        (40)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_10_SIZE_BYTES  (40)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_11_FORMAT      (AI_BUFFER_FORMAT_S8)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_11_HEIGHT      (4)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_11_WIDTH       (4)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_11_CHANNEL     (4)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_11_SIZE        (64)
#define AI_STUDENT_C1J_PTQ_INT8_OUT_11_SIZE_BYTES  (64)

/******************************************************************************/
#define AI_STUDENT_C1J_PTQ_INT8_N_NODES (111)


AI_API_DECLARE_BEGIN

/*!
 * @defgroup student_c1j_ptq_int8
 * @brief Public neural network APIs
 * @details This is the header for the network public APIs declarations
 * for interfacing a generated network model.
 * @details The public neural network APIs hide the structure of the network
 * and offer a set of interfaces to create, initialize, query, configure, 
 * run and destroy a network instance.
 * To handle this, an opaque handler to the network context is provided 
 * on creation.
 * The APIs are meant as stadard interfaces for the calling code; depending on
 * the supported platforms and the models, different implementations could be
 * available.
 */

/******************************************************************************/
/*! Public API Functions Declarations */

/*!
 * @brief Get network library info as a datastruct.
 * @ingroup student_c1j_ptq_int8
 * @param[in] network: the handler to the network context
 * @param[out] report a pointer to the report struct where to
 * store network info. See @ref ai_network_report struct for details
 * @return a boolean reporting the exit status of the API
 */
AI_DEPRECATED
AI_API_ENTRY
ai_bool ai_student_c1j_ptq_int8_get_info(
  ai_handle network, ai_network_report* report);



/*!
 * @brief Get network library report as a datastruct.
 * @ingroup student_c1j_ptq_int8
 * @param[in] network: the handler to the network context
 * @param[out] report a pointer to the report struct where to
 * store network info. See @ref ai_network_report struct for details
 * @return a boolean reporting the exit status of the API
 */
AI_API_ENTRY
ai_bool ai_student_c1j_ptq_int8_get_report(
  ai_handle network, ai_network_report* report);


/*!
 * @brief Get first network error code.
 * @ingroup student_c1j_ptq_int8
 * @details Get an error code related to the 1st error generated during
 * network processing. The error code is structure containing an 
 * error type indicating the type of error with an associated error code
 * Note: after this call the error code is internally reset to AI_ERROR_NONE
 * @param network an opaque handle to the network context
 * @return an error type/code pair indicating both the error type and code
 * see @ref ai_error for struct definition
 */
AI_API_ENTRY
ai_error ai_student_c1j_ptq_int8_get_error(ai_handle network);


/*!
 * @brief Create a neural network.
 * @ingroup student_c1j_ptq_int8
 * @details Instantiate a network and returns an object to handle it;
 * @param network an opaque handle to the network context
 * @param network_config a pointer to the network configuration info coded as a 
 * buffer
 * @return an error code reporting the status of the API on exit
 */
AI_API_ENTRY
ai_error ai_student_c1j_ptq_int8_create(
  ai_handle* network, const ai_buffer* network_config);


/*!
 * @brief Destroy a neural network and frees the allocated memory.
 * @ingroup student_c1j_ptq_int8
 * @details Destroys the network and frees its memory. The network handle is returned;
 * if the handle is not NULL, the unloading has not been successful.
 * @param network an opaque handle to the network context
 * @return an object handle : AI_HANDLE_NULL if network was destroyed
 * correctly. The same input network handle if destroy failed.
 */
AI_API_ENTRY
ai_handle ai_student_c1j_ptq_int8_destroy(ai_handle network);


/*!
 * @brief Initialize the data structures of the network.
 * @ingroup student_c1j_ptq_int8
 * @details This API initialized the network after a successfull
 * @ref ai_student_c1j_ptq_int8_create. Both the activations memory buffer 
 * and params (i.e. weights) need to be provided by caller application
 * 
 * @param network an opaque handle to the network context
 * @param params the parameters of the network (required). 
 * see @ref ai_network_params struct for details
 * @return true if the network was correctly initialized, false otherwise
 * in case of error the error type could be queried by 
 * using @ref ai_student_c1j_ptq_int8_get_error
 */
AI_API_ENTRY
ai_bool ai_student_c1j_ptq_int8_init(
  ai_handle network, const ai_network_params* params);


/*!
 * @brief Create and initialize a neural network (helper function)
 * @ingroup student_c1j_ptq_int8
 * @details Helper function to instantiate and to initialize a network. It returns an object to handle it;
 * @param network an opaque handle to the network context
 * @param activations array of addresses of the activations buffers
 * @param weights array of addresses of the weights buffers
 * @return an error code reporting the status of the API on exit
 */
AI_API_ENTRY
ai_error ai_student_c1j_ptq_int8_create_and_init(
  ai_handle* network, const ai_handle activations[], const ai_handle weights[]);


/*!
 * @brief Get network inputs array pointer as a ai_buffer array pointer.
 * @ingroup student_c1j_ptq_int8
 * @param network an opaque handle to the network context
 * @param n_buffer optional parameter to return the number of outputs
 * @return a ai_buffer pointer to the inputs arrays
 */
AI_API_ENTRY
ai_buffer* ai_student_c1j_ptq_int8_inputs_get(
  ai_handle network, ai_u16 *n_buffer);


/*!
 * @brief Get network outputs array pointer as a ai_buffer array pointer.
 * @ingroup student_c1j_ptq_int8
 * @param network an opaque handle to the network context
 * @param n_buffer optional parameter to return the number of outputs
 * @return a ai_buffer pointer to the outputs arrays
 */
AI_API_ENTRY
ai_buffer* ai_student_c1j_ptq_int8_outputs_get(
  ai_handle network, ai_u16 *n_buffer);


/*!
 * @brief Run the network and return the output
 * @ingroup student_c1j_ptq_int8
 *
 * @details Runs the network on the inputs and returns the corresponding output.
 * The size of the input and output buffers is stored in this
 * header generated by the code generation tool. See AI_STUDENT_C1J_PTQ_INT8_*
 * defines into file @ref student_c1j_ptq_int8.h for all network sizes defines
 *
 * @param network an opaque handle to the network context
 * @param[in] input buffer with the input data
 * @param[out] output buffer with the output data
 * @return the number of input batches processed (default 1) or <= 0 if it fails
 * in case of error the error type could be queried by 
 * using @ref ai_student_c1j_ptq_int8_get_error
 */
AI_API_ENTRY
ai_i32 ai_student_c1j_ptq_int8_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output);


/*!
 * @brief Runs the network on the inputs.
 * @ingroup student_c1j_ptq_int8
 *
 * @details Differently from @ref ai_network_run, no output is returned, e.g. for
 * temporal models with a fixed step size.
 *
 * @param network the network to be run
 * @param[in] input buffer with the input data
 * @return the number of input batches processed (usually 1) or <= 0 if it fails
 * in case of error the error type could be queried by 
 * using @ref ai_student_c1j_ptq_int8_get_error
 */
AI_API_ENTRY
ai_i32 ai_student_c1j_ptq_int8_forward(
  ai_handle network, const ai_buffer* input);

AI_API_DECLARE_END

#endif /* AI_STUDENT_C1J_PTQ_INT8_H */
