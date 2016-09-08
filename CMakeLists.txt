CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
IF(APPLE)
    message(STATUS "FBCUNN Untested/Unsupported on OSX")
ENDIF()

SET(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_MODULE_PATH}")

FIND_PACKAGE(Torch REQUIRED)
INCLUDE(MultiLevelIncludes)
MLI_SET_DEPTH(2)
FIND_PACKAGE(Folly REQUIRED)
FIND_PACKAGE(CUDA 7.5 REQUIRED)
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_35")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# bad to hardcode like this, but i dont see any other way yet. baby steps
SET(CMAKE_INSTALL_RPATH "${Torch_INSTALL_LIB}/lua/5.1;/usr/local/lib:${CMAKE_INSTALL_RPATH}")

INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/THC")
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/src")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/cuda")

##################### Lua stuff  ########################

FILE(GLOB luasrc fbcunn/*.lua)
INSTALL(
  FILES
  ${luasrc}
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/fbcunn")

###################  C++ Stuff  #########################
################# libfbcunn
SET(src-cuda
  src/init.cu
  )
CUDA_ADD_LIBRARY(libfbcunn MODULE ${src-cuda})
TARGET_LINK_LIBRARIES(libfbcunn luaT THC TH thpp folly fbcunn_custate fbcuda_util)

### Torch packages supposes libraries prefix is "lib"
SET_TARGET_PROPERTIES(libfbcunn PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS libfbcunn
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

################# fbcunn_custate
SET(src-cuda
  src/Utils.cpp
  )
CUDA_ADD_LIBRARY(fbcunn_custate SHARED ${src-cuda})
TARGET_LINK_LIBRARIES(fbcunn_custate luaT THC TH thpp folly)

SET_TARGET_PROPERTIES(fbcunn_custate PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS fbcunn_custate
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")


################# layers_cuda
SET(src-cuda
  src/CrossMapNormalization.cu
  src/LocallyConnectedUpdateOutput.cu
  src/LocallyConnectedUpdateGradInput.cu
  src/LocallyConnectedAccGradParameters.cu
  src/LookupTableGPU.cu
  src/HSM.cu
  src/TemporalKMaxPooling.cu
  src/SparseNLLCriterion.cu
  src/WeightedLookupTable.cu
  )
CUDA_ADD_LIBRARY(layers_cuda SHARED ${src-cuda})
TARGET_LINK_LIBRARIES(layers_cuda luaT THC TH thpp folly feature_lp_pooling one_bit_quantization)

SET_TARGET_PROPERTIES(layers_cuda PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS layers_cuda
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

################# cuda_ext
SET(src-cuda
  src/InitCuda.cpp
  src/CrossMapNormalizationHost.cpp
  src/LocallyConnectedHost.cpp
  src/LookupTableGPUHost.cpp
  src/HSMHost.cpp
  src/TemporalConvolutionFBHost.cpp
  src/TemporalKMaxPoolingHost.cpp
  src/OneBitQuantizationHost.cpp
  src/SparseNLLCriterionHost.cpp
  src/FeatureLPPoolingHost.cpp
  src/CuBLASWrapperLua.cpp
  src/fft/FFTWrapperLua.cpp
  src/fft/SpatialConvolutionCuFFT.cpp
  src/fft/SpatialConvolutionCuFFTHost.cpp
  src/fft/SpatialConvolutionCuFFTTuner.cpp
  src/WeightedLookupTableHost.cpp
  )
CUDA_ADD_LIBRARY(cuda_ext MODULE ${src-cuda})
TARGET_LINK_LIBRARIES(cuda_ext luaT THC TH thpp folly libtorch_fb_fbcunn_convolution_bias cublas_wrapper cufft_convolution_cuda cufft_convolution_host fbcunn_custate layers_cuda torch_fb_fbcunn_mm fbcuda_util)

SET_TARGET_PROPERTIES(cuda_ext PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS cuda_ext
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}/fbcunn"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}/fbcunn")

################# libhalfprec
SET(src-cuda
  src/HalfPrec.cpp
  )
CUDA_ADD_LIBRARY(libhalfprec MODULE ${src-cuda})
TARGET_LINK_LIBRARIES(libhalfprec luaT THC TH thpp folly fbcunn_custate libcudahalf fbcuda_util)

SET_TARGET_PROPERTIES(libhalfprec PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS libhalfprec
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

################# libcudahalf
SET(src-cuda
  src/HalfPrecKernels.cu
  )
CUDA_ADD_LIBRARY(libcudahalf SHARED ${src-cuda})
TARGET_LINK_LIBRARIES(libcudahalf luaT THC TH thpp folly util ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

SET_TARGET_PROPERTIES(libcudahalf PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS libcudahalf
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

################# cublas_wrapper
SET(src-cuda
  src/BLASParameters.cpp
  src/CuBLASWrapper.cpp
  )
CUDA_ADD_LIBRARY(cublas_wrapper SHARED ${src-cuda})
TARGET_LINK_LIBRARIES(cublas_wrapper luaT THC TH thpp folly ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

SET_TARGET_PROPERTIES(cublas_wrapper PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS cublas_wrapper
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

################# torch_fb_fbcunn_mm
SET(src-cuda
  src/MM.cu
  )
CUDA_ADD_LIBRARY(torch_fb_fbcunn_mm SHARED ${src-cuda})
TARGET_LINK_LIBRARIES(torch_fb_fbcunn_mm luaT THC TH thpp folly ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

SET_TARGET_PROPERTIES(torch_fb_fbcunn_mm PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS torch_fb_fbcunn_mm
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

################# torch_fb_fbcunn_FFTIteratedConvolution
SET(src-cuda
  src/fft/FFTIteratedConvolution.cu
  )
CUDA_ADD_LIBRARY(torch_fb_fbcunn_FFTIteratedConvolution SHARED ${src-cuda})
TARGET_LINK_LIBRARIES(torch_fb_fbcunn_FFTIteratedConvolution luaT THC TH thpp folly ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

SET_TARGET_PROPERTIES(torch_fb_fbcunn_FFTIteratedConvolution PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS torch_fb_fbcunn_FFTIteratedConvolution
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

################# torch_fb_fbcunn_batch_norm
SET(src-cuda
  src/BatchNormalization.cu
  src/SpatialBatchNormalization.cu
  )
CUDA_ADD_LIBRARY(torch_fb_fbcunn_batch_norm SHARED ${src-cuda})
TARGET_LINK_LIBRARIES(torch_fb_fbcunn_batch_norm luaT THC TH thpp folly ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

SET_TARGET_PROPERTIES(torch_fb_fbcunn_batch_norm PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS torch_fb_fbcunn_batch_norm
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

################# one_bit_quantization
SET(src-cuda
  src/OneBitQuantization.cu
  )
CUDA_ADD_LIBRARY(one_bit_quantization SHARED ${src-cuda})
TARGET_LINK_LIBRARIES(one_bit_quantization luaT THC TH thpp folly)

SET_TARGET_PROPERTIES(one_bit_quantization PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS one_bit_quantization
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")


################# feature_lp_pooling
SET(src-cuda
  src/FeatureLPPooling.cu
  )
CUDA_ADD_LIBRARY(feature_lp_pooling SHARED ${src-cuda})
TARGET_LINK_LIBRARIES(feature_lp_pooling luaT THC TH thpp folly)

SET_TARGET_PROPERTIES(feature_lp_pooling PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS feature_lp_pooling
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

################# cuda_tensor_utils
SET(src-cuda
  src/CudaTensorUtils.cpp
  )
CUDA_ADD_LIBRARY(cuda_tensor_utils SHARED ${src-cuda})
TARGET_LINK_LIBRARIES(cuda_tensor_utils luaT THC TH thpp folly)

SET_TARGET_PROPERTIES(cuda_tensor_utils PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS cuda_tensor_utils
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

################# fbfft_wrapper
SET(src-cuda
  src/fft/FBFFTHost.cpp
  )
CUDA_ADD_LIBRARY(fbfft_wrapper SHARED ${src-cuda})
TARGET_LINK_LIBRARIES(fbfft_wrapper luaT THC TH thpp folly cuda_tensor_utils fbfft_lib fbcuda_kernel_timer)

SET_TARGET_PROPERTIES(fbfft_wrapper PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS fbfft_wrapper
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

################# fbfft_lib
SET(src-cuda
  src/fft/FBFFTDevice.cu
  )
CUDA_ADD_LIBRARY(fbfft_lib SHARED ${src-cuda})
TARGET_LINK_LIBRARIES(fbfft_lib luaT THC TH thpp folly)

SET_TARGET_PROPERTIES(fbfft_lib PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS fbfft_lib
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

################# cufft_convolution_cuda
SET(src-cuda
  src/fft/CuFFTConvolution_UpdateOutput.cu
  src/fft/CuFFTConvolution_AccGradParameters.cu
  src/fft/CuFFTConvolution_UpdateGradInput.cu
  )
CUDA_ADD_LIBRARY(cufft_convolution_cuda SHARED ${src-cuda})
TARGET_LINK_LIBRARIES(cufft_convolution_cuda luaT THC TH thpp folly libtorch_fb_fbcunn_convolution_bias cublas_wrapper cufft_wrapper cufft_convolution_host ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

SET_TARGET_PROPERTIES(cufft_convolution_cuda PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS cufft_convolution_cuda
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

################# cufft_convolution_host
SET(src-cuda
  src/fft/CuFFTConvolution.cpp
  )
CUDA_ADD_LIBRARY(cufft_convolution_host SHARED ${src-cuda})
TARGET_LINK_LIBRARIES(cufft_convolution_host luaT THC TH thpp folly cufft_wrapper cublas_wrapper fbfft_wrapper torch_fb_fbcunn_mm)

SET_TARGET_PROPERTIES(cufft_convolution_host PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS cufft_convolution_host
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

################# cufft_wrapper
SET(src-cuda
  src/fft/CuFFTWrapper.cu
  )
CUDA_ADD_LIBRARY(cufft_wrapper SHARED ${src-cuda})
TARGET_LINK_LIBRARIES(cufft_wrapper luaT THC TH thpp folly ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})

SET_TARGET_PROPERTIES(cufft_wrapper PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS cufft_wrapper
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

################# util
SET(src-cuda
  src/util/Transform.cu
  src/util/AsyncCopier.cpp
  src/util/GlobalAsyncCopier.cpp
  src/util/Misc.cpp
  )
CUDA_ADD_LIBRARY(util SHARED ${src-cuda})
TARGET_LINK_LIBRARIES(util luaT THC TH thpp folly fbcuda_util)

SET_TARGET_PROPERTIES(util PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS util
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

################# libtorch_fb_fbcunn_convolution_bias
SET(src-cuda
  src/ConvolutionBias.cu
  )
CUDA_ADD_LIBRARY(libtorch_fb_fbcunn_convolution_bias SHARED ${src-cuda})
TARGET_LINK_LIBRARIES(libtorch_fb_fbcunn_convolution_bias luaT THC TH thpp folly)

SET_TARGET_PROPERTIES(libtorch_fb_fbcunn_convolution_bias PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS libtorch_fb_fbcunn_convolution_bias
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

################# fbcuda_util
SET(src-cuda
  cuda/util/CachedDeviceProperties.cpp
  )
CUDA_ADD_LIBRARY(fbcuda_util SHARED ${src-cuda})
TARGET_LINK_LIBRARIES(fbcuda_util luaT THC TH thpp folly)

SET_TARGET_PROPERTIES(fbcuda_util PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS fbcuda_util
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

################# fbcuda_kernel_timer
SET(src-cuda
  cuda/KernelTimer.cpp
  )
CUDA_ADD_LIBRARY(fbcuda_kernel_timer SHARED ${src-cuda})
TARGET_LINK_LIBRARIES(fbcuda_kernel_timer luaT THC TH thpp folly)

SET_TARGET_PROPERTIES(fbcuda_kernel_timer PROPERTIES
  PREFIX ""
  IMPORT_PREFIX "")

INSTALL(TARGETS fbcuda_kernel_timer
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")
