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
FIND_PACKAGE(CUDA 6.5 REQUIRED)
LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_35")

#SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
#SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
#SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# bad to hardcode like this, but i dont see any other way yet. baby steps
SET(CMAKE_INSTALL_RPATH "${Torch_INSTALL_LIB}/lua/5.1;/usr/local/lib:${CMAKE_INSTALL_RPATH}") 

INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/THC")
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")

INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}")
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/src/cuda")

FILE(GLOB luasrc luasrc/*.lua)

########################################################
SET(src-cuda
  src/cuda/init.cu
  )
CUDA_ADD_LIBRARY(fbcunn MODULE ${src-cuda})
TARGET_LINK_LIBRARIES(fbcunn luaT THC TH  fbcunnlayers_cuda)

### Torch packages supposes libraries prefix is "lib"
SET_TARGET_PROPERTIES(fbcunn PROPERTIES
  PREFIX "lib"
  IMPORT_PREFIX "lib")

INSTALL(TARGETS fbcunn
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

#######################
SET(src-layers-cuda
  src/cuda/CrossMapNormalization.cu
  src/cuda/FeatureLPPooling.cu
  src/cuda/HalfPrecKernels.cu
  src/cuda/HSM.cu
  src/cuda/LookupTableGPU.cu
  src/cuda/OneBitQuantization.cu
  src/cuda/SparseNLLCriterion.cu
  src/cuda/TemporalKMaxPooling.cu
  src/cuda/bias/ConvolutionBias.cu
  src/cuda/util/Transform.cu
  src/cuda/util/Misc.cpp
  src/cuda/fft/CuFFTWrapper.cu
  src/cuda/fft/FBFFT.cu
  src/cuda/fft/CuFFTConvolution_AccGradParameters.cu
  src/cuda/fft/CuFFTConvolution.cpp
  src/cuda/fft/CuFFTConvolution_UpdateOutput.cu
  src/cuda/fft/CuFFTConvolution_UpdateGradInput.cu
  src/cuda/fft/CuFFTConvolution_AccGradParameters.cu
  src/cuda/fft/SpatialConvolutionCuFFT.cpp
  src/cuda/fft/SpatialConvolutionCuFFTTuner.cpp
  src/cuda/CudaTensorUtils.cpp
  src/cuda/CuBLASWrapper.cpp
  src/cuda/BLASParameters.cpp
  cuda/KernelTimer.cpp
  )

CUDA_ADD_LIBRARY(fbcunnlayers_cuda SHARED ${src-layers-cuda})
TARGET_LINK_LIBRARIES(fbcunnlayers_cuda luaT THC TH folly ${CUDA_cufft_LIBRARY})
INSTALL(TARGETS fbcunnlayers_cuda
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

SET(src-layers-cpp
  src/cuda/InitCuda.cpp
  src/cuda/CrossMapNormalizationHost.cpp
  src/cuda/LookupTableGPUHost.cpp
  src/cuda/HSMHost.cpp
  src/cuda/TemporalConvolutionFBHost.cpp
  src/cuda/TemporalKMaxPoolingHost.cpp
  src/cuda/OneBitQuantizationHost.cpp
  src/cuda/SparseNLLCriterionHost.cpp
  src/cuda/FeatureLPPoolingHost.cpp
  src/cuda/fft/SpatialConvolutionCuFFTHost.cpp
  src/cuda/fft/FFTWrapperLua.cpp
  src/cuda/CuBLASWrapperLua.cpp
)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ADD_LIBRARY(fbcunnlayers MODULE ${src-layers-cpp})
TARGET_LINK_LIBRARIES(fbcunnlayers fbcunnlayers_cuda luaT THC TH folly)

INSTALL(TARGETS fbcunnlayers
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

INSTALL(
  FILES
  ${luasrc}
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/fbcunn")