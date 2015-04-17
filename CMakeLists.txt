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
INCLUDE_DIRECTORIES("${CMAKE_CURRENT_SOURCE_DIR}/src")

FILE(GLOB luasrc fbcunn/*.lua)

########################################################
SET(src-cuda
  src/init.cu
  src/Utils.cpp
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
  src/CrossMapNormalization.cu
  src/FeatureLPPooling.cu
  src/HalfPrecKernels.cu
  src/HSM.cu
  src/LookupTableGPU.cu
  src/OneBitQuantization.cu
  src/SparseNLLCriterion.cu
  src/TemporalKMaxPooling.cu
  # src/TemporalMaxPooling.cu this is included directly in init.cu

  src/ConvolutionBias.cu

  src/fft/CuFFTWrapper.cu
  src/fft/FBFFTDevice.cu
  src/fft/CuFFTConvolution_UpdateOutput.cu
  src/fft/CuFFTConvolution_UpdateGradInput.cu
  src/fft/CuFFTConvolution_AccGradParameters.cu
  src/fft/CuFFTConvolution.cpp
  src/fft/SpatialConvolutionCuFFT.cpp
  src/fft/SpatialConvolutionCuFFTTuner.cpp

  src/LocallyConnectedAccGradParameters.cu
  src/LocallyConnectedUpdateGradInput.cu
  src/LocallyConnectedUpdateOutput.cu

  src/MM.cu

  src/util/Transform.cu
  src/util/Misc.cpp

  src/CudaTensorUtils.cpp
  src/CuBLASWrapper.cpp
  src/BLASParameters.cpp
  cuda/KernelTimer.cpp
  src/fft/FBFFTHost.cpp
  )

CUDA_ADD_LIBRARY(fbcunnlayers_cuda SHARED ${src-layers-cuda})
TARGET_LINK_LIBRARIES(fbcunnlayers_cuda luaT THC TH folly ${CUDA_cufft_LIBRARY})
INSTALL(TARGETS fbcunnlayers_cuda
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

SET(src-layers-cpp
  src/InitCuda.cpp
  src/CrossMapNormalizationHost.cpp
  src/LookupTableGPUHost.cpp
  src/HSMHost.cpp
  src/TemporalConvolutionFBHost.cpp
  src/TemporalKMaxPoolingHost.cpp
  src/OneBitQuantizationHost.cpp
  src/SparseNLLCriterionHost.cpp
  src/FeatureLPPoolingHost.cpp
  src/fft/SpatialConvolutionCuFFTHost.cpp
  src/fft/FFTWrapperLua.cpp
  src/CuBLASWrapperLua.cpp

  src/LocallyConnectedHost.cpp
  src/Utils.cpp
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