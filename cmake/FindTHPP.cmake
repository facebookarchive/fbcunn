#  Copyright (c) 2014, Facebook, Inc.
#  All rights reserved.
#
#  This source code is licensed under the BSD-style license found in the
#  LICENSE file in the root directory of this source tree. An additional grant
#  of patent rights can be found in the PATENTS file in the same directory.
#
# - Try to find thpp
# This will define
# THPP_FOUND
# THPP_INCLUDE_DIR
# THPP_LIBRARIES

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.7 FATAL_ERROR)

INCLUDE(FindPackageHandleStandardArgs)

FIND_LIBRARY(THPP_LIBRARY thpp)
FIND_PATH(THPP_INCLUDE_DIR "thpp/Tensor.h")

SET(THPP_LIBRARIES ${THPP_LIBRARY})

FIND_PACKAGE_HANDLE_STANDARD_ARGS(Folly
  REQUIRED_ARGS THPP_INCLUDE_DIR THPP_LIBRARIES)
