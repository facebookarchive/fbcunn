#  Copyright (c) 2014, Facebook, Inc.
#  All rights reserved.
#
#  This source code is licensed under the BSD-style license found in the
#  LICENSE file in the root directory of this source tree. An additional grant
#  of patent rights can be found in the PATENTS file in the same directory.

# Some projects are installed individually as part of a larger tree, but
# include paths always reference the full include path in the tree. This
# module makes it easier to do so.
#
# Suppose you have a source tree fblualib/thrift/submodule, which is built at
# the submodule level (so you have fblualib/thrift/submodule/CMakeLists.txt)
# Files inside submodule include each other (and files from other sibling
# directories) with the full path:
#
# #include <fblualib/thrift/submodule/SomeFile.h>
# #include <fblualib/thrift/othermodule/OtherFile.h>
# #include <fblualib/thrift/Bar.h>
# #include <fblualib/meow/Foo.h>
#
# MLI_SET_DEPTH(2) at the root of your CMakeLists.txt would set "../.."
# as the include path (so fblualib is a subdirectory of that), making
# the includes work. Also, it will set MLI_INCLUDE_OUTPUT_DIR and
# MLI_INCLUDE_RELATIVE_OUTPUT_DIR to directories inside the build tree
# where any generators should output header files so they can be found
# via #include. (we recreate the lowest 2 levels of the hierarchy underneath
# ${CMAKE_BINARY_DIR})
CMAKE_MINIMUM_REQUIRED(VERSION 2.8.7 FATAL_ERROR)

FUNCTION(MLI_SET_DEPTH level)
  SET(dirs)
  SET(dir ${CMAKE_SOURCE_DIR})
  SET(relinc)
  FOREACH(i RANGE 1 ${level})
    GET_FILENAME_COMPONENT(bn ${dir} NAME)
    GET_FILENAME_COMPONENT(dir ${dir} PATH)
    LIST(APPEND dirs ${bn})
    SET(relinc "${relinc}/..")
  ENDFOREACH()
  LIST(REVERSE dirs)
  STRING(REPLACE ";" "/" relpath "${dirs}")
  SET(MLI_INCLUDE_OUTPUT_DIR
    "${CMAKE_BINARY_DIR}/${relpath}"
    PARENT_SCOPE)
  SET(MLI_INCLUDE_RELATIVE_OUTPUT_DIR
    "${relpath}"
    PARENT_SCOPE)
  INCLUDE_DIRECTORIES(
    "${CMAKE_SOURCE_DIR}/${relinc}"
    "${CMAKE_BINARY_DIR}")
ENDFUNCTION()
