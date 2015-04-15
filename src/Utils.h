#ifndef FBCUNN_UTILS_H
#define FBCUNN_UTILS_H

#include <lua.hpp>
#include "THCGeneral.h"

namespace facebook { namespace deeplearning { namespace torch {

THCState* getCutorchState(lua_State* L);

} } }

#endif
