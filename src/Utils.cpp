#include "Utils.h"

namespace facebook { namespace deeplearning { namespace torch {

THCState* getCutorchState(lua_State* L) {
  // Unfortunately cutorch lua headers aren't exported, so we have to
  // copy this. This is a copy from cunn.
  lua_getglobal(L, "cutorch");
  lua_getfield(L, -1, "getState");
  lua_call(L, 0, 1);
  THCState *state = (THCState*) lua_touserdata(L, -1);
  lua_pop(L, 2);
  return state;
}

} } }
