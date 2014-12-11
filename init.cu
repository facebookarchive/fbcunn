#include "luaT.h"
#include "THC.h"

#include "TemporalMaxPooling.cu"

LUA_EXTERNC DLL_EXPORT int luaopen_libfbcunn(lua_State *L);

int luaopen_libfbcunn(lua_State *L)
{
  lua_newtable(L);

  fbcunn_TemporalMaxPooling_init(L);

  return 1;
}
