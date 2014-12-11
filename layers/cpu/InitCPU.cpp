/**
 * Copyright 2014 Facebook
 * @author Tudor Bosman (tudorb@fb.com)
 */

#include <lua.hpp>

namespace facebook { namespace deeplearning { namespace torch {

void initCrossMapNormalization(lua_State* L);
void initLocallyConnected(lua_State* L);
void initKMaxPooling(lua_State* L);
void initGroupKMaxPooling(lua_State* L);
void initHSM(lua_State* L);
void initSparseNLLCriterion(lua_State* L);

}}}  // namespace

using namespace facebook::deeplearning::torch;

extern "C" int LUAOPEN(lua_State* L) {
  initCrossMapNormalization(L);
  initLocallyConnected(L);
  initKMaxPooling(L);
  initGroupKMaxPooling(L);
  initHSM(L);
  initSparseNLLCriterion(L);
  return 0;
}
