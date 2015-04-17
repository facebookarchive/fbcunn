package = "fbcunn"
version = "scm-1"

source = {
   url = "git://github.com/facebook/fbcunn.git",
}

description = {
   summary = "Facebook's extensions to torch/cunn. ",
   detailed = [[
   ]],
   homepage = "https://github.com/facebook/fbcunn",
   license = "BSD"
}

dependencies = {
   "torch >= 7.0",
   "nn >= 1.0",
   "cutorch >= 1.0",
   "multikey"
}

build = {
   type = "command",
   build_command = [[
   git submodule init
   git submodule update
cmake -E make_directory build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(LUA_BINDIR)/.." -DCMAKE_INSTALL_PREFIX="$(PREFIX)"
]],
   install_command = "cd build && $(MAKE) install"
}
