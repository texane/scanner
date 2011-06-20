#!/usr/bin/env sh

g++ \
-Wall \
-DdSINGLE=1 \
-I$HOME/install/include \
../src/main.cc \
-L$HOME/install/lib -ldrawstuff -lode -lGL -lGLU -lm -lX11
