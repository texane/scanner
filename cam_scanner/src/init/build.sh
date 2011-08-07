#!/usr/bin/env sh

g++ \
    -Wall -O3 \
    -I/usr/include/opencv \
    main.cc \
    ../common/conf.cc \
    -lcv -lcxcore -lhighgui
