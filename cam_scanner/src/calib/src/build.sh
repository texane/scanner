#!/usr/bin/env sh

g++ \
    -Wall -O3 \
    -I/usr/include/opencv -I../src -I../.. \
    ../src/main.cc \
    ../src/cvCalibrateProCam.cc \
    ../src/cvUtilProCam.cc \
    ../../common/utils.cc \
    -lcv -lcxcore -lhighgui
