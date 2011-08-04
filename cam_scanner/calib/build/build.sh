#!/usr/bin/env sh

g++ \
    -Wall -O3 \
    -I/usr/include/opencv -I../src \
    ../src/main.cc \
    ../src/cvCalibrateProCam.cc \
    ../src/cvUtilProCam.cc \
    -lcv -lcxcore -lhighgui
