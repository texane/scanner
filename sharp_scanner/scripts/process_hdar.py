#!/usr/bin/env python

# read a list of ahdr tuples and convert them into xyz
# if the radius (see below) component is missing, a fixed
# value is assumed.
#
# an ahdr tuple consists of
# hdar[0] the angle, in radians
# hdar[1] the height, in mm
# hdar[2] the depth, in mm
# hdar[3] the radius, in mm
# note the radius is constant and is the distance,
# in mm, between the measuring sensor and the center
# of the rotating plate.
#


import sys
import string
import os
import math


# fixed radius, if missing
CONFIG_FIXED_RADIUS = '%x'%100

# fixed per step values
def to_radians(d): return (2.0 * math.pi / 360.0) * d
CONFIG_RAD_PER_STEP = to_radians(15.0)
CONFIG_MM_PER_STEP = 57.0 / 200.0


def hdar_to_xyz(hdar):
    # convert from hdar to xyz
    xyz = [ 0, 0, 0 ]
    d = hdar[3] - hdar[1]
    xyz[0] = d * math.cos(hdar[2])
    xyz[1] = hdar[0]
    xyz[2] = d * math.sin(hdar[2])
    return xyz


# global (voltage, distance) pairs
sharp_pairs = [
    ( 2860, 30 ),
    ( 2770, 40 ),
    ( 2340, 50 ),
    ( 1950, 60 ),
    ( 1680, 70 ),
    ( 1470, 80 ),
    ( 1320, 90 ),
    ( 1170, 100 ),
    ( 980, 120 ),
    ( 810, 140 ),
    ( 710, 160 ),
    ( 600, 180 ),
    ( 520, 200 ),
    ( 460, 220 ),
    ( 410, 250 ),
    ( 310, 300 )
]

def adc10_to_mm(adc):
    # convert adc to millivolts
    # vref is set to 3.5v == 3500mv
    # adc is 10 bits (1024)
    # sharp model is gp2y0a41sk0f
    # assume that distance is >= 30mm (see sharp datasheet)
    # mv = (3500 / 1024) * adc
    # return (10000 / mv) - 4.2
    mv = (3500 / 1024) * adc
    for pos in range(0, len(sharp_pairs)):
        if (mv > sharp_pairs[pos][0]): break
    if pos == 0: return 30
    elif pos == len(sharp_pairs): return 400
    vd = sharp_pairs[pos - 1][0] - sharp_pairs[pos][0];
    dd = sharp_pairs[pos][1] - sharp_pairs[pos - 1][1];
    return sharp_pairs[pos - 1][1] + ((sharp_pairs[pos - 1][0] - mv) * dd) / vd;

def step_to_radius(step):
    return step * CONFIG_RAD_PER_STEP

def step_to_height(step):
    return step * CONFIG_MM_PER_STEP

def read_hdar_tuples(filename):
    tuples = []
    fd = open(filename, 'r')
    for line in fd: 
        toks = line.strip().split()
        tuple = [ 0, 0, 0, 0, 0, 0, 0 ]
        tuple[4] = string.atol(toks[0], 16)
        tuple[5] = string.atol(toks[1], 16)
        tuple[6] = string.atol(toks[2], 16)
        tuple[0] = step_to_height(string.atol(toks[0], 16))
        tuple[1] = adc10_to_mm(string.atol(toks[1], 16))
        # skip, assume no contact point
        if tuple[1] >= 400: continue
        tuple[2] = step_to_radius(string.atol(toks[2], 16))
        r = CONFIG_FIXED_RADIUS
        if len(toks) == 4: r = toks[3]
        tuple[3] = string.atol(r, 16)
        tuples.append(tuple)
    fd.close();
    return tuples

def convert_hdar_tuples(hdar_tuples):
    xyz_tuples = []
    for t in hdar_tuples:
        xyz_tuples.append(hdar_to_xyz(t))
        bar = xyz_tuples[-1]
        print('%d %d %d | %d %d %d | %f %f %f'%(t[4], t[5], t[6], t[0],t[1],t[2],bar[0],bar[1],bar[2]))
    return xyz_tuples

def print_xyz_tuples(tuples):
    for t in tuples:
        print("%lf %lf %lf"%(t[0], t[1], t[2]))
    return

def write_xyz_tuples(tuples):
    fd = open('/tmp/__xyz', 'w')
    i = 0
    for t in tuples:
        fd.write("%lf %lf %lf"%(t[0], t[1], t[2]) + '\n')
    fd.close()
    return

def plot_xyz_tuples(tuples):
    write_xyz_tuples(tuples)
    cmdline = 'gnuplot -e \"'
    cmdline += 'set nokey; '
    cmdline += 'set xrange [-300:300]; '
    cmdline += 'set yrange [-300:300]; '
    cmdline += 'set zrange [-300:300]; '
    cmdline += 'set ticslevel 0; '
    cmdline += 'set size ratio 1; '
    cmdline += "splot('/tmp/__xyz'); "
    cmdline += "replot; "
    cmdline += "pause -1; "
    cmdline += '\"'
    os.system(cmdline)
    return

# main
hdar_filename = '../dat/full.dat'
if len(sys.argv) > 1: hdar_filename = sys.argv[1]
hdar_tuples = read_hdar_tuples(hdar_filename)
xyz_tuples = convert_hdar_tuples(hdar_tuples)
plot_xyz_tuples(xyz_tuples)
