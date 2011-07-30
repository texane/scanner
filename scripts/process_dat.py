#!/usr/bin/env python

import sys
import string
import os

def step_to_mm(step):
    return step * 57 / 200

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

def read_pairs(filename):
    pairs = []
    fd = open(filename, 'r')
    for line in fd: 
        pair = line.strip().split()
        if len(pair) != 2: continue
        pair[0] = step_to_mm(string.atol(pair[0], 16))
        pair[1] = adc10_to_mm(string.atol(pair[1], 16))
        pairs.append(pair)
    fd.close();
    return pairs

def average_pairs(pairs):
    asize = max(pairs, key = lambda p: p[0])[0] + 1
    counts = [ 0 ] * asize
    sums = [ 0 ] * asize
    # integrate per pairs[0]
    for p in pairs:
        counts[p[0]] += 1
        sums[p[0]] += p[1]
    # reduce
    apairs = []
    for i in range(0, asize):
        aver = 0
        if counts[i] != 0:
            aver = sums[i] / counts[i]
        apairs.append((i, aver))
    return apairs

def split_passes(pairs):
    # todo
    # used to compare the different passes of a
    # multipass scan. can detect and correct phases
    # due to errors related to stepping.
    # code must be reworked, too redundant
    pairlist = []
    len_pairs = len(pairs)
    i = 0
    while i < len_pairs:
        # stride until p[i] != p[j]
        d = 0
        for j in range(i + 1, max(i + 1, len_pairs)):
            d = pairs[i][0] - pairs[j][0]
            if d != 0: break
        # find min or max value
        if d < 0: # increasing order
            for j in range(j, len_pairs):
                if pairs[j][0] < pairs[j - 1][0]: break
            if j == (len_pairs - 1): j += 1
        elif d > 0: # decreasing order
            for j in range(j, len_pairs):
                if pairs[j][0] > pairs[j - 1][0]: break
            if j == (len_pairs - 1): j += 1
        pairlist.append(pairs[i:max(j, i + 1)])
        # all equal or list too small
        if d == 0: break
        i = j
    # end while
    return pairlist

def quantize_pairs(pairs):
    # todo
    return pairs
    qpairs = []
    for p in pairs:
        d = p[1]
        if d & 1: d -= 1
        qpairs.append((p[0], d))
    return qpairs

def print_pairs(pairs):
    for pair in pairs:
        print("%u %u"%(pair[0], pair[1]))
    return

def print_pairlist(pl):
    for pairs in pl:
        print('----')
        print_pairs(pairs)
    return

def write_pairs(pairs, filename):
    fd = open(filename, 'w')
    for pair in pairs:
        fd.write("%u %u"%(pair[0], pair[1]) + '\n')
    fd.close()
    return

def write_pairlist(pl):
    for i in range(0,  len(pl)):
        oname = '/tmp/__o_' + str(i)
        write_pairs(pl[i], oname)
    return

def plot_pairs(pairs):
    write_pairs(pairs, '/tmp/__o')
    # make gnuplot command line and execute
    cmdline = 'gnuplot -e \"'
    cmdline += 'set nokey; '
    cmdline += 'set rmargin 0; '
    cmdline += 'set lmargin 0; '
    cmdline += 'set tmargin 0; '
    cmdline += 'set bmargin 0; '
    cmdline += 'set size ratio -1; ' # orthonormal
    cmdline += "plot('/tmp/__o') with lines; "
    cmdline += "replot; "
    cmdline += "pause -1; "
    cmdline += '\"'
    os.system(cmdline)
    return

# octave cmdline
# subplot(3,1,1);
# d = load('/tmp/__o_0');
# plot(d(1:size(d)(1),1), d(1:size(d)(1),2));

def plot_pairlist(pl):
    cmdline = 'gnuplot -e \"'

    cmdline += 'set size ratio -1; ' # orthonormal
    cmdline += 'set noxtic; '
    cmdline += 'set noytic; '
    cmdline += 'set nokey; '
    cmdline += 'set rmargin 0; '
    cmdline += 'set lmargin 0; '
    cmdline += 'set tmargin 0; '
    cmdline += 'set bmargin 0; '

    cmdline += 'set multiplot; '
    orig_y = 0
    for i in range(0,  len(pl)):
        oname = '/tmp/__o_' + str(i)
        write_pairs(pl[i], oname)
        if i: cmdline += 're'
        cmdline += "plot('" + oname + "') with lines; "
    cmdline += 'unset multiplot; '

    cmdline += 'replot; '
    cmdline += 'pause -1; '
    cmdline += '\"'

    os.system(cmdline)

    return

# main
filename = '../dat/0.dat'
if len(sys.argv) > 1: filename = sys.argv[1]

do_passes = 0

pairs = read_pairs(filename)

if do_passes:
    pairlist = split_passes(pairs)
    # print_pairlist(pairlist)
    plot_pairlist(pairlist)
else:
    apairs = average_pairs(pairs)
    qpairs = quantize_pairs(apairs)
    # print_pairs(apairs)
    plot_pairs(qpairs)
