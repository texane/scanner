#!/usr/bin/env python

import string
import os

def read_pairs(filename):
    pairs = []
    fd = open(filename, 'r')
    for line in fd: 
        pair = line.strip().split()
        pair[0] = string.atol(pair[0], 16)
        pair[1] = string.atol(pair[1], 16)
        pairs.append(pair)
    fd.close();
    return pairs

def print_pairs(pairs):
    for pair in pairs:
        print("%u %u"%(pair[0], pair[1]))
    return

def plot_pairs(pairs):
    fd = open('/tmp/__o', 'w')
    for pair in pairs:
        fd.write("%u %u"%(pair[0], pair[1]) + '\n')
    fd.close()
    os.system("gnuplot -e \"plot('/tmp/__o') with lines; pause -1;\"")
    return

# main
pairs = read_pairs('../host_stepper/dat/1.dat')
# print_pairs(pairs)
plot_pairs(pairs)
