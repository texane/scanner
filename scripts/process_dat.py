#!/usr/bin/env python

import string

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

# main
pairs = read_pairs('../host_stepper/dat/1.dat')
print_pairs(pairs)
