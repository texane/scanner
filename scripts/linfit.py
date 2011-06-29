#!/usr/bin/env python

import numpy
from matplotlib.pyplot import *

sharp_pairs = [
    # (0, 0),
    # (1350, 5),
    # (1850, 10),
    # (2225, 20),
    # (2750, 25),
    (3050, 30),
    (3000, 35),
    (2750, 40),
    (2350, 45),
    (2025, 60),
    (1775, 70),
    (1575, 80),
    (1400, 90),
    (1260, 100),
    (1050, 120),
    (920, 140),
    (810, 160),
    (730, 180),
    (650, 200),
    (520, 250),
    (420, 300),
    (380, 350),
    (300, 400)
]

x = []
y = []
for p in sharp_pairs:
    x.append(p[0])
    y.append(p[1])

coefficients = numpy.polyfit(x, y, 6)
polynomial = numpy.poly1d(coefficients)
ys = polynomial(x)
print coefficients
print polynomial

print(str(polynomial(900)))

plot(x, y, 'o')
plot(x, ys)
ylabel('y')
xlabel('x')
show()
