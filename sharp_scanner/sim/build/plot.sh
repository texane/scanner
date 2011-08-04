set size ratio 1
set xrange[-1:1]
set yrange[-1:1]
set zrange[-1:1]
set ticslevel 0
plot('../out/disk.out') using 1:3
splot('../out/disk.out')
