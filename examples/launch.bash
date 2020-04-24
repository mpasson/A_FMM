#!/bin/bash
export OMP_NUM_THREADS=1
python slow_multi_analytic.py 0
mv bands_max_0.out bands_V0.dat

python slow_multi_analytic.py 1
mv bands_max_0.out bands_V1.dat

python slow_multi_analytic.py 2
mv bands_max_0.out bands_V2.dat

python slow_multi_analytic.py 3
mv bands_max_0.out bands_V3.dat

python slow_multi_analytic.py 4
mv bands_max_0.out bands_V4.dat

python slow_multi_analytic.py 5
mv bands_max_0.out bands_V5.dat


