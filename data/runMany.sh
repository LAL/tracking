#!/bin/sh

for i in `seq 1439 2000`
do
    python produce.py --seed $i --N 100 --M 10
done

