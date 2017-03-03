#!/bin/sh

for i in `seq 902 1000` `seq 1506 2000`
do
    python produce.py --seed $i --N 100 --M 10
done

