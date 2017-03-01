#!/bin/sh

for i in `seq 11 200`
do
    python produce.py --seed $i --N 100 --M 10
done

