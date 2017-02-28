#!/bin/sh

for i in `seq 0 10`
do
    python produce.py --seed i --N 100 --M 10
done

