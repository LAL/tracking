#!/bin/sh                                                                       

for i in `seq 0 10`
do
    for file in particles hits
    do
	cat ${file}_100_${i}.csv >> ${file}_merged.csv
    done
done

