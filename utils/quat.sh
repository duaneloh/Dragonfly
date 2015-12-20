#!/bin/bash

if [ $# -lt 1 ]
then
	echo Need one command line argument, n
	exit
fi

echo Generating quaternion file for n = $1
./utils/quatgen $1
printf -v num "%.2d" $1
mv quat_${num}_unsorted.dat aux/quat_${num}.dat
#(head -n 2 $fname && tail -n +3 $fname | sort -k1,1n -k2,2n -k3,3n -k4,4n) > aux/quat_${num}.dat
#rm $fname
