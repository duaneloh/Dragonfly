#!/bin/bash

cd $(dirname "${BASH_SOURCE[0]}")/..
root_dir=$(pwd)

dir=$(./init_new_recon.py -t testing|grep Created|awk '{print $4}')

cd ${dir}
cp ../sample_configs/testing.ini config.ini
./sim_setup.py -y --skip_data
./make_data -T -t 4

cd ${root_dir}/pyemc
python setup.py build_ext --inplace
res=$?
if [ ${res} -ne 0 ]
then
	echo Error in building output
	exit ${res}
fi

echo
./unit.py -f ${dir}
cd ${root_dir}

rm -r ${dir}
