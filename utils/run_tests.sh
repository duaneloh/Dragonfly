#!/bin/bash

cleanup () {
	cd ${root_dir}
	echo
	echo Removing ${test_dir}
	rm -r ${test_dir}
	exit
}

cd $(dirname "${BASH_SOURCE[0]}")/..
root_dir=$(pwd)

test_dir=$(./dragonfly_init -t testing|grep Created|awk '{print $4}')
echo Created $test_dir

cd ${test_dir}
cp ../sample_configs/testing.ini config.ini
./sim_setup.py -y --skip_data
./make_data -T -t 4

cd ${root_dir}/pyemc
python -V
python setup.py build_ext --inplace
res=$?
if [ ${res} -ne 0 ]
then
	echo
	echo Error in building Cython testing code
	cleanup
fi

echo
./unit.py -f ${test_dir}

cleanup
