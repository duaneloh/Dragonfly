
#!/usr/bin/env python
import numpy as np
import os
import subprocess
from subprocess
import argparse
import sys
from py_src import py_utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Starts EMC reconstruction")
    parser.add_argument("-Q", "--make_quat_only", dest="make_quat_only", action="store_true", default=False)
    # Option for non-mpi execution
    # Default num_mpi and num_threads
    # Setup default system configurations
    args = parser.parse_args()

