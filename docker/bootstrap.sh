#!/bin/bash

cd /usr/src/myapp/build

echo "Bootstraping project"
echo
echo "    => Installing system requirements"
echo
apt-get update

# This fixes the 'apt-utils' not installed error.
# Taken from https://hub.docker.com/r/1maa/debian/~/dockerfile/
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends  apt-utils

DEBIAN_FRONTEND=noninteractive apt-get -y install gfortran libsuitesparse-dev swig petsc-dev python-vtk

# Fix the python path for VTK
export PYTHONPATH=$PYTHONPATH:/usr/share/pyshared/:/usr/lib/pyshared/python2.7/vtk/

# Fix all the PETSc build errors
export PETSC_DIR=/usr/lib/petscdir/3.4.2/
export PETSC_ARCH=linux-gnu-c-opt

# Install python requirements
echo
echo "    => Installing Python requirements"
echo
pip install numpy six
pip install -r requirements.txt

echo
echo "    => Building SfePy"
echo
git clone https://github.com/sfepy/sfepy.git
cd sfepy
python setup.py build
python setup.py install
