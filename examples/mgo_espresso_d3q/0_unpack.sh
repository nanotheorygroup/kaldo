#!/bin/bash

# This is to unpack the forceconstants
tar -xvzf tools/mgo_qe_d3q.tgz
for file in POSCAR FORCE_CONSTANTS_3RD
do
    cp forces/${file} forces/uncharged/
done
