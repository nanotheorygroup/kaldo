#!/bin/bash

# This is to unpack the forceconstants
tar -xvzf tools/mgo-qe-d3q.tgz
for file in POSCAR FORCE_CONSTANTS_3RD
do
    cp forces/${file} forces/uncharged/
done
