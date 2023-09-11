#!/bin/sh

export PATH=$PATH:/opt/openmpi/bin

export OMP_PROC_BIND=close
export OMP_PLACES=cores