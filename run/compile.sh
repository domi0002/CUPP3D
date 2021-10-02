#!/bin/bash

#Note the architecture 'SM'. 

# TESLA V-100
nvcc -arch=sm_70 -DCUPP_DOUBLE -I$CUPP/include -I$CUPP/src eu3d.cu -o eu3d

# QUADRO K4200
#nvcc -arch=sm_30 -DCUPP_DOUBLE -I$CUPP/include -I$CUPP/src eu3d.cu -o eu3d

