# This sets the path 
# DChandar

export MPI_ROOT=/opt/gridware/depots/54e7fb3c/el7/pkg/mpi/openmpi/4.0.0/gcc-4.8.5/
export CUDA=/opt/apps/nvidia-cuda/10.1.168/

export LD_LIBRARY_PATH=$MPI_ROOT/lib
export PATH=$MPI_ROOT/bin:$PATH
export PATH=$CUDA/bin:$PATH

echo "export CUPP=`pwd`" > .path.sh
chmod +x .path.sh
source .path.sh
export PATH=$CUPP/bin:$PATH
echo " Path has been Set "
