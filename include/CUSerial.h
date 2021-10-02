//! ----------------------------------------------
//! CPU Code Emulator Functions for CUDA
//! Dominic Chandar
//! 03/23/2011
//! ----------------------------------------------

#ifdef __SERIAL__

#define __powf(a,b) pow(a,b)
#define __expf(a) exp(a)


void cudaMalloc(void** ptr, size_t size)
{
   *ptr = (void*)malloc(size);
}

void cudaFree(void* ptr)
{
  if ( ptr != NULL )
   free(ptr);
}

#define __global__
#define __device__
#define __shared__
#define cudaMemcpyHostToDevice 0
#define cudaMemcpyDeviceToHost 1
#define cudaMemcpyDeviceToDevice 2

void cudaMemcpy(real* dst, real* src, size_t count, int kind)
{
  int N = count/sizeof(real); 
      for ( int i = 0 ; i < N ; i++)
         dst[i]=src[i];

}

void cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t pitch, size_t width, size_t height, int kind)
{
// Do nothing
}

void cudaSetDevice(int deviceID)
{
// Do Nothing.. no device to be set
}

void cudaThreadExit()
{
// No device to Exit..
}

void cudaThreadSynchronize()
{
// Nothing to Synchronize...
}
class dim3
{
  public:
  int x, y, z;
};
dim3 junkObject;

#define threadIdx junkObject
#define blockIdx  junkObject
#define blockDim  junkObject
#define gridDim   junkObject

double atomicAdd(double* address, double val)
{
  *address += val;
 return *address;
}

float atomicAdd(float* address, float val)
{
  *address += val;
  return *address;
}

int atomicAdd(int* address, int val)
{
  *address += val;
  return *address;
}
#endif

