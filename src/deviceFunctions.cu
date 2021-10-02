//! ------------------------------------------------------------------
//! CU++ET or CUPPET
//! 10/12/2010
//! Dominic Chandar
//! University of Wyoming
//! Solving Discrete equations using CUDA and Expression Templates
//! This code generates CUDA kernels automatically at compile time
//!
//! CODE   : deviceFunctions.cu
//! Purpose:
//! This section of the code has kernels for copying data between
//! the host and the device, and the main kernel to evaluate any
//! abstract expression.
//! 
//! ------------------------------------------------------------------
//!
#ifndef __DEVICEFUNC__
#define __DEVICEFUNC__
static int deviceID;
static int threadsPerBlockx, threadsPerBlocky, threadsPerBlockz;
static int blocksPerGridx, blocksPerGridy, blocksPerGridz;
static dim3 grid, block;

#ifndef __SERIAL__
__global__ void setVector(real* setMeVec, real val, int size) 
{
  // Thread ID corresponding to a local block
    int tx = threadIdx.x;
  
  // Block ID of a given grid
    int bx = blockIdx.x;
  
  // Dimensions of the block : The number of threads it contains
    int bNx = blockDim.x;
  
  // Since each row is processed by one thread, we find the global ThreadID :
   int TID = tx + bx*bNx; // All blocks are stacked horizontally

    if ( TID < size )
      {
       setMeVec[TID] = val ;
      }
}

__global__ void setVector1D(real* setMeVec, real val, int Nx, int span0)
{
    // Thread ID corresponding to a local block
    int tx = threadIdx.x;
   
    // Block ID of a given grid
    int bx = blockIdx.x;
    
    // Dimensions of the block : The number of threads it contains
    int bNx = blockDim.x;
    
    // Since each row is processed by one thread, we find the global ThreadID :
    int TID = tx + bx*bNx; // All blocks are stacked horizontally

  
    if ( TID <= span0 )
     {
       setMeVec[TID] = val;
     }

}

__global__ void setVectorUnstruc(real* setMeVec, real* indx, real val, int Nx, int span0)
{
  // Thread ID corresponding to a local block
    int tx = threadIdx.x;
   
    // Block ID of a given grid
    int bx = blockIdx.x;
    
    // Dimensions of the block : The number of threads it contains
    int bNx = blockDim.x;
    
    // Since each row is processed by one thread, we find the global ThreadID :
    int TID = tx + bx*bNx; // All blocks are stacked horizontally
    
    // Setting ThreadID
    int setTID = TID - Nx + span0+1 ;
    
    if ( setTID >= 0)
     setMeVec[(int)indx[setTID]] = val;

}
__global__ void setVector2D(real* setMeVec, real val, int Nx, int Ny, int span0, int span1 )
{
    // Thread ID corresponding to a local block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Block ID of a given grid
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // Dimensions of the block : The number of threads it contains
    int bNx = blockDim.x;
    int bNy = blockDim.y;
    // Since each row is processed by one thread, we find the global ThreadID :
    int TIDx = tx + bx*bNx; // All blocks are stacked horizontally
    int TIDy = ty + by*bNy; // All blocks are stacked horizontally
    int TID  = TIDx + Nx*TIDy;

    if ( TIDx <= span0 && TIDy <= span1  )
     {
       setMeVec[TID] = val;
     }

}

__global__ void setVector3D(real* setMeVec, real val, int Nx, int Ny, int Nz, int span0, int span1, int span2 )
{
    // Thread ID corresponding to a local block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Block ID of a given grid
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Dimensions of the block : The number of threads it contains
    int bNx = blockDim.x;
    int bNy = blockDim.y;
    
    // Since each row is processed by one thread, we find the global ThreadID :
    int TIDx       = tx + bx*bNx; 
    int TIDy       = ty + by*bNy;
    int TID        = TIDx + Nx*TIDy;
    
    int zPlane     = TIDy/Ny;
    int TIDYlocal  = TIDy - zPlane*Ny;

    if ( TIDx <= span0 && TIDYlocal <= span1 && zPlane <= span2 )
     {
       setMeVec[TID] = val;
     }

}
//! ------------------------------------------------------------------------
//! Grid Set Vector Kernels
//! ------------------------------------------------------------------------
__global__ void setVectorGrid1D(real* setMeVec, real dx, int Nx, real xmin, real xmax)
{
    // Thread ID corresponding to a local block
    int tx = threadIdx.x;
   
    // Block ID of a given grid
    int bx = blockIdx.x;
    
    // Dimensions of the block : The number of threads it contains
    int bNx = blockDim.x;
    
    // Since each row is processed by one thread, we find the global ThreadID :
    int TID = tx + bx*bNx; // All blocks are stacked horizontally

    
    if ( TID < Nx-1 )
     setMeVec[TID] = xmin + TID*dx;
     
    if ( TID == Nx-1 )
    setMeVec[TID] = xmax;
}

__global__ void setVectorGrid2D(real* setMeVecX, real* setMeVecY, real dx, real dy, int Nx, int Ny, real xmin, real xmax, real ymin, real ymax )
{
    // Thread ID corresponding to a local block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Block ID of a given grid
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // Dimensions of the block : The number of threads it contains
    int bNx = blockDim.x;
    int bNy = blockDim.y;
    // Since each row is processed by one thread, we find the global ThreadID :
    int TIDx = tx + bx*bNx; // All blocks are stacked horizontally
    int TIDy = ty + by*bNy; // All blocks are stacked horizontally
    int TID  = TIDx + Nx*TIDy;

   setMeVecX[TID] = xmin + dx*TIDx;
   setMeVecY[TID] = ymin + dy*TIDy; 
}

__global__ void setVectorGrid3D(real* setMeVecX, real* setMeVecY, real* setMeVecZ,
                                real dx, real dy, real dz,
                                int Nx, int Ny, int Nz,
                                real xmin, real xmax, real ymin, real ymax, real zmin, real zmax)
{
    // Thread ID corresponding to a local block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Block ID of a given grid
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Dimensions of the block : The number of threads it contains
    int bNx = blockDim.x;
    int bNy = blockDim.y;
    
    // Since each row is processed by one thread, we find the global ThreadID :
    int TIDx       = tx + bx*bNx; 
    int TIDy       = ty + by*bNy;
    int TID        = TIDx + Nx*TIDy;
    
    int zPlane     = TIDy/Ny;
    int TIDYlocal  = TIDy - zPlane*Ny;

    setMeVecX[TID] = xmin + dx*TIDx;
    setMeVecY[TID] = ymin + dy*TIDYlocal;
    setMeVecZ[TID] = zmin + dz*zPlane;
     
}


__global__ void vecCopy1D(real *fromA, int rNx, real* toB, int Nx, int span0 )
 {
    // Thread ID corresponding to a local block
    int tx = threadIdx.x;

    // Block ID of a given grid
    int bx = blockIdx.x;

    // Dimensions of the block : The number of threads it contains
    int bNx = blockDim.x;

    int TID1 = tx + bx*bNx; // All blocks are stacked horizontally   
 
    if ( TID1 <= span0 )
     {
       toB[TID1] = fromA[TID1];
     }

 }								    



__global__ void vecCopy2D(real *fromA, int rNx, int rNy, real* toB, int Nx, int Ny, int span0, int span1  )
 {
    // Thread ID corresponding to a local block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Block ID of a given grid
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // Dimensions of the block : The number of threads it contains
    int bNx = blockDim.x;
    int bNy = blockDim.y;
    // Since each row is processed by one thread, we find the global ThreadID :
    int TIDx = tx + bx*bNx; // All blocks are stacked horizontally
    int TIDy = ty + by*bNy; // All blocks are stacked horizontally
    int TIDR  = TIDx + rNx*TIDy;
    int TIDL  = TIDx + Nx*TIDy;

    if ( TIDx <= span0 && TIDy <= span1  )
     {
       toB[TIDL] = fromA[TIDR];
     }

 }								    
/*
__global__ void vecCopy3D(float *fromA, int rNx, int rNy, int rNz, float* toB, int Nx, int Ny, int Nz, int span0, int span1, int span2  )
 {
    // Thread ID corresponding to a local block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Block ID of a given grid
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Dimensions of the block : The number of threads it contains
    int bNx = blockDim.x;
    int bNy = blockDim.y;
    
    // Since each row is processed by one thread, we find the global ThreadID :
    int TIDx       = tx + bx*bNx; 
    int TIDy       = ty + by*bNy;
    int TID        = TIDx + Nx*TIDy;
    
    int zPlane     = TIDy/Ny;
    int TIDYlocal  = TIDy - zPlane*Ny;

    if ( TIDx <= span0 && TIDYlocal <= span1 && zPlane <= span2 )
     {
       toB[TID] = fromA[TID];
     }

 }								    
*/
__global__ void vecCopy1DReverse(real *fromA, int rNx, real* toB, int Nx, int span0, int direction )
 {
    // Thread ID corresponding to a local block
    int tx = threadIdx.x;

    // Block ID of a given grid
    int bx = blockIdx.x;

    // Dimensions of the block : The number of threads it contains
    int bNx = blockDim.x;

    int TID1   = tx + bx*bNx; // All blocks are stacked horizontally
    int TIDRHS = (span0-TID1)*(1-direction)/2  + TID1*(1+direction)/2;  

    if ( TID1 <= span0 )
     {
       toB[TID1] = fromA[TIDRHS];
     }

 }								    



__global__ void vecCopy2DReverse(real *fromA, int rNx, int rNy, real* toB, int Nx, int Ny, int span0, int span1, int directionX, int directionY  )
 {
    // Thread ID corresponding to a local block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Block ID of a given grid
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // Dimensions of the block : The number of threads it contains
    int bNx = blockDim.x;
    int bNy = blockDim.y;
    // Since each row is processed by one thread, we find the global ThreadID :
    int TIDx = tx + bx*bNx; // All blocks are stacked horizontally
    int TIDy = ty + by*bNy; // All blocks are stacked horizontally
    int TID  = TIDx + Nx*TIDy;
    
    int TIDxRHS = (span0-TIDx)*(1-directionX)/2 + TIDx*(1+directionX)/2  ;
    int TIDyRHS = (span1-TIDy)*(1-directionY)/2 + TIDy*(1+directionY)/2  ;
    int TIDRHS  = TIDxRHS + rNx*TIDyRHS;

    if ( TIDx <= span0 && TIDy <= span1  )
     {
       toB[TID] = fromA[TIDRHS];
     }

 }								    

__global__ void vecCopy3DReverse(real *fromA, int rNx, int rNy, int rNz, real* toB, int Nx, int Ny, int Nz, int span0, int span1, int span2, int directionX, int directionY, int directionZ   )
 {
    // Thread ID corresponding to a local block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Block ID of a given grid
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Dimensions of the block : The number of threads it contains
    int bNx = blockDim.x;
    int bNy = blockDim.y;
    
    // Since each row is processed by one thread, we find the global ThreadID :
    // LHS part
    int TIDx       = tx + bx*bNx; 
    int TIDy       = ty + by*bNy;
    int TID        = TIDx + Nx*TIDy;
    
    volatile int zPlane     = TIDy/Ny;
    volatile int TIDYlocal  = TIDy - zPlane*Ny;
    
    //RHS part
    int TIDxRHS    = (span0-TIDx)*(1-directionX)/2 + TIDx*(1+directionX)/2  ;
    int TIDyRHS    = (span1-TIDYlocal)*(1-directionY)/2 + TIDYlocal*(1+directionY)/2  ;
    int TIDzRHS    = (span2-zPlane)*(1-directionZ)/2 + zPlane*(1+directionZ)/2;
    int TIDRHS     = TIDxRHS + rNx*TIDyRHS + rNx*rNy*TIDzRHS;

    if ( TIDx <= span0 && TIDYlocal <= span1 &&  
                         
                             zPlane <= span2 )
     {
       toB[TID] = fromA[TIDRHS];
     }

 }								 


template < typename E>
__global__ void computeExpressionInGPU1D(E Eob,real *cTemp, int Nx, int span0 )
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int bNx= blockDim.x;
  int TID= tx + bx*bNx;
  
  
   if ( TID <= span0 )
    {
       cTemp[TID] = Eob[TID];
    }
   
}


template < typename E>
__global__ void computeExpressionInGPU2D(E Eob, real* cTemp, int Nx, int Ny, int span0, int span1 )
{
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bNx= blockDim.x;
  int bNy= blockDim.y;
  int TID1= tx + bx*bNx;
  int TID2= ty + by*bNy;
  int TID = TID1 + Nx*TID2;
   
   if ( TID1 <= span0 && TID2 <= span1  )
    {
       cTemp[TID] = Eob[TID];
     
    }
    
}

template < typename E>
__global__ void computeExpressionInGPU3D(E Eob,real *cTemp, int Nx, int Ny, int Nz, int span0, int span1, int span2 )
{
  // Thread ID corresponding to a local block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Block ID of a given grid
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Dimensions of the block : The number of threads it contains
    int bNx = blockDim.x;
    int bNy = blockDim.y;
    
    // Since each row is processed by one thread, we find the global ThreadID :
    int TIDx       = tx + bx*bNx; 
    int TIDy       = ty + by*bNy;
    int TID        = TIDx + Nx*TIDy;
    
    int zPlane     = TIDy/Ny;
    int TIDYlocal  = TIDy - zPlane*Ny;

    if ( TIDx <= span0 && TIDYlocal <= span1 && zPlane <= span2 )
     {
       cTemp[TID] = Eob[TID];
     }
   
}

__global__ void copyResult1D(real *C,real *cTemp, int Nx, int span0 )
{
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int bNx= blockDim.x;
  int TID= tx + bx*bNx;
  
   if ( TID <= span0 )
    {
       C[TID] = cTemp[TID];
    }
  
}

__global__ void copyResult2D(real *C,real *cTemp, int Nx, int Ny, int span0, int span1 )
{
  // Thread ID corresponding to a local block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Block ID of a given grid
    int bx = blockIdx.x;
    int by = blockIdx.y;
    // Dimensions of the block : The number of threads it contains
    int bNx = blockDim.x;
    int bNy = blockDim.y;
    // Since each row is processed by one thread, we find the global ThreadID :
    int TIDx = tx + bx*bNx; // All blocks are stacked horizontally
    int TIDy = ty + by*bNy; // All blocks are stacked horizontally
    int TID  = TIDx + Nx*TIDy;

    if ( TIDx <= span0 && TIDy <= span1  )
     {
       C[TID] = cTemp[TID];
     }
  
}

__global__ void copyResult3D(real *C,real *cTemp, int Nx, int Ny, int Nz, int span0, int span1, int span2 )
{
   // Thread ID corresponding to a local block
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Block ID of a given grid
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Dimensions of the block : The number of threads it contains
    int bNx = blockDim.x;
    int bNy = blockDim.y;
    
    // Since each row is processed by one thread, we find the global ThreadID :
    int TIDx       = tx + bx*bNx; 
    int TIDy       = ty + by*bNy;
    int TID        = TIDx + Nx*TIDy;
    
    int zPlane     = TIDy/Ny;
    int TIDYlocal  = TIDy - zPlane*Ny;

    if ( TIDx <= span0 && TIDYlocal <= span1 && zPlane <= span2 )
     {
       C[TID] = cTemp[TID];
     }
  
}
#endif

#endif


