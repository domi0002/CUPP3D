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

#ifdef __SERIAL__
void setVector(real* setMeVec, real val, int size) 
{

    for ( int TID = 0 ; TID < size ; TID++ )
      {
       setMeVec[TID] = val ;
      }

}


void setVector1D(real* setMeVec, real val, int Nx, int span0)
{
    for ( int TID  = 0 ; TID <= span0 ; TID++ )
     {
       setMeVec[TID] = val;
     }

}


void setVector2D(real* setMeVec, real val, int Nx, int Ny, int span0, int span1 )
{

    for ( int TIDy = 0 ; TIDy <= span1 ; TIDy++ )
     {
      for ( int TIDx = 0 ; TIDx <= span0 ; TIDx++ )
       {
          setMeVec[TIDx+Nx*TIDy]=val;
       }
     }


}

void setVector3D(real* setMeVec, real val, int Nx, int Ny, int Nz, int span0, int span1, int span2 )
{

    for ( int TIDz = 0 ; TIDz <= span2 ; TIDz++ )
     {
       for ( int TIDy = 0 ; TIDy <= span1 ; TIDy++ )
        {
          for ( int TIDx = 0 ; TIDx <= span0 ; TIDx++ )
           {
             setMeVec[TIDx + Nx*TIDy + Nx*Ny*TIDz] = val;
           }
        }
   }

}
//! ------------------------------------------------------------------------
//! Grid Set Vector Kernels
//! ------------------------------------------------------------------------
void setVectorGrid1D(real* setMeVec, real dx, int Nx, real xmin, real xmax)
{
    
    for  ( int TID = 0 ; TID < Nx-1 ; TID++)
        setMeVec[TID] = xmin + TID*dx;
    
    int TID = Nx-1;
    setMeVec[TID] = xmax;
}

void setVectorGrid2D(real* setMeVecX, real* setMeVecY, real dx, real dy, int Nx, int Ny, real xmin, real xmax, real ymin, real ymax )
{
   for ( int TIDy = 0 ; TIDy < Ny ; TIDy++ )
    {
      for ( int TIDx = 0 ; TIDx < Nx ; TIDx++ )
       {
         setMeVecX[TIDx+Nx*TIDy] = xmin + dx*TIDx;
         setMeVecY[TIDx+Nx*TIDy] = ymin + dy*TIDy; 
       }
   }
}

void setVectorGrid3D(real* setMeVecX, real* setMeVecY, real* setMeVecZ,
                                real dx, real dy, real dz,
                                int Nx, int Ny, int Nz,
                                real xmin, real xmax, real ymin, real ymax, real zmin, real zmax)
{

   for ( int TIDz = 0 ; TIDz < Nz ; TIDz++ )
    {
     for ( int TIDy = 0 ; TIDy < Ny ; TIDy++ )
      {
       for ( int TIDx = 0; TIDx < Nx ; TIDx++ )
        {
          int TID = TIDx + Nx*TIDy + Nx*Ny*TIDz;
          setMeVecX[TID] = xmin + dx*TIDx;
          setMeVecY[TID] = ymin + dy*TIDy;
          setMeVecZ[TID] = zmin + dz*TIDz;
        }
     }
   }
     
}


void vecCopy1D(real *fromA, int rNx, real* toB, int Nx, int span0 )
 {

    for ( int TID1 = 0 ; TID1  <= span0 ; TID1++ )
     {
       toB[TID1] = fromA[TID1];
     }

 }								    



void vecCopy2D(real *fromA, int rNx, int rNy, real* toB, int Nx, int Ny, int span0, int span1  )
 {

  for ( int y = 0 ; y <= span1 ; y++ )
   {
    for ( int x = 0 ; x  <= span0 ; x++ )
     {
       int i = x + Nx*y;
       toB[i] = fromA[i];
     }
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
void vecCopy1DReverse(real *fromA, int rNx, real* toB, int Nx, int span0, int direction )
 {

    for ( int x = 0 ; x <= span0 ; x++ )
     {
       int xr = (span0-x)*(1-direction)/2 + x*(1+direction)/2;
       toB[x] = fromA[xr];
     }

 }								    



void vecCopy2DReverse(real *fromA, int rNx, int rNy, real* toB, int Nx, int Ny, int span0, int span1, int directionX, int directionY  )
 {

    for ( int y = 0 ; y <= span1 ; y++ )
     {
      for ( int x = 0 ; x <= span0 ; x++ )
       {
           int iLeft = x + Nx*y;
           int xr    = (span0-x)*(1-directionX)/2 + x*(1+directionX)/2;
           int yr    = (span1-y)*(1-directionY)/2 + y*(1+directionY)/2;
           int iRite = xr + rNx*yr;
           toB[iLeft] = fromA[iRite];
       }
    }


 }								    

void vecCopy3DReverse(real *fromA, int rNx, int rNy, int rNz, real* toB, int Nx, int Ny, int Nz, int span0, int span1, int span2, int directionX, int directionY, int directionZ   )
 {

  for ( int z = 0 ; z <= span2 ; z++ )
   {
    for ( int y = 0 ; y <= span1 ; y++ )
     {
      for ( int x = 0 ; x <= span0 ; x++ )
       {
           int iLeft = x + Nx*y + Nx*Ny*z;
           int xr    = (span0-x)*(1-directionX)/2 + x*(1+directionX)/2;
           int yr    = (span1-y)*(1-directionY)/2 + y*(1+directionY)/2;
           int zr    = (span1-z)*(1-directionZ)/2 + z*(1+directionZ)/2;
           int iRite = xr + rNx*yr + rNx*rNy*zr;
           toB[iLeft] = fromA[iRite];
       }
    }
  }

}								 


template < typename E>
void computeExpressionInGPU1D(E Eob,real *cTemp, int Nx, int span0 )
{
  
  for ( int i = 0 ; i <= span0 ; i++ ) 
   {
       cTemp[i] = Eob[i];
    }
   
}


template < typename E>
void computeExpressionInGPU2D(E Eob, real* cTemp, int Nx, int Ny, int span0, int span1 )
{

  for ( int y = 0 ; y <= span1 ; y++ )
  {
   for ( int x = 0 ; x <= span0 ; x++ )
    {
      int i = x + Nx*y;
      cTemp[i] = Eob[i];
    }
  }

   
    
}

template < typename E>
void computeExpressionInGPU3D(E Eob,real *cTemp, int Nx, int Ny, int Nz, int span0, int span1, int span2 )
{
  
 for ( int z = 0 ; z <= span2 ; z++ )
  {
   for ( int y = 0 ; y <= span1 ; y++ )
   {
    for ( int x = 0 ; x <= span0 ; x++ )
     {
       int i = x + Nx*y + Nx*Ny*z;
       cTemp[i] = Eob[i];
     }
   }
 }
 
}

void copyResult1D(real *C,real *cTemp, int Nx, int span0 )
{
  
   for ( int i = 0  ; i <= span0 ; i++ )
    {
       C[i] = cTemp[i];
    }
  
}

void copyResult2D(real *C,real *cTemp, int Nx, int Ny, int span0, int span1 )
{

    for ( int y = 0 ; y <= span1 ; y++ )
     {
       for ( int x = 0 ; x <= span0 ; x++ )
        {
          int i = x + Nx*y;
          C[i] = cTemp[i];
         }
     }
  
}

void copyResult3D(real *C,real *cTemp, int Nx, int Ny, int Nz, int span0, int span1, int span2 )
{

   for ( int z = 0 ; z <= span2 ; z++ )
    {
     for ( int y = 0 ; y <= span1 ; y++ )
     {
       for ( int x = 0 ; x <= span0 ; x++ )
        {
          int i = x + Nx*y + Nx*Ny*z;
          C[i] = cTemp[i];
         }
     }
    }
 
}
#endif

#endif



