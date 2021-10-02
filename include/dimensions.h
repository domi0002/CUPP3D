//! ------------------------------------------------------------------
//! CU++ET or CUPPET
//! 10/12/2010
//! Dominic Chandar
//! University of Wyoming 2010
//! Queen's University Belfast 2021
//! Solving Discrete equations using CUDA and Expression Templates
//! This code generates CUDA kernels automatically at compile time
//!
//! CODE    : dimensions.h
//! Purpose :
//! This section of the code calls functions for different spatial 
//! dimensions.
//!  
//! ------------------------------------------------------------------
//!
class dimensionalSpace
//! ------------------------------------------------------------------------------
//! This is the base class that has functions that behaves in a different manner
//! for varying spatial dimension. If a One-Dimensional array is declared, then
//! an instance of class oneDimension is created, and the corresponding functions
//! are called. This procedure avoids the use of if/then/else loops to choose the
//! spatial dimension
//! ------------------------------------------------------------------------------
//!
{
 public:
 
 virtual inline void fillVector( real *vec, real value, int Nx, int Ny, int Nz, int span0, int span1, int span2){}
 
 virtual inline void fillVector( real *vec, real *indx, real val,int Nx, int span0){}
 
 virtual inline void copyVectorToVector(real *rhs, int rNx, int rNy, int rNz, real *lhs, int Nx, int Ny, int Nz,int span0, int span1,int span2, int direction[3]){} 
 
};

class oneDimension : public dimensionalSpace
{
 public:
   inline void fillVector(real *vec, real value, int Nx, int Ny, int Nz, int span0, int span1, int span2)
  {
    //printf("span = %d\n",span0);
    setVector1D<<<grid,block>>>(vec,value,Nx,span0);
  }
  
   inline void fillVector(real *vec, real *indx, real val, int Nx, int span0)
  {
    //setVectorUnstruc<<<grid,block>>>(vec,indx,val,Nx,span0);
  }
   
  inline void copyVectorToVector(real *rhs, int rNx, int rNy, int rNz, real *lhs, int Nx, int Ny, int Nz, int span0, int span1, int span2, int direction[3])
  {
    vecCopy1DReverse<<<grid,block>>>(rhs,rNx,lhs,Nx,span0,direction[0]);	      
  }
  

};

class twoDimensions : public dimensionalSpace
{
  public:
    inline void fillVector( real *vec, real value, int Nx, int Ny, int Nz, int span0, int span1, int span2)
   {
     setVector2D<<<grid,block>>>(vec,value,Nx,Ny,span0,span1);
   }
    inline void copyVectorToVector(real *rhs, int rNx, int rNy, int rNz, real *lhs, int Nx, int Ny, int Nz,int span0, int span1,int span2, int direction[3])
   {
     vecCopy2DReverse<<<grid,block>>>(rhs,rNx,rNy,lhs,Nx,Ny,span0,span1,direction[0],direction[1]);	      
   }
   

};

class threeDimensions : public dimensionalSpace
{
  public:
    inline void fillVector( real *vec, real value, int Nx, int Ny, int Nz, int span0, int span1, int span2)
   {
     setVector3D<<<grid,block>>>(vec,value,Nx,Ny,Nz,span0,span1,span2);
   }
    inline void copyVectorToVector(real *rhs, int rNx, int rNy, int rNz, real *lhs, int Nx, int Ny, int Nz,int span0, int span1,int span2, int direction[3])
   {
     vecCopy3DReverse<<<grid,block>>>(rhs,rNx,rNy,rNz,lhs,Nx,Ny,Nz,span0,span1,span2,direction[0],direction[1],direction[2]);	      
   }
   

};


