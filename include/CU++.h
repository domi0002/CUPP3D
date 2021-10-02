//! ------------------------------------------------------------------
//! Solving Discrete equations using CUDA and Expression Templates
//! This code generates CUDA kernels automatically at compile time
//! CU++ET or CUPPET
//! 10/18/2010
//! Dominic Chandar
//! University of Wyoming (2010)
//! Queen's University Belfast (2021)
//! ------------------------------------------------------------------
//!
#ifndef __CUPPET_H__
#define __CUPPET_H__


//! Use this if you want to print the size of the Abstract Object being passed to the device 
#if (printPassingTypeAndSize )
 #include<cxxabi.h>
#endif
//!

//! Standard Header Includes
#include <iostream>
#include <typeinfo>
#include <assert.h>
#include <stdio.h>
//!

//! Tesla C1060 supports float and double
//! Tesla C2050,C2070 supports double with improved performance.
#ifdef CUPP_DOUBLE
 #define real double
#else
 #define real float
#endif
//!

#ifdef __SERIAL__
#include <stdlib.h>
#include <math.h>
#include "CUSerial.h"
#endif

//! Other Misc Includes
#ifdef __SERIAL__
 #include "SdeviceFunctions.cu"
#else
 #include "deviceFunctions.cu"
#endif
#include "dimensions.h"
//!

//! Mix 'n' Match : CUDA + MPI 
//! The Current Implementation supports Row-wise partitioning ONLY ( for gathering operation )
#ifdef useMultiGPU
 #define useHybridCUDAMPI
 #include "mpi.h"
  #ifdef MPI_REAL // used in MPI-fortran, so we discard its use here
    #undef MPI_REAL
  #endif
 #define MPI_REAL MPI_FLOAT
#endif
//!

using namespace std;
class cartGrid;
class vectorGridFunction;

class Array
//! -----------------------------------------------------------------------------
//! This class is used to represent an array (1/2/3 dimensions)
//! The CUDA Kernel utilizes the array named dVEC to peform arithmetic operations
//! -----------------------------------------------------------------------------
{
  public:
  Array()
  {
   isAllocated = false;
  }
  template <typename E>
  Array( int N, E* dptr, int Nx, int Ny, int Nz, int nD)
  {
   //VEC = new real [N];
   cudaMalloc((void**)dVEC,N*sizeof(real) );
   dptr->fillVector(dVEC,0.0,Nx,Ny,Nz,Nx-1,Ny-1,Nz-1);
   isAllocated = true ;
  }
  template <typename E>
  void allocate(int N, E* dptr, int Nx, int Ny, int Nz, int nD)
  {
   //VEC = new real [N];
   cudaMalloc((void**)&dVEC,N*sizeof(real) );
   dptr->fillVector(dVEC,0.0,Nx,Ny,Nz,Nx-1,Ny-1,Nz-1);
   isAllocated = true ;
  
  }
  void allocate(int N, bool & init)
  {
   //VEC = new real [N];
   cudaMalloc((void**)&dVEC,N*sizeof(real) );

   if ( init )
   setVector<<<grid,block>>>(dVEC,0.0,N);

   isAllocated = true ;
  
  }
  void deAllocate()
  {
    if ( isAllocated )
     { 
      //delete [] VEC ;
      cudaFree(dVEC);
      //dVEC=NULL;
      isAllocated = false;
     }
  } 
  ~Array()
   { 
   if ( isAllocated )
     {
     
      //delete [] VEC ;
      #ifndef __SERIAL__
          cudaFree(dVEC);
      #endif
      //dVEC=NULL;
      isAllocated= false; 
     }
   }
  //real *VEC;
  real *dVEC;
 
  mutable bool isAllocated;
  
  __device__  real & operator []( int i ) {       return dVEC[i]; }
  __device__  real   operator []( int i ) const { return dVEC[i]; }
  
};

class Index
//! ------------------------------------------------------------------------
//! The Index class is used to store the base and bounds of an array object
//! Each time the operator () is called with Index object as a parameter,
//! the base and bounds of the array at that instant is recorded
//! This class also supports Index shifting via overloading the +/- operators
//! Eg: Index I(start,end), or Index I(0,5)
//! ------------------------------------------------------------------------
{
   public:
  int base;
  int bound;
  int direction; // +1 is from lowest to highest, -1 is from highest to lowest

  Index(int from_, int to_) {base=from_; bound=to_; direction=1; }
  Index(int from_, int to_, int direction_){base=from_; bound=to_;direction=direction_;}
  Index(){direction = 1;}
  ~Index(){}
  inline void flip(){ direction = -direction; }
  inline void display(){
   cout<<" Base = "<<base<<"  Bound = "<<bound<<endl;
                     }
  inline int getLength(){ return bound-base+1 ; }

};

class distArray
//! ---------------------------------------------------------------------------
//! The Main class for array operations. This has all the information that one
//! needs regarding a particular array.
//! Eg: To declare an array 
//!     distArray u(size_along_x)                                : 1 Dimension
//!     distArray u(size_along_x, size_along_y)                  : 2 Dimensions
//!     distArray u(size_along_x, size_along_y, size_along_z)    : 3 Dimensions 
//! All operations MUST be in reference to this particular class
//! ---------------------------------------------------------------------------
{
 public:
   distArray(int N);
   distArray(int Nx, int Ny);
   distArray(int Nx, int Ny, int Nz);

   
   distArray( cartGrid & cg);
  
   distArray(){}
  ~distArray();
   int size;
   int Nx, Ny, Nz;
   int numberOfDimensions;
   int* nFringeArray;
   bool arrayInitializedByGrid;
   Array array;
   static real *dVECTemp;
   static bool temporaryNeedsToBeAllocated;
   real & operator()(int i)               { return VEC[i]; }
   real & operator()(int i, int j)        { return VEC[i+Nx*j]; }
   real & operator()(int i, int j, int k) { return VEC[i+Nx*j+Nx*Ny*k]; }
   
   
   distArray & operator()(Index const I1);

   
   distArray & operator()(distArray const u);
      
   template < typename E1, typename E2>
   distArray & operator()(E1 const I1, E2 const I2);
 
   template < typename E1, typename E2, typename E3>
   distArray & operator()(E1 const I1, E2 const I2, E3 const I3);

   distArray & operator = (real val);
   distArray & operator = (  distArray   larray);

   #ifdef useHybridCUDAMPI
    inline static void gatherLocalArrayToRoot( distArray* ulocal, distArray* uRoot);
    static int MPI_RANK;
   #endif

   int offset ;
   int span0;
   int span1;
   int span2;
   bool offsetDone;
   int direction[3];

   inline void offsetData ( )
   { 
     array.dVEC += offset;    
     offsetDone = false;
   }

   inline int getOffset()
   {
     return offset;
   }

  inline void resetOffset ( )
   {
    if ( !offsetDone )
      {
        offsetDone = true;
        array.dVEC -= offset; 
      }
   }

  inline void resetOffset (char* s )
   {
    if ( !offsetDone )
      {
        cout<<s<<endl;
        offsetDone = true;
        array.dVEC -= offset; 
      }
   }
  
  /* not working yet */inline void transpose() { int temp = Nx; Nx=Ny ; Ny = temp ;}

  __device__  real & operator []( int i ) {       return array.dVEC[i]; }
  __device__  real   operator []( int i ) const { return array.dVEC[i]; }


   
   template < typename E >
   inline void compute(E & Eob);

   template < typename E >  
   distArray & operator = ( E & Eob) 
    {
     compute(Eob);
     return *this;
    }

  inline void display();
  inline void display(Index & I1);
  inline void display(Index & I1, Index & I2);
  inline void display(Index & I1, Index & I2, Index & I3);
  inline void outputArrayToFile(char* file);
  
  void update( cartGrid & cg); 

  void push() {cudaMemcpy(this->array.dVEC, this->VEC, size*sizeof(real), cudaMemcpyHostToDevice) ;} 
  void pull() {cudaMemcpy(this->VEC, this->array.dVEC, size*sizeof(real), cudaMemcpyDeviceToHost) ;}
  
  // Pull only specific elements from GPU as directed by the Index I1, I2, and I3.
  void pull( Index & I1 )
   {
    cudaMemcpy(this->VEC+I1.base, this->array.dVEC+I1.base, (I1.bound-I1.base+1)*sizeof(real), cudaMemcpyDeviceToHost) ;
   }
  void pull( Index & I1, Index & I2 )
   {
    int shift = I1.base+Nx*I2.base;
    int sizex = I1.bound-I1.base+1;
    int sizey = I2.bound-I2.base+1;
    cudaMemcpy2D(this->VEC+shift,       this->Nx*sizeof(real),
                 this->array.dVEC+shift,this->Nx*sizeof(real), sizex*sizeof(real), sizey*sizeof(real),cudaMemcpyDeviceToHost) ;
    array.isAllocated = false; // doesnt work without this ??  but standard pull() works without this.. :(
   }  

  #ifdef useHybridCUDAMPI
  // parallel pull() from specific process
  void pull(int rank) 
   { 
     if ( distArray::MPI_RANK = rank )
     cudaMemcpy(this->VEC, this->array.dVEC, size*sizeof(real), cudaMemcpyDeviceToHost) ;
   }
  #endif


  // Report memory usage:
  static int nBytes;
  
  static void setCudaProperties(int Nx, int tpb);
  static void setCudaProperties(int Nx, int tpb,int device);

  static void setCudaProperties(int Nx, int Ny, int tpbx, int tpby);
  static void setCudaProperties(int Nx, int Ny, int tpbx, int tpby, int device);

  static void setCudaProperties(int Nx, int Ny, int Nz, int tpbx, int tpby, int tpbz);
  static void setCudaProperties(int Nx, int Ny, int Nz, int tpbx, int tpby, int tpbz, int device);
  static void cleanUp(); 
  dimensionalSpace *__dim__[3]; // Increase this number if we need higher dimensions...
  real *VEC;
  
  //! -------------------------
  //! Unstructured data
  //! -------------------------
     real* uStrucIndex;
     bool useUnstructuredKernel;
  //!---------------------------
  
};

 
 bool distArray::temporaryNeedsToBeAllocated=true;
 #ifdef useHybridCUDAMPI
  int distArray::MPI_RANK;
 #endif

 #ifdef useHybridCUDAMPI
  template < typename E1, typename E2 >
  void pPrintf(E1 str1, E2 str2, int rank)
   {
     if ( distArray::MPI_RANK == rank)
       cout<<"Rank =  "<<rank<<"  "<<str1<<" "<<str2<<endl;
   } 
  #endif

 int distArray::nBytes;

 #ifndef __deviceTempVector__
  #define __deviceTempVector__
    real* distArray::dVECTemp;   
 #endif
 
#include "addTemplate.cu"
#include "subTemplate.cu"
#include "mulTemplate.cu"
#include "divTemplate.cu"
#include "templateDefinitions.cu"
#include "mathFunctions.cu"
#include "grid.h"
#include "distArray.cu"
#include "boundaryConditionSetup.h"
#include "vectorGridFunction.h"
#include "vectorGridFunction.cu"
#include "grid.cu"
#include "clock.C"


#endif



