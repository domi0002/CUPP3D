//! ------------------------------------------------------------------
//! CU++ET or CUPPET
//! 10/12/2010
//! Dominic Chandar
//! University of Wyoming
//! Solving Discrete equations using CUDA and Expression Templates
//! This code generates CUDA kernels automatically at compile time
//! 
//! CODE   : distArray.cu
//! Purpose:
//! This section of the code has all the function definitions of
//! the class distArray. 
//! 
//! ------------------------------------------------------------------
//!
//#include "CU++.h"


distArray&  distArray::operator()(Index const I1)
//-------------------------------------------------
//  This 1D operator assigns the base and bound
//  for an Index object. 
// I1 -- along X
// I2 -- along Y
//-------------------------------------------------
{  
   #if __ArrayBoundsCheck__
    checkArrayBounds(I1, I2, I3);
   #endif
   
   resetOffset();
   
   offset       =  I1.base  + nFringeArray[0];
   span0        =  I1.bound - I1.base;
   direction[0] =  I1.direction;

   offsetData();  
     
   return *this;
}


distArray & distArray::operator()( distArray const  u )
//! ----------------------------------------------------------
//! The array u holds the indices where *this has to be set
//! The unstructured kernel is called in this case
//! Only implemented in 1D as unstructured data can be cast in 1D
//! 11/01/2010
//! ----------------------------------------------------------
{
 u.array.isAllocated = false;
 uStrucIndex = u.array.dVEC;
 
 span0 = u.span0;
 useUnstructuredKernel = true; 
 return *this ;
}

template < typename E1, typename E2 >
distArray&   distArray::operator()(E1 const I1, E2 const I2)
//-------------------------------------------------
//  This 2D operator assigns the base and bound
// I1 -- along X
//  for an Index object. 
// I2 -- along Y
//-------------------------------------------------
{  
   #if __ArrayBoundsCheck__
    checkArrayBounds(I1, I2, I3);
   #endif
   
   resetOffset();
   
   offset       = I1.base + nFringeArray[0] + Nx*( I2.base + nFringeArray[1] );
   span0        = I1.bound - I1.base;
   span1        = I2.bound - I2.base;
   direction[0] = I1.direction;
   direction[1] = I2.direction; 

   offsetData();  
     
   return *this;
}


template < typename E1, typename E2, typename E3 >
distArray&  distArray::operator()(E1 const I1, E2 const I2, E3 const I3)
//-------------------------------------------------
//  This 3D operator assigns the base and bound
//  for an Index object. 
// I1 -- along X
// I2 -- along Y
//-------------------------------------------------
{  
   #if __ArrayBoundsCheck__
    checkArrayBounds(I1, I2, I3);
   #endif
   
   resetOffset();
   
   offset = I1.base + nFringeArray[0] 
                    + Nx*(    I2.base + nFringeArray[1] ) 
                    + Nx*Ny*( I3.base + nFringeArray[2] );

   
   span0  = I1.bound - I1.base;
   span1  = I2.bound - I2.base;
   span2  = I3.bound - I3.base;

   //cout<<I1.direction<<" "<<I2.direction<<" "<<I3.direction<<endl;
   direction[0] = I1.direction;
   direction[1] = I2.direction;
   direction[2] = I3.direction;

   offsetData();  
     
   return *this;
}


distArray& distArray::operator=(real val)
//-------------------------------------------------
// This operator performs a parallel assignment for 
// the vector pointed by the object. Only the indices
// defined by base and bound for the given object are
// assigned. 
//-------------------------------------------------
{
  // Smart Assignment without if loop. Works for 1/2/3 dimensions
  if ( !useUnstructuredKernel )
     __dim__[numberOfDimensions-1]->fillVector(this->array.dVEC,val,Nx,Ny,Nz,span0,span1,span2);
  else
   {
     __dim__[numberOfDimensions-1]->fillVector(this->array.dVEC,this->uStrucIndex,val,Nx,span0);
   }
  
  cudaDeviceSynchronize();
  
  resetOffset();
  
  return *this;

}

distArray & distArray::operator=(  distArray  larray)
//-------------------------------------------------
// This operator copies the vector pointed by array
// into the vector pointed by the this pointer.
// This also resets the counter called gc which keeps
// track of the temporary arrays created. This assignment
// should always be the last step of an operation
//-------------------------------------------------
{
      
 
  larray.array.isAllocated = false;
 
  // Smart Assignment without if loop. Works for 1/2/3 dimensions
  __dim__[numberOfDimensions-1]->copyVectorToVector(larray.array.dVEC,larray.Nx, larray.Ny, larray.Nz,this->array.dVEC,Nx,Ny,Nz,span0,span1,span2,larray.direction);	      
  
  cudaDeviceSynchronize();  
  larray.resetOffset();
  this->resetOffset();

  return *this;

}


void distArray::setCudaProperties(int Nx, int tpb)
//!  ----------------------------------------------------------------------------------
//!  Set the CUDA grid properties for a one-dimensional array
//!  True one-dimensional layout where each thread corresponds to 
//!  a grid co-ordinate (x)
//!  10/08/2010
//!  ----------------------------------------------------------------------------------
//!
{

    deviceID=0;
    cudaSetDevice(deviceID);
    int maxGridSize = Nx;
    
    threadsPerBlockx = tpb;
    blocksPerGridx = maxGridSize / threadsPerBlockx;

    grid.x = blocksPerGridx; grid.y=1 ;
    block.x= threadsPerBlockx; block.y=1;

    if ( temporaryNeedsToBeAllocated )
    {
      cudaMalloc((void**)&distArray::dVECTemp,Nx*sizeof(real) );
      setVector1D<<<grid,block>>>(distArray::dVECTemp,0.0,Nx,Nx-1);
      temporaryNeedsToBeAllocated = false;
    }     
  
}

void distArray::setCudaProperties(int Nx, int tpb, int device)
//!  ----------------------------------------------------------------------------------
//!  Set the CUDA grid properties for a one-dimensional array
//!  True one-dimensional layout where each thread corresponds to 
//!  a grid co-ordinate (x)
//!  10/08/2010
//!  ----------------------------------------------------------------------------------
//!
{

    //deviceID=0;
    cudaSetDevice(device);
    int maxGridSize = Nx;
    
    threadsPerBlockx = tpb;
    blocksPerGridx = maxGridSize / threadsPerBlockx;

    grid.x = blocksPerGridx; grid.y=1 ;
    block.x= threadsPerBlockx; block.y=1;

    if ( temporaryNeedsToBeAllocated )
    {
      cudaMalloc((void**)&distArray::dVECTemp,Nx*sizeof(real) );
      setVector1D<<<grid,block>>>(distArray::dVECTemp,0.0,Nx,Nx-1);
      temporaryNeedsToBeAllocated = false;
    }     
  
}

void distArray::setCudaProperties(int Nx, int Ny, int tpbx, int tpby)
//!  ----------------------------------------------------------------------------------
//!  Set the CUDA grid properties for a two-dimensional array
//!  True two-dimensional layout where each thread corresponds to 
//!  a grid co-ordinate (x,y)
//!  10/10/2010
//!  ----------------------------------------------------------------------------------
//!
{

    deviceID=0;
    cudaSetDevice(deviceID);
       
    threadsPerBlockx = tpbx;
    threadsPerBlocky = tpby;
    blocksPerGridx = Nx / threadsPerBlockx;
    blocksPerGridy = Ny / threadsPerBlocky;

    grid.x  = blocksPerGridx;   grid.y  = blocksPerGridy ;
    block.x = threadsPerBlockx; block.y = threadsPerBlocky;
    
    if ( temporaryNeedsToBeAllocated)
    {
     cudaMalloc((void**)&distArray::dVECTemp,Nx*Ny*sizeof(real) );
     setVector2D<<<grid,block>>>(distArray::dVECTemp,0.0,Nx,Ny,Nx-1,Ny-1);
     temporaryNeedsToBeAllocated = false;
    }     
}

void distArray::setCudaProperties(int Nx, int Ny, int tpbx, int tpby, int device)
//!  ----------------------------------------------------------------------------------
//!  Set the CUDA grid properties for a two-dimensional array
//!  True two-dimensional layout where each thread corresponds to 
//!  a grid co-ordinate (x,y)
//!  10/10/2010
//!  ----------------------------------------------------------------------------------
//!
{

    //deviceID=0;
    cudaSetDevice(device);
       
    threadsPerBlockx = tpbx;
    threadsPerBlocky = tpby;
    blocksPerGridx = Nx / threadsPerBlockx;
    blocksPerGridy = Ny / threadsPerBlocky;

    grid.x  = blocksPerGridx;   grid.y  = blocksPerGridy ;
    block.x = threadsPerBlockx; block.y = threadsPerBlocky;
    
    if ( temporaryNeedsToBeAllocated)
    {
     cout<<" Allocating Temporay array on device :"<<device<<endl;
     cudaMalloc((void**)&distArray::dVECTemp,Nx*Ny*sizeof(real) );
     setVector2D<<<grid,block>>>(distArray::dVECTemp,0.0,Nx,Ny,Nx-1,Ny-1);
     temporaryNeedsToBeAllocated = false;
    }     
}

void distArray::setCudaProperties(int Nx, int Ny, int Nz, int tpbx, int tpby, int tpbz)
//!  ----------------------------------------------------------------------------------
//!  Set the CUDA grid properties for a three-dimensional array
//!  Quasi-3D Setting of grid and block properties
//!  Strictly there can be only 1 block along the z-direction
//!  Hence the z- direction is merged along the y-direction and the resulting
//!  grid looks like Nx * (Ny*Nz) rather than Nx * Ny * Nz
//!  10/12/2010
//!  ----------------------------------------------------------------------------------
//!
{
    deviceID=0;
    cudaSetDevice(deviceID);
       
    threadsPerBlockx = tpbx;
    threadsPerBlocky = tpby*tpbz;
    threadsPerBlockz = tpbz ;// Used Only when this function is called again..    

    blocksPerGridy = Ny*Nz / threadsPerBlocky;
    blocksPerGridx = Nx / threadsPerBlockx;
    

    grid.x  = blocksPerGridx;   grid.y  = blocksPerGridy ;  grid.z=1;
    block.x = threadsPerBlockx; block.y = threadsPerBlocky; block.z=1;
    
    if ( temporaryNeedsToBeAllocated )
    {
     
     cudaMalloc((void**)&distArray::dVECTemp,Nx*Ny*Nz*sizeof(real) );
     setVector3D<<<grid,block>>>(distArray::dVECTemp,0.0,Nx,Ny,Nz,Nx-1,Ny-1,Nz-1);
     temporaryNeedsToBeAllocated = false;
    }
    
}

void distArray::setCudaProperties(int Nx, int Ny, int Nz, int tpbx, int tpby, int tpbz, int device)
//!  ----------------------------------------------------------------------------------
//!  Set the CUDA grid properties for a three-dimensional array
//!  Quasi-3D Setting of grid and block properties
//!  Strictly there can be only 1 block along the z-direction
//!  Hence the z- direction is merged along the y-direction and the resulting
//!  grid looks like Nx * (Ny*Nz) rather than Nx * Ny * Nz
//!  10/12/2010
//!  ----------------------------------------------------------------------------------
//!
{
    //deviceID=0;
    cudaSetDevice(device);
       
    threadsPerBlockx = tpbx;
    threadsPerBlocky = tpby*tpbz;
    threadsPerBlockz = tpbz ;// Used Only when this function is called again..    

    blocksPerGridy = Ny*Nz / threadsPerBlocky;
    blocksPerGridx = Nx / threadsPerBlockx;
    

    grid.x  = blocksPerGridx;   grid.y  = blocksPerGridy ;  grid.z=1;
    block.x = threadsPerBlockx; block.y = threadsPerBlocky; block.z=1;
    
    if ( temporaryNeedsToBeAllocated )
    {
     cudaMalloc((void**)&distArray::dVECTemp,Nx*Ny*Nz*sizeof(real) );
     setVector3D<<<grid,block>>>(distArray::dVECTemp,0.0,Nx,Ny,Nz,Nx-1,Ny-1,Nz-1);
     temporaryNeedsToBeAllocated = false;
    }
    
}



distArray::distArray( int N)
{
  __dim__[0] = new oneDimension();
  numberOfDimensions=1;
  useUnstructuredKernel=false;
  size = N;
  Nx   = N;
  arrayInitializedByGrid = false;
  array.allocate(size,__dim__[0],Nx,1/*Ny*/,1/*Nz*/,1/*dimensions*/);
  VEC = new real [size];

  nFringeArray = new int[6];
  for ( int i = 0 ; i < 6 ; i++)
    nFringeArray[i]=0;

  direction[0] = 1;
  direction[1] = 1;
  direction[2] = 1;
  offset = 0;
  offsetDone = true;

  
}

distArray::distArray( int _Nx, int _Ny)
{
  __dim__[1] = new twoDimensions();
  numberOfDimensions=2;

  Nx = _Nx;
  Ny = _Ny; 
  size = _Nx*_Ny;
  arrayInitializedByGrid = false;
  useUnstructuredKernel = false;
 
  array.allocate(size,__dim__[1],Nx,Ny,1,2);  
  VEC = new real [size];
  
  nFringeArray = new int[6];
  for ( int i = 0 ; i < 6 ; i++)
    nFringeArray[i]=0;
  
  direction[0] = 1;
  direction[1] = 1;
  direction[2] = 1;
 
  offset = 0;
  offsetDone = true;

  

}

distArray::distArray( int _Nx, int _Ny, int _Nz)
{

  __dim__[2] = new threeDimensions();
  numberOfDimensions=3;

  Nx = _Nx;
  Ny = _Ny; 
  Nz = _Nz;
  size = _Nx*_Ny*_Nz;
  arrayInitializedByGrid = false;
  VEC = new real [size];
  useUnstructuredKernel = false;
  array.allocate(size,__dim__[2],Nx,Ny,Nz,3);  
  
  nFringeArray = new int[6];
  for ( int i = 0 ; i < 6 ; i++)
    nFringeArray[i]=0;  

  direction[0] = 1;
  direction[1] = 1;
  direction[2] = 1;

  offset = 0;
  offsetDone = true;


}

distArray::distArray( cartGrid & cg )
//! ---------------------------------------------------
//! Initialize a distArray from a known cartesian Grid
//! ---------------------------------------------------
{
   Nx = cg.cartGridNx;
   Ny = cg.cartGridNy;
   Nz = cg.cartGridNz;
   size   = Nx*Ny*Nz;
   numberOfDimensions = cg.numberOfDimensions;
   nFringeArray = &cg.nFringe[0][0];
   arrayInitializedByGrid = true;
   VEC = new real [size];

   if ( numberOfDimensions == 1 )
    {
      __dim__[0] = new oneDimension();
      array.allocate(size,__dim__[0],Nx,1,1,1);  
    }
   if ( numberOfDimensions == 2 )
    {
      __dim__[1] = new twoDimensions();
      array.allocate(size,__dim__[1],Nx,Ny,1,2);  
    }
   if ( numberOfDimensions == 3 )
    {
      __dim__[2] = new threeDimensions();
      array.allocate(size,__dim__[2],Nx,Ny,Nz,3);     
    } 

   direction[0] = 1;
   direction[1] = 1;
   direction[2] = 1;

   offset     = 0;
   offsetDone = true;
}


void distArray::update( cartGrid & cg )
//! --------------------------------------------------------------
//! Update an existing distArray from a known cartesian Grid
//! This routine may be called if fringe points have been assigned
//! The old vector is deleted and a new vector is formed
//! --------------------------------------------------------------
{
   Nx = cg.cartGridNx;
   Ny = cg.cartGridNy;
   Nz = cg.cartGridNz;
   size   = Nx*Ny*Nz;
   numberOfDimensions = cg.numberOfDimensions;
   nFringeArray = &cg.nFringe[0][0];
   arrayInitializedByGrid = true;

   if ( numberOfDimensions == 1 )
    {
       assert( __dim__[0] != NULL ); // Must be previously assigned
       array.deAllocate();
       
       if ( VEC != NULL)
       delete [] VEC;
       array.allocate(size,__dim__[0],Nx,1,1,1);  
       VEC = new real [size];
    }
   if ( numberOfDimensions == 2 )
    {
       assert( __dim__[1] != NULL ); // Must be previously assigned
       array.deAllocate();

       if ( VEC != NULL)
       delete [] VEC;
       array.allocate(size,__dim__[1],Nx,Ny,1,2);  
       VEC = new real [size];
    }
   if ( numberOfDimensions == 3 )
    {
       assert( __dim__[2] != NULL ); // Must be previously assigned
       array.deAllocate();
      
       if ( VEC != NULL)
       delete [] VEC;
       array.allocate(size,__dim__[2],Nx,Ny,Nz,3);     
       VEC = new real [size];
    } 

   offset     = 0;
   offsetDone = true;
}

#ifdef useHybridCUDAMPI
void distArray::gatherLocalArrayToRoot( distArray* uLocal, distArray* uRoot)
//! 10/18/2010
//! This routine assumes Row-wise partitioning ONLY
{
  // first pull all values from all processors 
  uLocal->pull(); 
  MPI_Barrier(MPI_COMM_WORLD);
  // Gather all values to processor 0    
  MPI_Gather(uLocal->VEC,uLocal->Nx*uLocal->Ny,MPI_FLOAT,uRoot->VEC,uLocal->Nx*uLocal->Ny,MPI_FLOAT,0,MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD);

}
#endif

void distArray::display( Index & I1 )
{
   assert(numberOfDimensions==1);
   for ( int i =I1.base ; i < I1.bound+1 ; i++)
	 {
	  printf("        %d   ",i);
	 }
	  printf("\n"); 
   for ( int i = I1.base ; i < I1.bound+1; i++)
	  printf("-------------");
	  printf("\n");

   for ( int j = I1.base ; j < I1.bound+1 ; j++)
	  { 
            #ifndef CUPP_DOUBLE
            printf("   %f ",VEC[j]);
            #else
            printf("   %lf",VEC[j]);
            #endif
	  } 
            printf("\n");

}

void distArray::display(Index & I1,Index & I2)
{
   
   assert(numberOfDimensions==2);
   
        for ( int i =I1.base ; i < I1.bound+1 ; i++)
	 {
	  printf("        %d   ",i);
	 }
	  printf("\n"); 
        for ( int i = I1.base ; i < I1.bound+1; i++)
	  printf("-------------");
	  printf("\n");
      
     
      for ( int i = I2.base ; i < I2.bound+1 ; i++ )
       {
            printf("%d",i);
         for ( int j = I1.base ; j < I1.bound+1 ; j++)
	  { 
            printf("   %f ",VEC[j+i*Nx]);
	   } 
            printf("\n");
       }   
       
}

void distArray::display(Index & I1,Index & I2, Index & I3)
{
   
   assert(numberOfDimensions==3);
   
   for ( int k = I3.base ; k < I3.bound+1 ; k++)
    {
         printf(" Printing values for Z plane =%d\n",k);
      
        for ( int i =I1.base ; i < I1.bound+1 ; i++)
	 {
	  printf("        %d   ",i);
	 }
	  printf("\n"); 
        for ( int i = I1.base ; i < I1.bound+1; i++)
	  printf("-------------");
	  printf("\n");
      
     
      for ( int i = I2.base ; i < I2.bound+1 ; i++ )
       {
            printf("%d",i);
         for ( int j = I1.base ; j < I1.bound+1 ; j++)
	  { 
            printf("   %f ",VEC[j+i*Nx+k*Nx*Ny]);
	   } 
            printf("\n");
       }   
            printf("\n");
    
    }   
}
void distArray::display()
{
  if ( numberOfDimensions == 1)
   {
     Index I1(0,Nx-1);
     display(I1);
   }

  if ( numberOfDimensions == 2)
   {
     Index I1(0,Nx-1), I2(0,Ny-1);
     display(I1,I2);
   }

  if ( numberOfDimensions == 3)
   {
     Index I1(0,Nx-1), I2(0,Ny-1), I3(0,Nz-1);
     display(I1,I2,I3);
   }

}

void distArray::outputArrayToFile(char* file)
{
   FILE *fp = fopen(file,"w");

   if ( numberOfDimensions == 1)
     {
      for ( int i = -nFringeArray[0] ; i < size-nFringeArray[3] ; i++)
         fprintf(fp,"%f\t",VEC[i + nFringeArray[0] ]);
     }  
   else if ( numberOfDimensions == 2)
     {
        
     
      for ( int i = -nFringeArray[1] ; i < Ny-nFringeArray[4] ; i++ )
       {
            
         for ( int j = -nFringeArray[0] ; j < Nx-nFringeArray[3] ; j++)
	  { 
            fprintf(fp,"   %f ",VEC[j + nFringeArray[0] 
	                          + (i + nFringeArray[1])*Nx]);
	   } 
            fprintf(fp,"\n");
       }   
    }
   else if ( numberOfDimensions == 3)
    { 
      for ( int k = -nFringeArray[2] ; k < Nz-nFringeArray[5] ; k++)
    {
       for ( int i = -nFringeArray[1] ; i < Ny-nFringeArray[4] ; i++ )
       {
         for ( int j = -nFringeArray[0] ; j < Nx-nFringeArray[3] ; j++)
	  { 
            fprintf(fp,"   %f ",VEC[ j+nFringeArray[0]  
	                          +(i + nFringeArray[1])*Nx
				  +(k + nFringeArray[2])*Nx*Ny]);
	   } 
            fprintf(fp,"\n");
       }   
    
            fprintf(fp,"\n");
    }   
     
    }   
     
      fclose(fp);
}


distArray::~distArray()
{
  if ( !arrayInitializedByGrid && array.isAllocated )
    {
      delete [] nFringeArray;
      
     }

  if ( array.isAllocated )
    delete [] VEC ;
}


void distArray::cleanUp()
{
 
  cudaFree(dVECTemp);  
  
}



