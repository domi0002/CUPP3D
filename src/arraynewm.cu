
distArray&  distArray::operator()(Index & I1, Index & I2)
//-------------------------------------------------
//  This 2D operator assigns the base and bound
//  for an Index object. The maximum number of
//  base, bound is defined by MAX_BASE
// I1 -- along X
// I2 -- along Y
//-------------------------------------------------
{  
   #if __ArrayBoundsCheck__
    checkArrayBounds(I1, I2, I3);
   #endif
   
   resetOffset();
   
   offset = I1.base + Nx*I2.base;
   span0  = I1.bound - I1.base;
   span1  = I2.bound - I2.base;
 
   offsetData();  
     
   return *this;
}
distArray& distArray::operator=(real val)
//-------------------------------------------------
// This operator performs a parallel assignment for 
// the vector pointed by the object. Only the indices
// defined by base and bound for the given object are
// assigned. Similar to the Matlab vector assignment
//-------------------------------------------------
{
  
  setVector2D<<<grid,block>>>(this->array.dVEC,val,Nx,Ny,span0,span1);
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
    
  
  cout<<"RHS offset = "<<larray.offset<<endl;
  cout<<"LHS offset = "<<this->offset<<endl;
  larray.array.isAllocated = false;
  
  //assert both spans are equal
  assert( (this->span0==larray.span0) && ( this->span1==larray.span1) );

  
  vecCopy2D<<<grid,block>>>(larray.array.dVEC,this->array.dVEC,Nx,Ny,larray.offset,this->offset, span0, span1);	      
  cudaDeviceSynchronize();  
  larray.resetOffset();
  this->resetOffset();

  return *this;

}


template < typename E >
void distArray::compute( E & Eob)
{
   #if(printSize)
   
   int status;
   char* passName = abi::__cxa_demangle(typeid(E).name(),0,0,&status);
   cout<<"Type of passing data     ="<<passName<<endl;
   cout<<"size of passing dataType ="<<sizeof(E)<<endl;
   free (passName);

   #endif
     
   reshapeLArray<<<grid,block>>>(Eob,dVECTemp,Nx,Ny,span0,span1);
   cudaDeviceSynchronize();
 
   copyResult<<<grid,block>>>(array.dVEC,dVECTemp,Nx,Ny,span0,span1);
   cudaDeviceSynchronize();
   
   resetOffset();
     
}


void distArray::setCudaProperties(int Nx)
{

    deviceID=0;
    cudaSetDevice(deviceID);
    int maxGridSize = Nx;
    
    threadsPerBlock = 50;
    blocksPerGrid = maxGridSize / threadsPerBlock;

    grid.x = blocksPerGrid; grid.y=1 ;
    block.x= threadsPerBlock; block.y=1;
       
}

void distArray::setCudaProperties(int Nx, int Ny, int tpb)
{

    deviceID=0;
    cudaSetDevice(deviceID);
    int maxGridSize = Nx*Ny;
    
    threadsPerBlock = tpb;
    blocksPerGrid = maxGridSize / threadsPerBlock;

    grid.x = blocksPerGrid; grid.y=1 ;
    block.x= threadsPerBlock; block.y=1;

    cudaMalloc((void**)&distArray::dVECTemp,Nx*Ny*sizeof(real) );
    setVector<<<grid,block>>>(distArray::dVECTemp,0.0,Nx*Ny);

}


distArray::distArray( int N)
{
  size = N;
  array.allocate(size);
  numberOfDimensions=1;
}

distArray::distArray( int _Nx, int _Ny)
{
  Nx = _Nx;
  Ny = _Ny; 
  size = _Nx*_Ny;
  array.allocate(size);  
  numberOfDimensions=2;
  
  offset = 0;
  offsetDone = true;
}


distArray::~distArray()
{
  
  
}
void distArray::cleanUp()
{
 
  cudaFree(dVECTemp);  
  
}



int main()
{
  #if ( useSimple )
  int Nx = 10, Ny= 10;
  distArray::setCudaProperties(Nx,Ny,50);
  distArray  u1(Nx,Ny), u2(Nx,Ny), u3(Nx,Ny);


  Index I1(4,4), I2(0,9), I3(6,6), I4(8,8), I5(0,0);
  u1(I1,I2)=10.0;
  u1(I3,I2)=15.0;
  u1(I4,I2)=25.0;
  u2(I3,I2)=11.2345;
  
  //u1(I4,I2) = 0.1*(0.1*u1(I1,I2) + 0.2*u1(I4,I2) + 0.3*u1(I3,I2) + 0.4*u1(I1,I2)) + 0.1*( u1(I1,I2) + 0.1*u1(I3,I2) );
    u3(I5,I2) = u1(I1,I2) + u2(I3,I2);
    u3(I5,I2) = u1(I1,I2) + u2(I3,I2);
  u3.pull();
  u3.display();

 #endif

 

 #if ( useJacobi )
  clock_t t1,t2;
  int Nx = 1024, Ny= 1024;
  distArray::setCudaProperties(Nx,Ny,512);
 // Jacobi Iteration
   distArray U(Nx,Ny);   // U is the solution
   
   // On a 0<x<1 0<y<1 grid with equal spacing
   float dh = (float)1/(Nx-1); // dx = dy = dh
   int nSteps = 2;      // Number of iterations
   
   Index IXR(2,Nx-1), IXL(0,Nx-3);
   Index IYT(2,Ny-1), IYB(0,Ny-3);
   
   Index IX(1,Nx-2), IY(1,Ny-2);
  
   // Set Boundary conditions, u = 0.5 on the boundary everywhere
   // Lower Boundary
   Index IXBL(0,Nx-1), IYBL(0,0);
   U(IXBL,IYBL) = 5;
   // Upper Boundary
   Index IXBU(0,Nx-1), IYBU(Ny-1,Ny-1);
   U(IXBU,IYBU) = 5;
   // Left Boundary
   Index IXBLL(0,0), IYBLL(0,Ny-1);
   U(IXBLL,IYBLL) = 5;
   // Right Boundary
   Index IXBR(Nx-1,Nx-1),IYBR(0,Ny-1);
   U(IXBR,IYBR) = 5;
  
  // U.pull(); // no need to pull unless we are printing the values...
   
   // Display Initial conditions with boundary values
  // U.display();
  
  //clock_t t1, t2;
  
  
  // Magical Jacobi Iteration 
  t1 = clock();
   for ( int i = 0 ; i < nSteps; i++)
     {
      U(IX,IY) =   0.25*(  U(IXR,IY) +  U(IXL,IY) +  U(IX,IYT) + U(IX,IYB)   );     
      //U.pull(); // Use this unless you want to pull back the values to CPU.
      //U.display();
     }
  t2 = clock()-t1;
  
  cout<<" Number of clock cycles CU++ = "<<t2<<endl;
  
  U.pull(); // Use this unless you want to pull back the values to CPU.
  // Display Final values
  //Index IPR(34,38);
  //U.display();
  
  // +++++++++++++++++++++++++++++++++++++++++++++++++++
  // Serial Jacobi Code:
  // +++++++++++++++++++++++++++++++++++++++++++++++++++
  float *u = new float [ Nx*Ny ]; // We prefer the use of 1D arrays for multiD problems
  float *un= new float [ Nx*Ny ];
  
  cout<<"Running serial code"<<endl;
  
  // Initialize all to zero
  for ( int i = 0 ; i < Nx*Ny ; i++)
    {
     u[i]=0.0;
     un[i]=0.0;
    }
     
  // Set Boundary conditions
  // Lower boundary->
  for ( int i = 0 ; i < Nx ; i++ )
   u[i] =5;
  
  // Upper boundary->
  for ( int i = 0 ; i < Nx ; i++)
   u[i+Nx*(Ny-1)] = 5;
   
  // Left boundary->
  for ( int i = 0 ; i < Ny ; i++)
  u[Nx*i] = 5;
  
  // Right boundary->
  for ( int i = 0; i < Ny ; i++)
  u[Nx-1 + Nx*i] = 5; 

  // Snail Mail
  t1 = clock();  
     for ( int st = 0 ; st < nSteps ; st++ )
      {
        
	for ( int i = 1 ; i < Ny-1 ; i++ )
	  for ( int j = 1 ; j < Nx-1 ; j++)
             un[j+Nx*i] = 0.25*( u[j+1 + Nx*i] + u[j-1 + Nx*i] + u[j + Nx*(i+1)] + u[j + Nx*(i-1)]) ;      
        for ( int i = 1 ; i < Ny-1 ; i++)
	 for ( int j = 1; j < Nx-1 ; j++)
	   u[j+Nx*i] =un[j+Nx*i];
      
      }
 t2 = clock()-t1;
 cout<<" Number of clock cycles C++ = "<<t2<<endl; 
  
  //for ( int i  = 0 ; i < Nx*Ny ; i++)
  //  cout<<u[i]<<endl;
  // Compute Error between GPU and CPU
  real error = 0.0;
   for ( int i = 0 ; i < Ny ; i++)
      for ( int j = 0 ; j < Nx ; j++)
          error+=(u[j+Nx*i]-U(j,i))*(u[j+Nx*i]-U(j,i));

  cout<<"Error between CPU and GPU = "<<error<<endl;

  delete [] u;
  delete [] un;
 
 #endif
  distArray::cleanUp();
}



