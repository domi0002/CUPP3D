int __FACELOOP__::Nx;
int __FACELOOP__::Ny;
int __FACELOOP__::Nz;
int __FACELOOP__::MNx;
int __FACELOOP__::MNy;
int __FACELOOP__::MNz;

// Done conversion
cartGrid::cartGrid(real xmin, real xmax, int Nx)
{
 numberOfDimensions = 1;
 cartBase0 = xmin;
 cartBound0= xmax;
 
 cartGridNx   = Nx;
 cartGridNy   = 1;
 cartGridNz   = 1;
 gridCoord[0] = new distArray(Nx);
 
 hxspace = (xmax - xmin)/(Nx-1);
 
 setVectorGrid1D<<<grid,block>>>(gridCoord[0]->array.dVEC, hxspace, Nx, xmin, xmax);
 cudaDeviceSynchronize();

}

// Done conversion
cartGrid::cartGrid(real xmin, real xmax, int Nx, real ymin, real ymax, int Ny)
{
 numberOfDimensions = 2;
 cartBase0 = xmin;
 cartBound0= xmax;
 cartBase1 = ymin;
 cartBound1= ymax;
 cartBase2 = 0;
 cartBound2= 0;
 
 cartGridNx = Nx;
 cartGridNy = Ny;
 cartGridNz = 1;
 
 gridCoord[0] = new distArray(Nx,Ny);
 gridCoord[1] = new distArray(Nx,Ny);
 
 hxspace = (xmax - xmin)/(Nx-1);
 hyspace = (ymax - ymin)/(Ny-1);
 
 for ( int axis = 0 ; axis < 3 ; axis++)
   for ( int side = 0; side < 2 ; side++)
       nFringe[side][axis]=0;
 
 setVectorGrid2D<<<grid,block>>>(gridCoord[0]->array.dVEC,gridCoord[1]->array.dVEC,
                                                       hxspace, hyspace, Nx, Ny, xmin, xmax, ymin, ymax);
                                                                        
 cudaDeviceSynchronize();									

}

// Done conversion
cartGrid::cartGrid(real xmin, real xmax, int Nx, 
                   real ymin, real ymax, int Ny,
		   real zmin, real zmax, int Nz)
{
 numberOfDimensions = 3;
 cartBase0 = xmin;
 cartBound0= xmax;
 cartBase1 = ymin;
 cartBound1= ymax;
 cartBase2 = zmin;
 cartBound2= zmax;
 
 cartGridNx = Nx;
 cartGridNy = Ny;
 cartGridNz = Nz;
 
 
 gridCoord[0] = new distArray(Nx,Ny,Nz);
 gridCoord[1] = new distArray(Nx,Ny,Nz);
 gridCoord[2] = new distArray(Nx,Ny,Nz);
 
 hxspace = (xmax - xmin)/(Nx-1);
 hyspace = (ymax - ymin)/(Ny-1);
 hzspace = (zmax - zmin)/(Nz-1);

   
 for ( int axis = 0 ; axis < 3 ; axis++)
   for ( int side = 0; side < 2 ; side++)
       nFringe[side][axis]=0;
 
 
 setVectorGrid3D<<<grid,block>>>(gridCoord[0]->array.dVEC,
                                                       gridCoord[1]->array.dVEC,
						       gridCoord[2]->array.dVEC,
                                                                  hxspace, hyspace, hzspace,
                                                                  Nx, Ny, Nz,
								  xmin, xmax, ymin, ymax, zmin, zmax);
 cudaDeviceSynchronize();									

}


// Done conversion
void cartGrid::rebuildCartGrid(int nFringe[2][3])
{
  if (numberOfDimensions==2)
  {
   cartBase0  -= nFringe[0][0]*hxspace;
   cartBound0 += nFringe[1][0]*hxspace;

   cartBase1  -= nFringe[0][1]*hyspace;
   cartBound1 += nFringe[1][1]*hyspace;
   
   cartGridNx += nFringe[0][0] + nFringe[1][0];
   cartGridNy += nFringe[0][1] + nFringe[1][1];
   
   gridCoord[0]->update(*this);
   gridCoord[1]->update(*this);
   
   distArray::temporaryNeedsToBeAllocated = true;
   cudaFree(distArray::dVECTemp);
   distArray::setCudaProperties(cartGridNx, cartGridNy,threadsPerBlockx,threadsPerBlocky);

   setVectorGrid2D<<<grid,block>>>(gridCoord[0]->array.dVEC,gridCoord[1]->array.dVEC,
                                   hxspace, hyspace, cartGridNx, cartGridNy, 
                                   cartBase0, cartBound0, cartBase1, cartBound1);
                                                                        
   
   
  }
  else if ( numberOfDimensions == 3)
  {
   cartBase0  -= nFringe[0][0]*hxspace;
   cartBound0 += nFringe[1][0]*hxspace;

   cartBase1  -= nFringe[0][1]*hyspace;
   cartBound1 += nFringe[1][1]*hyspace;

   cartBase2  -= nFringe[0][2]*hzspace;
   cartBound2 += nFringe[1][2]*hzspace;
 
   cartGridNx += nFringe[0][0] + nFringe[1][0];
   cartGridNy += nFringe[0][1] + nFringe[1][1];
   cartGridNz += nFringe[0][2] + nFringe[1][2];
 
   gridCoord[0]->update(*this);
   gridCoord[1]->update(*this);
   gridCoord[2]->update(*this);
 
   distArray::temporaryNeedsToBeAllocated = true;
   cudaFree(distArray::dVECTemp); 
   cout<<" Resetting original grid "<<endl;
   cout<<threadsPerBlockx<<" "<<threadsPerBlocky<<" "<<threadsPerBlockz<<endl;
   distArray::setCudaProperties(cartGridNx, cartGridNy, cartGridNz, threadsPerBlockx, threadsPerBlocky, 1);
     
   setVectorGrid3D<<<grid,block>>>(gridCoord[0]->array.dVEC, gridCoord[1]->array.dVEC,gridCoord[2]->array.dVEC,
                                   hxspace, hyspace, hzspace,
                                   cartGridNx,cartGridNy, cartGridNz,
				   cartBase0, cartBound0,
				   cartBase1, cartBound1,
				   cartBase2, cartBound2);
   
  
  }

}

void cartGrid::setNumberOfFringePoints(int side,int axis, int numberOfPoints, bool done)
{
  nFringe[side][axis] = numberOfPoints;
  
  if ( done ){
  rebuildCartGrid(nFringe);
  	     }		       
}

void cartGrid::setNumberOfFringePoints(int numberOfPoints)
{
  for ( int axis = 0 ; axis < 3 ; axis++)
    for ( int side = 0 ; side < 2 ; side++)
       nFringe[side][axis]=numberOfPoints;

  rebuildCartGrid(nFringe);
  
}

cartGrid::BCType & cartGrid::setBoundaryConditionType(int side, int axis)
{
    return boundaryConditionType[side][axis] ;
    
}

void cartGrid::getIndexOfFringePoints(int side, int axis, Index & I1, Index & I2, Index & I3)
{
  // Order of faceID is (left 0, right 1, lower 2, upper 3, aft 4, fore 5)
  int faceID = 2*axis+side;

  I1.base = FACE[faceID]->base0;
  I1.bound= FACE[faceID]->bound0;

  I2.base = FACE[faceID]->base1;
  I2.bound= FACE[faceID]->bound1;

  I3.base = FACE[faceID]->base2;
  I3.bound= FACE[faceID]->bound2;

}

void cartGrid::getGenericIndex(int side, int axis, int nPoints, Index & I1, Index & I2, Index & I3)
{
  // Order of faceID is (left 0, right 1, lower 2, upper 3, aft 4, fore 5)
  // completeBoundarySetup **MUST** be called before accessing this function
  int faceID = 2*axis+side;
  FACE[faceID]->getGenericIndex(nPoints, I1, I2, I3);
  
}

void cartGrid::getGenericIndex1(int side, int axis, int nPoints, Index & I1, Index & I2, Index & I3)
{
  // Order of faceID is (left 0, right 1, lower 2, upper 3, aft 4, fore 5)
  // completeBoundarySetup **MUST** be called before accessing this function
  int faceID = 2*axis+side;
  FACE[faceID]->getGenericIndex1(nPoints, I1, I2, I3);
  
}

void cartGrid::getIndexOfBoundaries(int side, int axis, Index & I1, Index & I2, Index & I3)
{
  // Order of faceID is (left 0, right 1, lower 2, upper 3, aft 4, fore 5)
  // completeBoundarySetup **MUST** be called before accessing this function
  int faceID = 2*axis+side;
  FACE[faceID]->getIndexOfBoundaries(I1, I2, I3);
}
void cartGrid::getIndexOfInternalPoints(Index & I1, Index & I2, Index & I3)
{
  I1.base  = 1; I2.base = 1; I3.base = 1;
  I1.bound = cartGridNx-nFringe[0][0]-nFringe[1][0]-2;
  I2.bound = cartGridNy-nFringe[0][1]-nFringe[1][1]-2;
  I3.bound = cartGridNz-nFringe[0][2]-nFringe[1][2]-2;
}

void cartGrid::getIndexOfInternalAndBPoints(Index & I1, Index & I2, Index & I3)
{
  I1.base  = 0; I2.base = 0; I3.base = 0;
  I1.bound = cartGridNx-nFringe[0][0]-nFringe[1][0]-1;
  I2.bound = cartGridNy-nFringe[0][1]-nFringe[1][1]-1;
  I3.bound = cartGridNz-nFringe[0][2]-nFringe[1][2]-1;
}

void cartGrid::getFullIndex(Index & I1, Index & I2, Index & I3)
{
  I1.base  = -nFringe[0][0]; I2.base = -nFringe[0][1]; I3.base = -nFringe[0][2];
  I1.bound = cartGridNx-nFringe[0][0]-1;
  I2.bound = cartGridNy-nFringe[0][1]-1;
  I3.bound = cartGridNz-nFringe[0][2]-1;
}
void cartGrid::completeBoundarySetUp()
{
   // Need to Assert somewhere to have fringe points > 0 where ??
   for ( int i = 0; i < 6 ; i++)
    {
      int axis = i/2;
      int side = i - 2*axis; // or i%2

     switch (i)
     {
        case (0):
          FACE[i] = new __FACE0__(*this);
          break;
        case (1):
          FACE[i] = new __FACE1__(*this); 
          break;
        case (2):
          FACE[i] = new __FACE2__(*this); 
          break;
        case (3):
          FACE[i] = new __FACE3__(*this); 
          break;
        case (4):
          FACE[i] = new __FACE4__(*this); 
          break;
        case (5):
          FACE[i] = new __FACE5__(*this); 
          break;
     }
     switch ( boundaryConditionType[side][axis] )
      {
           case (PERIODIC):
               BDEFN[i] = new __PERIODIC__(i);
               break;
           case (PERIODICNEIGHBOR):
               BDEFN[i] = new __PERIODICNEIGHBOR__(i);
               break; 
           case (SYMMETRY):
               BDEFN[i] = new __SYMMETRY__(i);
               break;
           case (NEIGHBOR):
               BDEFN[i] = new __NEIGHBOR__(i);
               break;
      }
    }
}
cartGrid::~cartGrid()
{
 
  //delete [] gridCoord;

}
 cartGrid::cartGrid()
 {
 
 }

void __FACE0__::PERIODICupdate(cartGrid & cg, vectorGridFunction & Q)
{
      
      Index Ia1, Ib1, Ic1;
      Index Ia2, Ib2, Ic2;
      cg.getGenericIndex1(1,0,cg.nFringe[0][0],Ia1,Ib1,Ic1);
      cg.getIndexOfFringePoints(0,0,Ia2,Ib2,Ic2);
           
      for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++)  
          Q[comp](Ia2,Ib2,Ic2) = Q[comp](Ia1,Ib1,Ic1);
         
           
}
void __FACE0__::PERIODICupdateFromNeighborGrid(cartGrid & cg, vectorGridFunction & Q)
{
   /* cout<<" HIHIHIHIH from rank = "<<distArray::MPI_RANK<<endl;
   if ( distArray::MPI_RANK == 1)      
    {
      
      Index Ia1, Ib1, Ic1;
      cg.getGenericIndex(1,0,cg.nFringe[0][0],Ia1,Ib1,Ic1); // there might be a bug ?? [0][0] corresponds to the same grid !!
      
      int nSendx = Ia1.getLength();
      int nSendy = Ib1.getLength();
      int nSendz = Ic1.getLength();
      cout<<" Face 0 waiting to send from : "<<distArray::MPI_RANK<<endl;
      MPI_Send((void *)&nSendx, 1, MPI_INT, 0, 10, MPI_COMM_WORLD);
      MPI_Send((void *)&nSendy, 1, MPI_INT, 0, 11, MPI_COMM_WORLD);
      MPI_Send((void *)&nSendz, 1, MPI_INT, 0, 12, MPI_COMM_WORLD);
      cout<<" Face 0 sending from : "<<distArray::MPI_RANK<<endl;
      distArray temp(nSendx,nSendy,nSendz);
      Index I1(0,nSendx-1), I2(0,nSendy-1), I3(0,nSendz-1);  

      for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++ )
        {
         temp(I1,I2,I3) = Q[comp](Ia1,Ib1,Ic1);
         temp.pull();
         MPI_Send(temp.VEC, nSendx*nSendy*nSendz, MPI_REAL, 0, 13+comp, MPI_COMM_WORLD);
        } 
    }
 
  if ( distArray::MPI_RANK == 0 )
    {
     cout<<" Face 0 waiting to receive in : "<<distArray::MPI_RANK<<endl;
     int nRecvx, nRecvy, nRecvz;
     MPI_Status stat;
     MPI_Recv((void *)&nRecvx, 1, MPI_INT, 1, 10, MPI_COMM_WORLD, &stat);
     MPI_Recv((void *)&nRecvy, 1, MPI_INT, 1, 11, MPI_COMM_WORLD, &stat);
     MPI_Recv((void *)&nRecvz, 1, MPI_INT, 1, 12, MPI_COMM_WORLD, &stat);
     cout<<" Face 0 receviing in : "<<distArray::MPI_RANK<<endl;
     distArray temp(nRecvx, nRecvy, nRecvz);
     Index Ia2, Ib2, Ic2;
     cg.getIndexOfFringePoints(0,0,Ia2,Ib2,Ic2);
     
     Index I1(0,nRecvx-1), I2(0,nRecvy-1), I3(0,nRecvz-1);  
      for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++ )
       {
        MPI_Recv(temp.VEC, nRecvx*nRecvy*nRecvz, MPI_REAL, 1, 13+comp, MPI_COMM_WORLD, &stat);
        temp.push();
        Q[comp](Ia2,Ib2,Ic2) = temp(I1,I2,I3);        
       }

    }
            
      */     
}

void __FACE0__::updateFromNeighborGrid(cartGrid & cg, vectorGridFunction & Q)
{
  /* cout<<" Face 0 update from neighbor "<<endl;
   if ( distArray::MPI_RANK == 0)      
    {
      Index Ia1, Ib1, Ic1;
      cg.getGenericIndex(1,0,cg.nFringe[0][0],Ia1,Ib1,Ic1); // there might be a bug ?? [0][0] corresponds to the same grid !!
      
      int nSendx = Ia1.getLength();
      int nSendy = Ib1.getLength();
      int nSendz = Ic1.getLength();
      
      MPI_Send(&nSendx, 1, MPI_INT, 1, 20, MPI_COMM_WORLD);
      MPI_Send(&nSendy, 1, MPI_INT, 1, 21, MPI_COMM_WORLD);
      MPI_Send(&nSendz, 1, MPI_INT, 1, 22, MPI_COMM_WORLD);
      distArray temp(nSendx,nSendy,nSendz);
      Index I1(0,nSendx-1), I2(0,nSendy-1), I3(0,nSendz-1);  

      for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++ )
        {
         temp(I1,I2,I3) = Q[comp](Ia1,Ib1,Ic1);
         temp.pull();
         MPI_Send(temp.VEC, nSendx*nSendy*nSendz, MPI_REAL, 1, 23+comp, MPI_COMM_WORLD);
        } 
    }
 
  if ( distArray::MPI_RANK == 1 )
    {
     int nRecvx, nRecvy, nRecvz;
     MPI_Status stat;
     MPI_Recv(&nRecvx, 1, MPI_INT, 0, 20, MPI_COMM_WORLD, &stat);
     MPI_Recv(&nRecvy, 1, MPI_INT, 0, 21, MPI_COMM_WORLD, &stat);
     MPI_Recv(&nRecvz, 1, MPI_INT, 0, 22, MPI_COMM_WORLD, &stat);
     distArray temp(nRecvx, nRecvy, nRecvz);
     Index Ia2, Ib2, Ic2;
     cg.getIndexOfFringePoints(0,0,Ia2,Ib2,Ic2);
     
     Index I1(0,nRecvx-1), I2(0,nRecvy-1), I3(0,nRecvz-1);  
      for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++ )
       {
        MPI_Recv(temp.VEC, nRecvx*nRecvy*nRecvz, MPI_REAL, 0, 23+comp, MPI_COMM_WORLD, &stat);
        temp.push();
        Q[comp](Ia2,Ib2,Ic2) = temp(I1,I2,I3);        
       }

    }
   */         
           
}

void __FACE0__::SYMMETRYupdate(cartGrid & cg, vectorGridFunction & Q)
{
      Index Ia1, Ib1, Ic1;
      Index Ia2, Ib2, Ic2;
      cg.getGenericIndex1(0,0,cg.nFringe[0][0],Ia1,Ib1,Ic1);
      cg.getIndexOfFringePoints(0,0,Ia2,Ib2,Ic2);
      Ia1.flip();       
      for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++)  
          Q[comp](Ia2,Ib2,Ic2) = Q[comp](Ia1,Ib1,Ic1); 
}

void __FACE1__::PERIODICupdate(cartGrid & cg, vectorGridFunction & Q)
{
      Index Ia1, Ib1, Ic1;
      Index Ia2, Ib2, Ic2;
      cg.getGenericIndex1(0,0,cg.nFringe[1][0],Ia1,Ib1,Ic1);
      cg.getIndexOfFringePoints(1,0,Ia2,Ib2,Ic2);
           
      for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++)  
          Q[comp](Ia2,Ib2,Ic2) = Q[comp](Ia1,Ib1,Ic1);     
}

void __FACE1__::PERIODICupdateFromNeighborGrid(cartGrid & cg, vectorGridFunction & Q)
{
  /*cout<<" Face 1 periodic update from neighbor "<<endl;
   if ( distArray::MPI_RANK == 0)      
    {
      Index Ia1, Ib1, Ic1;
      cg.getGenericIndex(0,0,cg.nFringe[1][0],Ia1,Ib1,Ic1);// there might be a bug ?? [1][0] corresponds to the same grid !!
      
      int nSendx = Ia1.getLength();
      int nSendy = Ib1.getLength();
      int nSendz = Ic1.getLength();
      
      MPI_Send(&nSendx, 1, MPI_INT, 1, 30, MPI_COMM_WORLD);
      MPI_Send(&nSendy, 1, MPI_INT, 1, 31, MPI_COMM_WORLD);
      MPI_Send(&nSendz, 1, MPI_INT, 1, 32, MPI_COMM_WORLD);
      distArray temp(nSendx,nSendy,nSendz);
      Index I1(0,nSendx-1), I2(0,nSendy-1), I3(0,nSendz-1);  

      for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++ )
        {
         temp(I1,I2,I3) = Q[comp](Ia1,Ib1,Ic1);
         temp.pull();
         MPI_Send(temp.VEC, nSendx*nSendy*nSendz, MPI_REAL, 1, 33+comp, MPI_COMM_WORLD);
        } 
    }
 
  if ( distArray::MPI_RANK == 1 )
    {
     int nRecvx, nRecvy, nRecvz;
     MPI_Status stat;
     MPI_Recv(&nRecvx, 1, MPI_INT, 0, 30, MPI_COMM_WORLD, &stat);
     MPI_Recv(&nRecvy, 1, MPI_INT, 0, 31, MPI_COMM_WORLD, &stat);
     MPI_Recv(&nRecvz, 1, MPI_INT, 0, 32, MPI_COMM_WORLD, &stat);
     distArray temp(nRecvx, nRecvy, nRecvz);
     Index Ia2, Ib2, Ic2;
     cg.getIndexOfFringePoints(1,0,Ia2,Ib2,Ic2);
     
     Index I1(0,nRecvx-1), I2(0,nRecvy-1), I3(0,nRecvz-1);  
      for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++ )
       {
        MPI_Recv(temp.VEC, nRecvx*nRecvy*nRecvz, MPI_REAL, 0, 33+comp, MPI_COMM_WORLD, &stat);
        temp.push();
        Q[comp](Ia2,Ib2,Ic2) = temp(I1,I2,I3);        
       }

    }
            
     */      
}

void __FACE1__::updateFromNeighborGrid(cartGrid & cg, vectorGridFunction & Q)
{
  /*cout<<" Face 1 update from neighbor "<<endl;
   if ( distArray::MPI_RANK == 1)      
    {
      Index Ia1, Ib1, Ic1;
      cg.getGenericIndex(0,0,cg.nFringe[1][0],Ia1,Ib1,Ic1);// there might be a bug ?? [1][0] corresponds to the same grid !!
      
      int nSendx = Ia1.getLength();
      int nSendy = Ib1.getLength();
      int nSendz = Ic1.getLength();
      
      MPI_Send(&nSendx, 1, MPI_INT, 0, 40, MPI_COMM_WORLD);
      MPI_Send(&nSendy, 1, MPI_INT, 0, 41, MPI_COMM_WORLD);
      MPI_Send(&nSendz, 1, MPI_INT, 0, 42, MPI_COMM_WORLD);
      distArray temp(nSendx,nSendy,nSendz);
      Index I1(0,nSendx-1), I2(0,nSendy-1), I3(0,nSendz-1);  

      for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++ )
        {
         temp(I1,I2,I3) = Q[comp](Ia1,Ib1,Ic1);
         temp.pull();
         MPI_Send(temp.VEC, nSendx*nSendy*nSendz, MPI_REAL, 0, 43+comp, MPI_COMM_WORLD);
        } 
    }
 
  if ( distArray::MPI_RANK == 0 )
    {
     int nRecvx, nRecvy, nRecvz;
     MPI_Status stat;
     MPI_Recv(&nRecvx, 1, MPI_INT, 1, 40, MPI_COMM_WORLD, &stat);
     MPI_Recv(&nRecvy, 1, MPI_INT, 1, 41, MPI_COMM_WORLD, &stat);
     MPI_Recv(&nRecvz, 1, MPI_INT, 1, 42, MPI_COMM_WORLD, &stat);
     distArray temp(nRecvx, nRecvy, nRecvz);
     Index Ia2, Ib2, Ic2;
     cg.getIndexOfFringePoints(1,0,Ia2,Ib2,Ic2);
     
     Index I1(0,nRecvx-1), I2(0,nRecvy-1), I3(0,nRecvz-1);  
      for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++ )
       {
        MPI_Recv(temp.VEC, nRecvx*nRecvy*nRecvz, MPI_REAL, 1, 43+comp, MPI_COMM_WORLD, &stat);
        temp.push();
        Q[comp](Ia2,Ib2,Ic2) = temp(I1,I2,I3);        
       }

    }
   */         
           
}


void __FACE1__::SYMMETRYupdate(cartGrid & cg, vectorGridFunction & Q)
{
      Index Ia1, Ib1, Ic1;
      Index Ia2, Ib2, Ic2;
      cg.getGenericIndex1(1,0,cg.nFringe[1][0],Ia1,Ib1,Ic1);
      cg.getIndexOfFringePoints(1,0,Ia2,Ib2,Ic2);
      Ia1.flip();     
      for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++)  
          Q[comp](Ia2,Ib2,Ic2) = Q[comp](Ia1,Ib1,Ic1);     
}

void __FACE2__::PERIODICupdate(cartGrid & cg, vectorGridFunction & Q)
{
      Index Ia1, Ib1, Ic1;
      Index Ia2, Ib2, Ic2;
      cg.getGenericIndex1(1,1,cg.nFringe[0][1],Ia1,Ib1,Ic1);
      cg.getIndexOfFringePoints(0,1,Ia2,Ib2,Ic2);
           
      for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++)  
          Q[comp](Ia2,Ib2,Ic2) = Q[comp](Ia1,Ib1,Ic1);
}
void __FACE2__::SYMMETRYupdate(cartGrid & cg, vectorGridFunction & Q)
{
      Index Ia1, Ib1, Ic1;
      Index Ia2, Ib2, Ic2;
      cg.getGenericIndex1(0,1,cg.nFringe[0][1],Ia1,Ib1,Ic1);
      cg.getIndexOfFringePoints(0,1,Ia2,Ib2,Ic2);
      Ib1.flip();     
      for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++)  
          Q[comp](Ia2,Ib2,Ic2) = Q[comp](Ia1,Ib1,Ic1);       
}

void __FACE3__::PERIODICupdate(cartGrid & cg, vectorGridFunction & Q)
{
      Index Ia1, Ib1, Ic1;
      Index Ia2, Ib2, Ic2;
      cg.getGenericIndex1(0,1,cg.nFringe[1][1],Ia1,Ib1,Ic1);
      cg.getIndexOfFringePoints(1,1,Ia2,Ib2,Ic2);
           
      for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++)  
          Q[comp](Ia2,Ib2,Ic2) = Q[comp](Ia1,Ib1,Ic1);      
}
void __FACE3__::SYMMETRYupdate(cartGrid & cg, vectorGridFunction & Q)
{
      Index Ia1, Ib1, Ic1;
      Index Ia2, Ib2, Ic2;
      cg.getGenericIndex1(1,1,cg.nFringe[1][1],Ia1,Ib1,Ic1);
      cg.getIndexOfFringePoints(1,1,Ia2,Ib2,Ic2);
      Ib1.flip();     
      for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++)  
          Q[comp](Ia2,Ib2,Ic2) = Q[comp](Ia1,Ib1,Ic1);              
}

void __FACE4__::PERIODICupdate(cartGrid & cg, vectorGridFunction & Q)
{
  Index Ia1, Ib1, Ic1;
  Index Ia2, Ib2, Ic2;
  cg.getGenericIndex1(1,2,cg.nFringe[0][2],Ia1,Ib1,Ic1);
  cg.getIndexOfFringePoints(0,2,Ia2,Ib2,Ic2);
  for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++)      
          Q[comp](Ia2,Ib2,Ic2) = Q[comp](Ia1,Ib1,Ic1);    

    
}
void __FACE4__::SYMMETRYupdate(cartGrid & cg, vectorGridFunction & Q)
{
      Index Ia1, Ib1, Ic1;
      Index Ia2, Ib2, Ic2;
      cg.getGenericIndex1(0,2,cg.nFringe[0][2],Ia1,Ib1,Ic1);
      cg.getIndexOfFringePoints(0,2,Ia2,Ib2,Ic2);
      Ic1.flip();     
      for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++)  
          Q[comp](Ia2,Ib2,Ic2) = Q[comp](Ia1,Ib1,Ic1);          
}

void __FACE5__::PERIODICupdate(cartGrid & cg, vectorGridFunction & Q)
{
  Index Ia1, Ib1, Ic1;
  Index Ia2, Ib2, Ic2;
  cg.getGenericIndex1(0,2,cg.nFringe[1][2],Ia1,Ib1,Ic1);
  cg.getIndexOfFringePoints(1,2,Ia2,Ib2,Ic2);
  for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++)      
          Q[comp](Ia2,Ib2,Ic2) = Q[comp](Ia1,Ib1,Ic1);    
     
}
void __FACE5__::SYMMETRYupdate(cartGrid & cg, vectorGridFunction & Q)
{
      Index Ia1, Ib1, Ic1;
      Index Ia2, Ib2, Ic2;
      cg.getGenericIndex1(1,2,cg.nFringe[1][2],Ia1,Ib1,Ic1);
      cg.getIndexOfFringePoints(1,2,Ia2,Ib2,Ic2);
      Ic1.flip();     
      for ( int comp = 0 ; comp < Q.numberOfComponents ; comp++)  
          Q[comp](Ia2,Ib2,Ic2) = Q[comp](Ia1,Ib1,Ic1);       
}

