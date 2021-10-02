class __FACE0__ : public __FACELOOP__
{
  //private:
  //int base0, bound0, base1, bound1, base2, bound2;

  public:
    __FACE0__(cartGrid & cg)
   :__FACELOOP__(cg.cartGridNx, cg.cartGridNy, cg.cartGridNz,
                 cg.cartGridNx-cg.nFringe[0][0]-cg.nFringe[1][0],
                 cg.cartGridNy-cg.nFringe[0][1]-cg.nFringe[1][1],
                 cg.cartGridNz-cg.nFringe[0][2]-cg.nFringe[1][2]) 
    {
     // Only along face 0( side = 0, axis = 0), we set the fringe points base and bound
     base0 = 0-cg.nFringe[0][0];
     bound0= -1;
     
     base1 = 0;
   //  base1 = 0-cg.nFringe[0][1];
     bound1= Ny-1;
    // bound1= Ny-1+cg.nFringe[1][1];  
     
     base2 = 0;
   //  base2 = 0-cg.nFringe[0][2];
     bound2= Nz-1;
   //  bound2= Nz-1+cg.nFringe[1][2];
    
    }
    ~__FACE0__(){}
    void PERIODICupdate(cartGrid & cg, vectorGridFunction & Q);
    void PERIODICupdateFromNeighborGrid(cartGrid & cg,vectorGridFunction & Q);
    void updateFromNeighborGrid(cartGrid & cg, vectorGridFunction & Q);
    void SYMMETRYupdate(cartGrid & cg, vectorGridFunction & Q);
    inline void getGenericIndex(int nPoints, Index & I1, Index & I2, Index & I3)
     {
        I1.base = 0;
        I1.bound= nPoints-1;
        I2.base = base1;
        I2.bound= bound1;
        I3.base = base2;
        I3.bound= bound2;
     }
    inline void getGenericIndex1(int nPoints, Index & I1, Index & I2, Index & I3)
     {
        I1.base = 1;
        I1.bound= nPoints;
        I2.base = base1;
        I2.bound= bound1;
        I3.base = base2;
        I3.bound= bound2;
     }
  
    inline void getIndexOfBoundaries(Index & I1, Index & I2, Index & I3)
     {
        I1.base = 0;
        I1.bound= 0;
        I2.base = base1;
        I2.bound= bound1;
        I3.base = base2;
        I3.bound= bound2;
     }
   
 
};
class __FACE1__ : public __FACELOOP__
{
  //private: 
  //int base0, bound0, base1, bound1, base2, bound2;
  public:
    __FACE1__(cartGrid & cg)
   {
     // Only along face 1(side = 1, axis =0), we set the fringe points base and bound
     base0 = Nx;
     bound0= Nx-1+cg.nFringe[1][0];
  
     base1 = 0;
    // base1 = 0-cg.nFringe[0][1];
     bound1= Ny-1;
    // bound1= Ny-1+cg.nFringe[1][1];  
     
     base2 = 0;
   //  base2 = 0-cg.nFringe[0][2];
     bound2= Nz-1;
   //  bound2= Nz-1+cg.nFringe[1][2];
  
   }
    ~__FACE1__(){}
    void PERIODICupdate(cartGrid & cg, vectorGridFunction & Q);
    void PERIODICupdateFromNeighborGrid(cartGrid & cg,vectorGridFunction & Q);
    void updateFromNeighborGrid(cartGrid & cg, vectorGridFunction & Q);
    void SYMMETRYupdate(cartGrid & cg, vectorGridFunction & Q);

   inline void getGenericIndex(int nPoints, Index & I1, Index & I2, Index & I3)
     {
        I1.base = Nx-nPoints;
        I1.bound= Nx-1;
        I2.base = base1;
        I2.bound= bound1;
        I3.base = base2;
        I3.bound= bound2;
     }
   inline void getGenericIndex1(int nPoints, Index & I1, Index & I2, Index & I3)
     {
        I1.base = Nx-nPoints-1;
        I1.bound= Nx-1-1;
        I2.base = base1;
        I2.bound= bound1;
        I3.base = base2;
        I3.bound= bound2;
     }

   inline void getIndexOfBoundaries(Index & I1, Index & I2, Index & I3)
    {
        I1.base = Nx-1;
        I1.bound= Nx-1;
        I2.base = base1;
        I2.bound= bound1;
        I3.base = base2;
        I3.bound= bound2;
    }
};
class __FACE2__ : public __FACELOOP__
{
  //private: 
  //int base0, bound0, base1, bound1, base2, bound2;

  public:
   __FACE2__(cartGrid & cg)
  {
   // Only along face 2(side = 0, axis =1), we set the fringe points base and bound
     base0 = 0;
    // base0 = 0-cg.nFringe[0][0];
     bound0= Nx-1;
    // bound0= Nx-1+cg.nFringe[1][0];  

     base1 = 0-cg.nFringe[0][1];
     bound1= -1;
     
     base2 = 0;//-cg.nFringe[0][2];
     bound2= Nz-1;//+cg.nFringe[1][2];
  }
  ~__FACE2__(){}
    void PERIODICupdate(cartGrid & cg, vectorGridFunction & Q);
    void PERIODICupdateFromNeighborGrid(cartGrid & cg,vectorGridFunction & Q){}
    void updateFromNeighborGrid(cartGrid & cg, vectorGridFunction & Q){}
    void SYMMETRYupdate(cartGrid & cg, vectorGridFunction & Q);
    inline void getGenericIndex(int nPoints, Index & I1, Index & I2, Index & I3)
     {
        I1.base = base0;
        I1.bound= bound0;
        I2.base = 0;
        I2.bound= nPoints-1;
        I3.base = base2;
        I3.bound= bound2;
     }
    inline void getGenericIndex1(int nPoints, Index & I1, Index & I2, Index & I3)
     {
        I1.base = base0;
        I1.bound= bound0;
        I2.base = 1;
        I2.bound= nPoints;
        I3.base = base2;
        I3.bound= bound2;
     }
 
    inline void getIndexOfBoundaries(Index & I1, Index & I2, Index & I3)
     {   
        I1.base = base0;
        I1.bound= bound0;
        I2.base = 0;
        I2.bound= 0;
        I3.base = base2;
        I3.bound= bound2;
     }
};
class __FACE3__ : public __FACELOOP__
{

  //private: 
  //int base0, bound0, base1, bound1, base2, bound2;

  public:
  __FACE3__(cartGrid & cg)
  {
  // Only along face 3(side = 1, axis =1), we set the fringe points base and bound
     base0 = 0;//-cg.nFringe[0][0];
     bound0= Nx-1;//+cg.nFringe[0][1];  

     base1 = Ny;
     bound1= Ny-1+cg.nFringe[1][1];
  
     base2 = 0;//-cg.nFringe[0][2];
     bound2= Nz-1;//+cg.nFringe[1][2]; 
  }
 ~__FACE3__(){} 
    void PERIODICupdate(cartGrid & cg, vectorGridFunction & Q);
    void PERIODICupdateFromNeighborGrid(cartGrid & cg,vectorGridFunction & Q){}
    void updateFromNeighborGrid(cartGrid & cg, vectorGridFunction & Q){}
    void SYMMETRYupdate(cartGrid & cg, vectorGridFunction & Q);
    inline void getGenericIndex(int nPoints, Index & I1, Index & I2, Index & I3)
     {
        I1.base = base0;
        I1.bound= bound0;
        I2.base = Ny-nPoints;
        I2.bound= Ny-1;
        I3.base = base2;
        I3.bound= bound2;
     }
    inline void getGenericIndex1(int nPoints, Index & I1, Index & I2, Index & I3)
     {
        I1.base = base0;
        I1.bound= bound0;
        I2.base = Ny-nPoints-1;
        I2.bound= Ny-1-1;
        I3.base = base2;
        I3.bound= bound2;
     }

    inline void getIndexOfBoundaries(Index & I1, Index & I2, Index & I3)
     {
        I1.base = base0;
        I1.bound= bound0;
        I2.base = Ny-1;
        I2.bound= Ny-1;
        I3.base = base2;
        I3.bound= bound2;
     }
};
class __FACE4__ : public __FACELOOP__
{

  //private: 
  //int base0, bound0, base1, bound1, base2, bound2;

  public:
   __FACE4__(cartGrid & cg)
    {
      // Only along face 4(side = 0, axis =2), we set the fringe points base and bound
     base0 = 0;//-cg.nFringe[0][0];
     bound0= Nx-1;//+cg.nFringe[1][0];  
     
     base1 = 0;//-cg.nFringe[0][1];
     bound1= Ny-1;//+cg.nFringe[1][1];
     
     base2 = 0-cg.nFringe[0][2];
     bound2= -1; 

   }
  ~__FACE4__(){}
    void PERIODICupdate(cartGrid & cg, vectorGridFunction & Q);
    void PERIODICupdateFromNeighborGrid(cartGrid & cg,vectorGridFunction & Q){}
    void updateFromNeighborGrid(cartGrid & cg, vectorGridFunction & Q){}
    void SYMMETRYupdate(cartGrid & cg, vectorGridFunction & Q);
    inline void getGenericIndex(int nPoints, Index & I1, Index & I2, Index & I3)
     {
        I1.base = base0;
        I1.bound= bound0;
        I2.base = base1;
        I2.bound= bound1;
        I3.base = 0;
        I3.bound= nPoints-1;
     }
     inline void getGenericIndex1(int nPoints, Index & I1, Index & I2, Index & I3)
     {
        I1.base = base0;
        I1.bound= bound0;
        I2.base = base1;
        I2.bound= bound1;
        I3.base = 1;
        I3.bound= nPoints;
     }

     inline void getIndexOfBoundaries(Index & I1, Index & I2, Index & I3)
     {
        I1.base = base0;
        I1.bound= bound0;
        I2.base = base1;
        I2.bound= bound1;
        I3.base = 0;
        I3.bound= 0;
     }

};
class __FACE5__ : public __FACELOOP__
{
  //private:
  //  int base0, bound0, base1, bound1, base2, bound2;

  public:
  __FACE5__(cartGrid & cg)
   {
    // Only along face 5(side = 1, axis =2), we set the fringe points base and bound
     base0 = 0;//-cg.nFringe[0][0];
     bound0= Nx-1;//+cg.nFringe[0][1];  

     base1 = 0;//-cg.nFringe[0][1];
     bound1= Ny-1;//+cg.nFringe[1][1]; 
 
     base2 = Nz;
     bound2= Nz-1+cg.nFringe[1][2];
  
   }
 ~__FACE5__(){}
    void PERIODICupdate(cartGrid & cg, vectorGridFunction & Q);
    void PERIODICupdateFromNeighborGrid(cartGrid & cg,vectorGridFunction & Q){}
    void updateFromNeighborGrid(cartGrid & cg, vectorGridFunction & Q){}
    void SYMMETRYupdate(cartGrid & cg, vectorGridFunction & Q);
    inline void getGenericIndex(int nPoints, Index & I1, Index & I2, Index & I3)
     {
        I1.base = base0;
        I1.bound= bound0;
        I2.base = base1;
        I2.bound= bound1;
        I3.base = Nz-nPoints;
        I3.bound= Nz-1;
     }
     inline void getGenericIndex1(int nPoints, Index & I1, Index & I2, Index & I3)
     {
        I1.base = base0;
        I1.bound= bound0;
        I2.base = base1;
        I2.bound= bound1;
        I3.base = Nz-nPoints-1;
        I3.bound= Nz-1-1;
     }

     inline void getIndexOfBoundaries(Index & I1, Index & I2, Index & I3)
     {
        I1.base = base0;
        I1.bound= bound0;
        I2.base = base1;
        I2.bound= bound1;
        I3.base = Nz-1;
        I3.bound= Nz-1;
     }

};


class __PERIODIC__ : public __BoundaryTypes__
// This class manages the periodic boundary condition
{
  public:
  int faceID;
   __PERIODIC__(int face){faceID=face;}
  ~__PERIODIC__(){}
  
  
  inline void genericBoundaryUpdate(cartGrid & cg, vectorGridFunction & Q)
   {
     // Depending on the type of FACE (FACE0, FACE1..etc) different functions are executed 
     //cout<<"periodic ";
     cg.FACE[faceID]->PERIODICupdate(cg,Q);
   }   
};

class __PERIODICNEIGHBOR__ : public __BoundaryTypes__
// This class manages the periodic boundary condition
{
  public:
  int faceID;
   __PERIODICNEIGHBOR__(int face){faceID=face;}
  ~__PERIODICNEIGHBOR__(){}
  
  
  inline void genericBoundaryUpdate(cartGrid & cg, vectorGridFunction & Q)
   {
     // Depending on the type of FACE (FACE0, FACE1..etc) different functions are executed 
     cg.FACE[faceID]->PERIODICupdateFromNeighborGrid(cg,Q);
   }   
};

class __NEIGHBOR__ : public __BoundaryTypes__
// This class manages the periodic boundary condition
{
  public:
  int faceID;
   __NEIGHBOR__(int face){faceID=face;}
  ~__NEIGHBOR__(){}
  
  
  inline void genericBoundaryUpdate(cartGrid & cg, vectorGridFunction & Q)
   {
     // Depending on the type of FACE (FACE0, FACE1..etc) different functions are executed 
     cg.FACE[faceID]->updateFromNeighborGrid(cg,Q);
   }   
};



class __SYMMETRY__ : public __BoundaryTypes__
// This class manages the symmetry boundary condition
{
  public:
  int faceID;
   __SYMMETRY__(int face){faceID=face;}
  ~__SYMMETRY__(){}
  
  inline void genericBoundaryUpdate(cartGrid & cg, vectorGridFunction & Q)
    { 
     //cout<<"symmetry "; 
     cg.FACE[faceID]->SYMMETRYupdate(cg,Q);
    }
};

