
// Base class for all boundary types
class __BoundaryTypes__
{
  public:
  virtual inline void genericBoundaryUpdate(cartGrid & cg, vectorGridFunction & Q){ }

};

// Each face represents a DataType. We use this methodology with templates to avoid -if loops-
class __FACELOOP__
{
  public:
  // These are the base and bounds for Fringe points..
  int base0, bound0, base1, bound1, base2, bound2;
  static int Nx, Ny, Nz;    // Grid points corresponding to original grid
  static int MNx, MNy, MNz; // Grid points corresponding to original grid + fringe points
  
  __FACELOOP__(int _MNx, int _MNy, int _MNz, int _Nx, int _Ny, int _Nz )
   {Nx=_Nx; Ny=_Ny; Nz=_Nz; MNx=_MNx; MNy=_MNy; MNz=_MNz;	};
  __FACELOOP__(){};
 ~__FACELOOP__(){};
    virtual void PERIODICupdate( cartGrid & cg, vectorGridFunction & Q)=0;
    virtual void SYMMETRYupdate( cartGrid & cg, vectorGridFunction & Q)=0;

    virtual inline void getGenericIndex( int nPoints, Index & I1, Index & I2, Index & I3){}
    virtual inline void getGenericIndex1( int nPoints, Index & I1, Index & I2, Index & I3){}
    virtual inline void getIndexOfBoundaries(Index & I1, Index & I2, Index & I3){}

    virtual void PERIODICupdateFromNeighborGrid( cartGrid & cg, vectorGridFunction & Q)=0;
    virtual void updateFromNeighborGrid( cartGrid & cg, vectorGridFunction & Q)=0;
}; 



// The Cartesian Grid Class holds data about the grid
class cartGrid
{
   public:
   
   real cartBase0, cartBound0, cartBase1, cartBound1, cartBase2, cartBound2;
   real hxspace, hyspace, hzspace;
   
   cartGrid();
   cartGrid( real base0, real bound0, int Nx); // 1D
   cartGrid( real base0, real bound0, int Nx, real base1, real bound1, int Ny); // 2D
   cartGrid( real base0, real bound0, int Nx, 
             real base1, real bound1, int Ny,
	     real base2, real bound2, int Nz); // 2D
   ~cartGrid();
   int numberOfDimensions;
   int cartGridNx, cartGridNy, cartGridNz;
   int cartGridN[3];
   
   int nFringe[2][3]; // 2 sides per axis and a total of 3 axis
   void getIndexOfFringePoints(int side, int axis, Index & I1, Index & I2, Index & I3);
   void getGenericIndex(int side, int axis, int nPoints, Index & I1, Index & I2, Index & I3);
   void getGenericIndex1(int side, int axis, int nPoints, Index & I1, Index & I2, Index & I3);
   void getIndexOfInternalPoints(Index & I1, Index & I2, Index & I3);
   void getIndexOfInternalAndBPoints(Index & I1, Index & I2, Index & I3);
   void getIndexOfBoundaries(int side, int axis, Index & I1, Index & I2, Index & I3);
   void getFullIndex(Index & I1, Index & I2, Index & I3);
   void setNumberOfFringePoints(int side,int axis, int numberOfPoints, bool done);
   void setNumberOfFringePoints(int numberOfPoints);
   void setGridCoord();
   void rebuildCartGrid(int nFringe[2][3]);
   void setBaseAndBound(real base0, real bound0);
   void setBaseAndBound(real base0, real bound0, real base1, real bound1);
   void setBaseAndBound(real base0, real bound0, real base1, real bound1, real base2, real bound2);
   distArray* gridCoord[3]; // 3 is for a maximum 3D grid to hold x, y and z coordinates

   inline distArray & operator[](int coord) { return *(this->gridCoord[coord]) ;} 

   enum BCType
    {
      PERIODIC,
      PERIODICNEIGHBOR,
      SYMMETRY,
      NEIGHBOR,
      DIRICHLET,
      NEUMANN
    };

   BCType & setBoundaryConditionType(int side, int axis);
   BCType boundaryConditionType[2][3]; // 2 sides and 3 axes
   void completeBoundarySetUp();
   
   __BoundaryTypes__ *BDEFN[6];
   __FACELOOP__ *FACE[6];
   
};


