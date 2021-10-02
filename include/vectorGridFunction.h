class vectorGridFunction
{
  public:

  vectorGridFunction( cartGrid & cg, int numberOfComponents );
  vectorGridFunction();
  ~vectorGridFunction();
  distArray **Q;
  int numberOfComponents;
  int numberOfDimensions;
  real hixspace, hiyspace, hizspace;
  
  //template < typename E, typename E1, typename E2, typename E3 >
  //inline E const dxc6( E1 const I1, E2 const I2, E3 const I3, int const comp);

  inline distArray & operator[](int comp) { return *(this->Q[comp]);} 
  void display(Index I1,Index I2, Index I3, int comp);
  void pull(int comp);
};


