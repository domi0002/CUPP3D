//- *** The usual stuff ***


vectorGridFunction::vectorGridFunction( cartGrid & cg, int _numberOfComponents )
 {
    numberOfComponents = _numberOfComponents;
    numberOfDimensions = cg.numberOfDimensions;
       
    Q = new distArray* [numberOfComponents];


    if ( cg.numberOfDimensions == 1)
    {
      for ( int i = 0 ; i < numberOfComponents ; i++)
        Q[i] = new distArray(cg);

        hixspace = 1/cg.hxspace;

    }
    
    if ( cg.numberOfDimensions == 2)
    {
      for ( int i = 0 ; i < numberOfComponents ; i++)
        Q[i] = new distArray(cg);
	
	hixspace = 1/cg.hxspace;
	hiyspace = 1/cg.hyspace;
 
    }
    
    else if (cg.numberOfDimensions == 3)
    {
      
      for ( int i = 0 ; i < numberOfComponents ; i++)
        Q[i] = new distArray(cg);
	
	hixspace = 1/cg.hxspace;
	hiyspace = 1/cg.hyspace;
	hizspace = 1/cg.hzspace; 
    
    }
    
 }


void vectorGridFunction::pull(int comp)
{
   distArray & QT = *Q[comp];
   cudaMemcpy(QT.VEC, QT.array.dVEC, QT.size*sizeof(float), cudaMemcpyDeviceToHost) ; 
 
}

void vectorGridFunction::display(Index I1,Index I2, Index I3, int comp)
{
  distArray & QT = *Q[comp];
  QT.display();     
}


void div2( vectorGridFunction & QU,
                               vectorGridFunction & E,
                               vectorGridFunction & F,
                               vectorGridFunction & G,  
                                      distArray & sigmax,
                                      distArray & sigmay,
                                      distArray & sigmaz,
                                      Index i, Index j, Index k)
{
 
  for ( int comp = 0 ; comp < E.numberOfComponents ; comp++)
     {
       distArray & fx = E[comp];
       distArray & fy = F[comp];
       distArray & fz = G[comp];
       distArray & q  = QU[comp];

       // Central Diff + Third order Dissipation (along x)
       fx(i,j,k) = 0.5*E.hixspace*( fx(i+1,j,k) - fx(i-1,j,k) )- E.hixspace*sigmax(i,j,k)*( 1.0*q(i+2,j,k) - 3.0*q(i+1,j,k) + 3.0*q(i,  j,k) - q(i-1,j,k) );
       fx(i,j,k) = fx(i,j,k) + E.hixspace*sigmax(i-1,j,k)*( 1.0*q(i+1,j,k) - 3.0*q(i  ,j,k) + 3.0*q(i-1,j,k) - q(i-2,j,k) );


       // Central Diff + Third order Dissipation (along y)
       fx(i,j,k) = fx(i,j,k) + 0.5*E.hiyspace*( fy(i,j+1,k) - fy(i,j-1,k) )- E.hiyspace*sigmay(i,j,k)*( 1.0*q(i,j+2,k) - 3.0*q(i,j+1,k) + 3.0*q(i,j  ,k) - q(i,j-1,k) );
       fx(i,j,k) = fx(i,j,k) + E.hiyspace*sigmay(i,j-1,k)*( 1.0*q(i,j+1,k) - 3.0*q(i  ,j,k) + 3.0*q(i,j-1,k) - q(i,j-2,k) );

   
       // Central Diff + Third order Dissipation (along z)
       fx(i,j,k) = fx(i,j,k) + 0.5*E.hizspace*( fz(i,j,k+1) - fz(i,j,k-1) )- E.hizspace*sigmaz(i,j,k)*( 1.0*q(i,j,k+2) - 3.0*q(i,j,k+1) + 3.0*q(i,j,k  ) - q(i,j,k-1) );
       fx(i,j,k) = fx(i,j,k) + E.hizspace*sigmaz(i,j,k-1)*( 1.0*q(i,j,k+1) - 3.0*q(i  ,j,k) + 3.0*q(i,j,k-1) - q(i,j,k-2) );
     }


}



void div6( vectorGridFunction & QU,
                               vectorGridFunction & E,
                               vectorGridFunction & F,
                               vectorGridFunction & G,  
                                      distArray & sigmax,
                                      distArray & sigmay,
                                      distArray & sigmaz,
                                      Index I1, Index I2, Index I3)
{

    // - Please fill me up

}

 vectorGridFunction::vectorGridFunction(){}
 vectorGridFunction::~vectorGridFunction(){}


