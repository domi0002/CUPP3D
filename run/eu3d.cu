# include "CU++.h"
# include "eu3d.h"

/* ---------------------------------------------------------
   3D Euler Solver on Cartesian Grids using GPU-CU++ Framework
   Dominic Chandar
   University of Wyoming (2010)
   Queen's University Belfast (2021)


   The CU++ framework enables the user to write simple finite-difference
   expressions on the GPU without having to write a kernel.

   ---------------------------------------------------------
*/

int main(int argc, char* argv[])
 {
    int Nx      = atoi(argv[1]);//406 ;
    int Ny      = atoi(argv[2]);//200 ;
    int Nz      = atoi(argv[3]);//150 ;
    real dt     = atof(argv[4]);
    int nSteps  = atoi(argv[5]);

    distArray::setCudaProperties(Nx,Ny,Nz,64,2,2);
    
    // Create a cartesian Grid in 3D
    cartGrid cg(0,20.0,Nx,0,20.0,Ny,0,1.0,Nz);
  
    // Set the number of Fringe Points on all sides
    // After setting this to 2, gridNx = Nx+4, gridNy = Ny+4, gridNz = Nz+4 ( Internally )
    // But Nx, Ny, Nz remains unchanged in the current context
     cg.setNumberOfFringePoints(3);
    
    cout<<"-------------------------"<<endl;
    cout<<" CUDA Device Properties "<<endl;
    cout<<"-------------------------"<<endl;
    cout<<threadsPerBlockx<<" "<<threadsPerBlocky<<endl;

    // Assign boundary condition types for all boundaries
    cg.setBoundaryConditionType(0,0) = cartGrid::PERIODIC ; // Left boundary
    cg.setBoundaryConditionType(1,0) = cartGrid::PERIODIC ; // Right boundary
    cg.setBoundaryConditionType(0,1) = cartGrid::SYMMETRY ; // Lower boundary
    cg.setBoundaryConditionType(1,1) = cartGrid::SYMMETRY ; // Upper boundary
    cg.setBoundaryConditionType(0,2) = cartGrid::PERIODIC ; // Aft boundary
    cg.setBoundaryConditionType(1,2) = cartGrid::PERIODIC ; // Fore boundary
    cg.completeBoundarySetUp(); // !!!! Absolutely Necessary !!!!
     
    // Create a function to hold the different components of Euler Equation 
    // The 5th component of Q will hold pressure ( component starts at 0 )
    vectorGridFunction Q(cg,6), QN(cg,6), E(cg,5), F(cg,5), G(cg,5);
    vectorGridFunction E0(cg,5);
    distArray sigmaX(cg), sigmaY(cg), sigmaZ(cg);

    // Generate Initial Condition ( Say a vortex situated at an arbitrary location )
    clock_t tinit, tend;
    tinit = clock();
    setInitialCondition(cg,Q);  
    tend  = clock()-tinit;
    cout<<" Time to initialize the domain = "<<(double)tend/CLOCKS_PER_SEC<<endl ;       

    // Set Boundary Conditions
    updateBoundaries(cg,Q);

    
    clock_t t1, t2;
     
    Q[0].pull();
    Q[0].outputArrayToFile("density0.dat");
   
    t1 = clock();
    
    // Iterate 
    real solTime=0;
    for ( int steps = 0 ; steps < nSteps; steps++)
    { 
	solTime = (steps+1)*dt;
        printf("Time = %f\n",solTime);

	// Third order Runge-Kutta Explicit Time Stepping
        runStep(cg,Q,QN,E,F,G,sigmaX,sigmaY,sigmaZ,dt,E0);

	// First order Euler Explicit Time Stepping
	//runStepEuler(cg,Q,QN,E,F,G,sigmaX,sigmaY,sigmaZ,dt);
    }
     
    t2= clock()-t1;
    printf("wall-clock time = %lf  s\n", (double)t2/CLOCKS_PER_SEC);


    // Write Numerical solution to file
    Q[0].pull();
    Q[0].outputArrayToFile("densityFinal.dat");  


    distArray::cleanUp();
    return 0;
}

void runStep( cartGrid & cg, vectorGridFunction & Q, vectorGridFunction & QN,
             vectorGridFunction & E, vectorGridFunction & F, vectorGridFunction & G, 
             distArray & sigmaX, distArray & sigmaY, distArray & sigmaZ, real dt, vectorGridFunction & E0)
/*
    III Order RK Time Stepping
    Wray, A. A., "Very Low Storage Time Advancement Schemes," Internal Report,
    NASA Ames Research Center, Moffett Field, C. A. (1986).
*/
{
  
  Index Ix, Iy, Iz;
  cg.getIndexOfInternalAndBPoints(Ix,Iy,Iz); 
  double f1 = 0.25;
  double f2 = 8.0/15.0;
  double f3 = 5.0/12.0;
  double f4 = 0.75;
  clock_t t1,t2,t3,t4;
  
    computeConvectiveFluxes(cg, Q, E, F, G); // All Grid points 
    computeSpecRad(cg, Q, sigmaX, sigmaY, sigmaZ);
    computeRHS(cg,Q,E,F,G,sigmaX,sigmaY,sigmaZ); // Only internal + boundary points
   
    for ( int comp=0; comp < E.numberOfComponents ; comp++)
      E0[comp](Ix,Iy,Iz) = E[comp](Ix,Iy,Iz);


    //Stage 1
    for ( int comp = 0 ; comp < E.numberOfComponents ; comp++)
    {
     	QN[comp](Ix,Iy,Iz) = Q[comp](Ix,Iy,Iz) -f2*dt*E[comp](Ix,Iy,Iz);
    }
    updatePressure(QN,Ix,Iy,Iz);
    updateBoundaries(cg,QN); // Fix fringe points for QN

    // Stage 2
    computeConvectiveFluxes(cg, QN, E, F, G); // All Grid points
    computeSpecRad(cg, QN, sigmaX, sigmaY, sigmaZ);
    computeRHS(cg,QN,E,F,G,sigmaX,sigmaY,sigmaZ); // Only internal + boundary points
     
    for ( int comp = 0 ; comp < E.numberOfComponents ; comp++)
    {
     	QN[comp](Ix,Iy,Iz) = Q[comp](Ix,Iy,Iz) -f1*dt*E0[comp](Ix,Iy,Iz)   -f3*dt*E[comp](Ix,Iy,Iz);
    }
     updatePressure(QN,Ix,Iy,Iz);
     updateBoundaries(cg,QN);

     // Stage 3
     computeConvectiveFluxes(cg, QN, E, F, G); // All Grid points 
     computeSpecRad(cg, QN, sigmaX, sigmaY, sigmaZ);
     computeRHS(cg,QN,E,F,G,sigmaX,sigmaY,sigmaZ); // Only internal + boundary points
     
     for ( int comp = 0 ; comp < E.numberOfComponents ; comp++)
     {
      Q[comp](Ix,Iy,Iz) = Q[comp](Ix,Iy,Iz) -f1*dt*E0[comp](Ix,Iy,Iz) -f4*dt*E[comp](Ix,Iy,Iz);
     }
     updatePressure(Q,Ix,Iy,Iz);
     updateBoundaries(cg,Q);
     
   
}

void runStepEuler( cartGrid & cg, vectorGridFunction & Q, vectorGridFunction & QN,
             vectorGridFunction & E, vectorGridFunction & F, vectorGridFunction & G, 
             distArray & sigmaX, distArray & sigmaY, distArray & sigmaZ, real dt)
{
  
    Index Ix, Iy, Iz;
    cg.getIndexOfInternalAndBPoints(Ix,Iy,Iz); 
   
    computeConvectiveFluxes(cg, Q, E, F, G); // All Grid points 
    computeSpecRad(cg, Q, sigmaX, sigmaY, sigmaZ);
    computeRHS(cg,Q,E,F,G,sigmaX,sigmaY,sigmaZ); // Only internal + boundary points
   
    for ( int comp = 0 ; comp < E.numberOfComponents ; comp++)
    {
     	Q[comp](Ix,Iy,Iz) = Q[comp](Ix,Iy,Iz) -dt*E[comp](Ix,Iy,Iz);
    }
   
    updatePressure(Q,Ix,Iy,Iz);
    updateBoundaries(cg,Q); // Fix fringe points for Q

   
}




void updatePressure(vectorGridFunction & Q, Index & Ix, Index & Iy, Index & Iz)
{
  Q[5](Ix,Iy,Iz) = GM1*( Q[4](Ix,Iy,Iz) - 0.5*( Q[1](Ix,Iy,Iz)*Q[1](Ix,Iy,Iz) 
                                               +Q[2](Ix,Iy,Iz)*Q[2](Ix,Iy,Iz) 
                                               +Q[3](Ix,Iy,Iz)*Q[3](Ix,Iy,Iz))/Q[0](Ix,Iy,Iz) ); 
}
void computeRHS(cartGrid & cg, vectorGridFunction & Q,
			                    vectorGridFunction & E,
				            vectorGridFunction & F,
                                            vectorGridFunction & G,
                                            distArray & sigmaX,
					    distArray & sigmaY,
                                            distArray & sigmaZ) 
{
    // RHS is computed for Internal + Boundary points(using fringe data)
    // And returned in 'E'
    Index Ix, Iy, Iz;
    cg.getIndexOfInternalAndBPoints(Ix,Iy,Iz); 
    div2(Q,E,F,G,sigmaX,sigmaY,sigmaZ,Ix,Iy,Iz);

    // Ways to write higher-order implementations ?
    //div4(Q,E,F,G,sigmaX,sigmaY,sigmaZ,Ix,Iy,Iz);
    //div6(Q,E,F,G,sigmaX,sigmaY,sigmaZ,Ix,Iy,Iz);
    //div8(Q,E,F,G,sigmaX,sigmaY,sigmaZ,Ix,Iy,Iz);

}


void updateBoundaries( cartGrid & cg, vectorGridFunction & Q )
// Smart boundary update without if/else or switch case statement
{
   for ( int i = 0 ; i < 6 ; i++ ) // Total Six faces to loop 
    {
       cg.BDEFN[i]->genericBoundaryUpdate(cg,Q);
       
  }
}


void computeSpecRad(cartGrid & cg, vectorGridFunction & Q, distArray & sigmaX, distArray & sigmaY, distArray & sigmaZ)
{
   int Nx = cg.cartGridNx-cg.nFringe[0][0]-cg.nFringe[1][0];
   int Ny = cg.cartGridNy-cg.nFringe[0][1]-cg.nFringe[1][1];
   int Nz = cg.cartGridNz-cg.nFringe[0][2]-cg.nFringe[1][2];

   Index S1(-1,Nx), S2(-1,Ny), S3(-1,Nz);
   Index I1, I2, I3;
   cg.getIndexOfInternalAndBPoints(I1,I2,I3);

   // spec radius sigma = abs(U+C)
   // U: Flow velocity, C: Speed of sound =sqrt(gamma P/rho)
   sigmaX(S1,I2,I3) = ABS(Q[1](S1,I2,I3)/Q[0](S1,I2,I3)) + SQRT(GM*Q[5](S1,I2,I3)/Q[0](S1,I2,I3));
   sigmaY(I1,S2,I3) = ABS(Q[2](I1,S2,I3)/Q[0](I1,S2,I3)) + SQRT(GM*Q[5](I1,S2,I3)/Q[0](I1,S2,I3));
   sigmaZ(I1,I2,S3) = ABS(Q[3](I1,I2,S3)/Q[0](I1,I2,S3)) + SQRT(GM*Q[5](I1,I2,S3)/Q[0](I1,I2,S3));   

   Index If1(-1,Nx-1), If2(-1,Ny-1), If3(-1,Nz-1);
   Index Ifp1(0,Nx), Ifp2(0,Ny), Ifp3(0,Nz);
 
   // Spec radius computed on each face as an avergage of the neighbouring cells
   // Scaling factor 0.04 is hardcoded here. Analogous to eps in Pulliam/Jameson's implementation
   // eps can be optimized for different orders of accuracy.
  
   sigmaX(If1,I2,I3) = -0.04*0.5*( sigmaX(Ifp1,I2,I3) + sigmaX(If1,I2,I3) );
   sigmaY(I1,If2,I3) = -0.04*0.5*( sigmaY(I1,Ifp2,I3) + sigmaY(I1,If2,I3) );
   sigmaZ(I1,I2,If3) = -0.04*0.5*( sigmaZ(I1,I2,Ifp3) + sigmaZ(I1,I2,If3) );


}


void computeConvectiveFluxes(cartGrid & cg, vectorGridFunction & Q,
			                    vectorGridFunction & E,
				            vectorGridFunction & F,
                                            vectorGridFunction & G)
{
  
  Index Ix, Iy, Iz;
  cg.getFullIndex(Ix,Iy,Iz);
 

 // Note how there are no GPU kernels here! - However these are executed on the GPU.
 
 // X-Flux
  E[0](Ix,Iy,Iz) = Q[1](Ix,Iy,Iz);
  E[1](Ix,Iy,Iz) = Q[5](Ix,Iy,Iz) + Q[1](Ix,Iy,Iz)*Q[1](Ix,Iy,Iz)/Q[0](Ix,Iy,Iz);
  E[2](Ix,Iy,Iz) = Q[1](Ix,Iy,Iz)*Q[2](Ix,Iy,Iz)/Q[0](Ix,Iy,Iz)	;
  E[3](Ix,Iy,Iz) = Q[1](Ix,Iy,Iz)*Q[3](Ix,Iy,Iz)/Q[0](Ix,Iy,Iz)	;
  E[4](Ix,Iy,Iz) = (Q[4](Ix,Iy,Iz) + Q[5](Ix,Iy,Iz))*Q[1](Ix,Iy,Iz)/Q[0](Ix,Iy,Iz);

  // Y-Flux
  F[0](Ix,Iy,Iz) = Q[2](Ix,Iy,Iz);
  F[1](Ix,Iy,Iz) = E[2](Ix,Iy,Iz);
  F[2](Ix,Iy,Iz) = Q[5](Ix,Iy,Iz) + Q[2](Ix,Iy,Iz)*Q[2](Ix,Iy,Iz)/Q[0](Ix,Iy,Iz);
  F[3](Ix,Iy,Iz) = Q[2](Ix,Iy,Iz)*Q[3](Ix,Iy,Iz)/Q[0](Ix,Iy,Iz)	;			
  F[4](Ix,Iy,Iz) = (Q[4](Ix,Iy,Iz) + Q[5](Ix,Iy,Iz))*Q[2](Ix,Iy,Iz)/Q[0](Ix,Iy,Iz);

  // Z-Flux
  G[0](Ix,Iy,Iz) = Q[3](Ix,Iy,Iz);
  G[1](Ix,Iy,Iz) = E[3](Ix,Iy,Iz);
  G[2](Ix,Iy,Iz) = F[3](Ix,Iy,Iz);
  G[3](Ix,Iy,Iz) = Q[5](Ix,Iy,Iz) + Q[3](Ix,Iy,Iz)*Q[3](Ix,Iy,Iz)/Q[0](Ix,Iy,Iz);
  G[4](Ix,Iy,Iz) = (Q[4](Ix,Iy,Iz) + Q[5](Ix,Iy,Iz))*Q[3](Ix,Iy,Iz)/Q[0](Ix,Iy,Iz);

  //G[4].pull();
  //G[4].display();
}

void setInitialCondition(cartGrid & cg, vectorGridFunction & Q)
{
   double x0 = 10.0;
   double y0 = 10.0;	
   double sg = 1.0;
   double st = 1.0;
   double af = st/(2*M_PI);
   double bf = -0.5*GM1IG*af*af/st;
   double rinf = 1.0;
   double uinf = 0.5;
   double vinf = 0.0;
   double pinf = GMI; 
   double tinf = pinf/rinf;
   double sinf = pinf/pow(rinf,GM);
 

  // use only internal and boundary points for initialization
   Index Ix, Iy, Iz;
   cg.getIndexOfInternalAndBPoints(Ix,Iy,Iz); 
  // cg.getFullIndex(Ix,Iy,Iz);
   Ix.display();
   Iy.display();
   Iz.display();
   distArray xBar(cg), yBar(cg), rsq(cg), ee(cg), u(cg), v(cg), w(cg);
   distArray t(cg), p(cg), rho(cg);
   distArray UB(cg);

   xBar(Ix,Iy,Iz) = cg[0](Ix,Iy,Iz)- x0;
   yBar(Ix,Iy,Iz) = cg[1](Ix,Iy,Iz)- y0;
    rsq(Ix,Iy,Iz) = xBar(Ix,Iy,Iz)*xBar(Ix,Iy,Iz) + yBar(Ix,Iy,Iz)*yBar(Ix,Iy,Iz) ;
     ee(Ix,Iy,Iz) = EXP(0.5*(1.0-sg*rsq(Ix,Iy,Iz)));
      u(Ix,Iy,Iz) = u(Ix,Iy,Iz) - af*ee(Ix,Iy,Iz)*yBar(Ix,Iy,Iz);
      v(Ix,Iy,Iz) = v(Ix,Iy,Iz) + af*ee(Ix,Iy,Iz)*xBar(Ix,Iy,Iz);
      t(Ix,Iy,Iz) = t(Ix,Iy,Iz) + bf*ee(Ix,Iy,Iz)*ee(Ix,Iy,Iz);
      u(Ix,Iy,Iz) = uinf + u(Ix,Iy,Iz);
      v(Ix,Iy,Iz) = vinf + v(Ix,Iy,Iz); 
      w(Ix,Iy,Iz) = 0.0;
     // u.pull(); u.display();  
      
     UB(Ix,Iy,Iz) = 0.5*( u(Ix,Iy,Iz)*u(Ix,Iy,Iz) + v(Ix,Iy,Iz)*v(Ix,Iy,Iz) + w(Ix,Iy,Iz)*w(Ix,Iy,Iz) );
      t(Ix,Iy,Iz) = tinf + t(Ix,Iy,Iz);
      p(Ix,Iy,Iz) = POW(POW(t(Iy,Iy,Iz),GM)*(1.0/sinf) , GM1I );
    //  t.pull();t.display();
     rho(Ix,Iy,Iz) = p(Ix,Iy,Iz)/t(Ix,Iy,Iz);

      Q[0](Ix,Iy,Iz) = rho(Ix,Iy,Iz);
      Q[1](Ix,Iy,Iz) = rho(Ix,Iy,Iz)*u(Ix,Iy,Iz);
      Q[2](Ix,Iy,Iz) = rho(Ix,Iy,Iz)*v(Ix,Iy,Iz);
      Q[3](Ix,Iy,Iz) = rho(Ix,Iy,Iz)*w(Ix,Iy,Iz);
      Q[4](Ix,Iy,Iz) = p(Ix,Iy,Iz)*GM1I + rho(Ix,Iy,Iz)*UB(Ix,Iy,Iz); 
      Q[5](Ix,Iy,Iz) = p(Ix,Iy,Iz);

      //Q[0].pull(); Q[0].display();
}



//--- End of code
