// General Includes
#ifndef _eu3d_h
#define _eu3d_h

void setInitialCondition
(
    cartGrid & cg,
    vectorGridFunction & Q
);


void setExactSolution
(
    cartGrid & cg,
    vectorGridFunction & Q,
    real  solTime
);

void computeConvectiveFluxes
(
    cartGrid & cg, 
    vectorGridFunction & Q,
    vectorGridFunction & E,
	vectorGridFunction & F,
    vectorGridFunction & G
);

void updateBoundaries
( 
    cartGrid & cg, 
    vectorGridFunction & Q 
);

void computeRHS
(
    cartGrid & cg, 
    vectorGridFunction & Q,
    vectorGridFunction & E,
	vectorGridFunction & F,
    vectorGridFunction & G,
    distArray & sigmaX,
    distArray & sigmaY,
    distArray & sigmaZ
);

void runStep
( 
    cartGrid & cg, 
    vectorGridFunction & Q, 
    vectorGridFunction & QN,
    vectorGridFunction & E, 
    vectorGridFunction & F, 
    vectorGridFunction & G, 
    distArray & sigmaX,
    distArray & sigmaY,
    distArray & sigmaZ, 
    real dt,
    vectorGridFunction & E0
);
void runStepEuler
( 
    cartGrid & cg, 
    vectorGridFunction & Q, 
    vectorGridFunction & QN,
    vectorGridFunction & E, 
    vectorGridFunction & F, 
    vectorGridFunction & G, 
    distArray & sigmaX,
    distArray & sigmaY,
    distArray & sigmaZ, 
    real dt
);
void computeSpecRad
(
    cartGrid & cg, 
    vectorGridFunction & Q, 
    distArray & sigmaX, 
    distArray & sigmaY, 
    distArray & sigmaZ
);

void updatePressure
(
    vectorGridFunction & Q, 
    Index & Ix, 
    Index & Iy, 
    Index & Iz
);

#define GM 1.4
#define GM1 (GM-1)
#define GGM1 GM*GM1
#define GMI (1.0/GM)
#define GM1I (1.0/GM1)
#define GM1IG  GM1/GM

#endif


