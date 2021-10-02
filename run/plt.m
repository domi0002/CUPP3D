%function f = plt(filename,Nx,Ny, nFringe)
clear all
grid1=122;

% Exact solution is the initial condition after 'N' periods where N is a positive integer
filename_e = strcat(num2str(grid1),'/density0.dat');

% This is the computed solution
filename_t = strcat(num2str(grid1),'/densityFinal.dat');

% Number of fringes used in the discretization
nFringe=3;

% Assuming isotropic mesh count in x and y.
Nx=grid1;
Ny=grid1;
npoints=Nx*Ny;


% Load the files 
sol_e= load(filename_e);
sol  = load(filename_t);

% Add fringe points
ncol = Nx+nFringe*2;
nrow = Ny+nFringe*2;

% Just pick the 1st Z-plane after all the fringe points
start = nrow*nFringe + nFringe + 1;
endt  = start + Ny-1;

% Create a uniform mesh for plotting
% Domain extents from the Euler code
[x ,y] = meshgrid(linspace(0,20.0,Nx),linspace(0,20.0,Ny));


% Pick out the solution
adjustedSolution=sol(start:endt,nFringe+1:ncol-nFringe);
adjustedSolutionE=sol_e(start:endt,nFringe+1:ncol-nFringe);

% Calculate Error
solErrorMatrix=adjustedSolution-adjustedSolutionE;

figure(1)
% Plot solution
contourf(x,y,adjustedSolution,20);
axis equal
grid


figure(2)
% Plot error
contourf(x,y,solErrorMatrix,20);

% Calculate the L2 Norm of the error
solErrorNorm=norm(solErrorMatrix,'fro')/sqrt(npoints)




