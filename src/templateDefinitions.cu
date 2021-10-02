//! ------------------------------------------------------------------
//! CU++ET or CUPPET
//! 10/12/2010
//! Dominic Chandar
//! University of Wyoming
//! Solving Discrete equations using CUDA and Expression Templates
//! This code generates CUDA kernels automatically at compile time
//!
//! CODE    : templateDefinitions.cu
//! Purpose :
//! This section of the code is the 'Final' stage of computing an expression
//! There is ONLY ONE kernel to evaluate ANY expression. The abstract
//! expression is given by the type 'Eob'
//! 
//! ------------------------------------------------------------------
//!
#define DO_NOT_USE_COPY 1

template < typename E >
void distArray::compute( E & Eob)
{
   #if( printPassingTypeAndSize )
   
   int status;
   char* passName = abi::__cxa_demangle(typeid(E).name(),0,0,&status);
   cout<<"Type of passing data     ="<<passName<<endl;
   cout<<"size of passing dataType ="<<sizeof(E)<<endl;
   free (passName);

   #endif
   
   if ( numberOfDimensions == 1)
   {  
        
        // New way: ( confusing, but time efficient )
        // 1. Copy the actual array into the temporary array
        #if ( DO_NOT_USE_COPY )
        array.dVEC-=offset;
        cudaMemcpy(dVECTemp,array.dVEC,Nx*sizeof(real),cudaMemcpyDeviceToDevice);
        array.dVEC+=offset;
        
        // 2. Return the computed value to the temporary array
        dVECTemp+=offset;
        computeExpressionInGPU1D<<<grid,block>>>(Eob,dVECTemp,Nx,span0);
        cudaDeviceSynchronize();
        dVECTemp-=offset;  
        
        // 3. Simply swap the temporary array with the actual array. 
        resetOffset();
        real* swap = array.dVEC;
        array.dVEC=dVECTemp; 
        dVECTemp = swap;
        #endif       
   
        #if ( !DO_NOT_USE_COPY )
        // Old way: Using Two kernels.. ( simple but slighlty expensive )
        // 1. Return the computed value into the temporary array
        computeExpressionInGPU1D<<<grid,block>>>(Eob,dVECTemp,Nx,span0);
        cudaDeviceSynchronize();
        
        // 2. copy the temporary array into the actual array
        copyResult1D<<<grid,block>>>(array.dVEC,dVECTemp,Nx,span0);
        cudaDeviceSynchronize();
        resetOffset();
        #endif    
   }


   if ( numberOfDimensions == 2)
   {  
        // New way: ( confusing, but time efficient )
        // 1. Copy the actual array into the temporary array
        #if ( DO_NOT_USE_COPY )
        array.dVEC-=offset;
        cudaMemcpy(dVECTemp,array.dVEC,Nx*Ny*sizeof(real),cudaMemcpyDeviceToDevice);
        array.dVEC+=offset;
        
        // 2. Return the computed value to the temporary array
        dVECTemp+=offset;
        computeExpressionInGPU2D<<<grid,block>>>(Eob,dVECTemp,Nx,Ny,span0,span1);
        cudaDeviceSynchronize();
        dVECTemp-=offset;  
        
        // 3. Simply swap the temporary array with the actual array. 
        resetOffset();
        real* swap = array.dVEC;
        array.dVEC=dVECTemp; 
        dVECTemp = swap;
        #endif       
   
        #if ( !DO_NOT_USE_COPY )
        // Old way: Using Two kernels.. ( simple but slighlty expensive )
        // 1. Return the computed value into the temporary array
        computeExpressionInGPU2D<<<grid,block>>>(Eob,dVECTemp,Nx,Ny,span0,span1);
        cudaDeviceSynchronize();
        
        // 2. copy the temporary array into the actual array
        copyResult2D<<<grid,block>>>(array.dVEC,dVECTemp,Nx,Ny,span0,span1);
        cudaDeviceSynchronize();
        resetOffset();
        #endif
    
   }
 
   if ( numberOfDimensions == 3)
   {  
        // New way: ( confusing, but time efficient )
        // 1. Copy the actual array into the temporary array
        #if ( DO_NOT_USE_COPY )
        array.dVEC-=offset;
        cudaMemcpy(dVECTemp,array.dVEC,Nx*Ny*Nz*sizeof(real),cudaMemcpyDeviceToDevice);
        array.dVEC+=offset;
        
        // 2. Return the computed value to the temporary array
        dVECTemp+=offset;
        computeExpressionInGPU3D<<<grid,block>>>(Eob,dVECTemp,Nx,Ny,Nz,span0,span1,span2);
        //cudaDeviceSynchronize();
        dVECTemp-=offset;  
        
        // 3. Simply swap the temporary array with the actual array. 
        resetOffset();
        real* swap = array.dVEC;
        array.dVEC=dVECTemp; 
        dVECTemp = swap;
        #endif       
   
        #if ( !DO_NOT_USE_COPY )
        // Old way: Using Two kernels.. ( simple but slighlty expensive )
        // 1. Return the computed value into the temporary array
        computeExpressionInGPU3D<<<grid,block>>>(Eob,dVECTemp,Nx,Ny,Nz,span0,span1,span2);
        cudaDeviceSynchronize();
        
        // 2. copy the temporary array into the actual array
        copyResult3D<<<grid,block>>>(array.dVEC,dVECTemp,Nx,Ny,Nz,span0,span1,span2);
        cudaDeviceSynchronize();
        resetOffset();
        #endif
   }
  
     
}

