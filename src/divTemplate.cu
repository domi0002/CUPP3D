//! ------------------------------------------------------------------
//! CU++ET or CUPPET
//! 10/12/2010
//! Dominic Chandar
//! University of Wyoming
//! Solving Discrete equations using CUDA and Expression Templates
//! This code generates CUDA kernels automatically at compile time
//!
//! CODE   : DivTemplate.cu
//! Purpose:
//! This section of the code handles the generation of abstract objects
//! at compile time for Divtiplication of operands
//! 
//! ------------------------------------------------------------------
//!

template < typename E1, typename E2 >
class DivArrayReal
{
    E1  const  _u;
    E2  const  _v;
    

 public:
     
  DivArrayReal( E1 const & u , E2 const & v ) : _u(u), _v(v)
    {
     _u.isAllocated = false;      
    
    #if (EMU)
    cout<<" Calling Constructor DivArrayReal"<<endl;     
    #endif
 
    }  
    
  ~DivArrayReal()
    {
      
    }    
  __device__ real operator [] (int i) const 
   {  
     #if (EMU)
      if ( i == 0 ) printf(" In operator DivArrayReal");
     #endif        

     return _u[i] / _v;  
	  
   }
    

};

template < typename E1, typename E2 >
class DivRealArray
{
    E1  const  _u;
    E2  const  _v;
    

 public:
     
  DivRealArray( E1 const & u , E2 const & v ) : _u(u), _v(v)//, _du(&u), _dv(&v)
    {
     _v.isAllocated = false;
    
    #if ( EMU )
    cout<<" Calling Constructor DivRealArray"<<endl;         
    #endif
        
    }
  
    
  ~DivRealArray()
    {
      
    }    
  __device__ real operator [] (int i) const 
   {  
      #if (EMU)
        if ( i == 0 ) printf(" In operator DivRealArray");
      #endif 
     
      return _u / _v[i];  
	  
   }
  

};

template < typename E1, typename E2 >
class DivGenReal
{
    E1  const  _u;
    E2  const  _v;
    

 public:
     
  DivGenReal( E1 const & u , E2 const & v ) : _u(u), _v(v)
    {
       #if (EMU)
        cout<<" Calling Constructor DivGenReal"<<endl;     
       #endif
       
    }
  
    
  ~DivGenReal()
    {
      
    }    
  __device__ real operator [] (int i) const 
   {  
      #if (EMU)
        if ( i == 0 ) printf(" In operator DivGenReal");
      #endif
     
      return _u[i] / _v;  
	  
   }
  

};

template < typename E1, typename E2 >
class DivRealGen
{
    E1  const  _u;
    E2  const  _v;
    
 public:
     
  DivRealGen( E1 const & u , E2 const & v ) : _u(u), _v(v)
    {
       #if ( EMU )
         cout<<" Calling Constructor DivRealGen"<<endl;     
       #endif
      
    }
  
    
  ~DivRealGen()
    {
      
    }    
  __device__ real operator [] (int i) const 
   {  
      #if (EMU)
       if ( i == 0 ) printf(" In operator DivRealGen");
      #endif
      
      return _u / _v[i];  
	  
   }
  

};


DivArrayReal< Array, real > const operator / ( distArray const  u , double const & v )
{
   u.array.isAllocated = false;
   return DivArrayReal<Array,real>(u.array,static_cast<real>(v) );
}

DivRealArray< real , Array > const operator / ( double const & u , distArray const  v )
{
  v.array.isAllocated = false;
   return DivRealArray<real,Array>(static_cast<real>(u),v.array);
}

template < typename E >
DivGenReal< E, real> const operator / ( E const  u , double const & v)
{
   return DivGenReal<E,real>(u,static_cast<real>(v));   
}

template < typename E >
DivRealGen<real, E> const operator / ( double const & u, E const  v)
{
   return DivRealGen<real, E>(static_cast<real>(u),v);
}   


//----------------------------------------------------------------
template < typename E1, typename E2 >
class Div
{
    E1  const  _u;
    E2  const  _v;
    

 public:
     
  Div( E1 const & u , E2 const & v ) : _u(u), _v(v)
    {
      #if (EMU)
        cout<<" Calling Constructor DivGenGen"<<endl;
      #endif
      
    }
   
  ~Div()
    {
      
    }    
  
__device__  real operator [] (int i) const 
   {  
  
      #if (EMU)
        if ( i == 0 ) printf(" In operator DivGenGen");
      #endif

       return   _u[i] / _v[i];
  
   }
  

};

template < typename E1, typename E2>
class DivArrayArray
{
    E1 const    _u;
    E2 const    _v;
    
 public:
     
  
  DivArrayArray( Array  const & u, Array  const & v ): _u(u), _v(v) 
  {
    _u.isAllocated = false ; _v.isAllocated =false;
    #if ( EMU )
     cout<<" Calling Constructor DivArrayArray"<<endl;
    #endif
    
  }  
     
  ~DivArrayArray()
    {
      
    }

  __device__  real operator [] (int i) const 
   { 
      
      return _u[i] / _v[i] ;
     
    }


};

template < typename E1, typename E2 >
class DivGenArray
{
    E1  const  _u;
    E2  const  _v;
    
 public:
     
  
  DivGenArray( E1 const & u, Array const & v ): _u(u), _v(v) 
  {
    _v.isAllocated =false;
    #if ( EMU )
     cout<<" Calling Constructor DivGenArray"<<endl; 
    #endif
    
   
  }  
     
  ~DivGenArray()
    {
      
    }    
  __device__  real operator [] (int i) const 
   {
      
      return _u[i] / _v[i];
      
   }
  

};

template < typename E1, typename E2 >
class DivArrayGen
{
    E1  const  _u;
    E2  const  _v;
    
 public:
     
  
  DivArrayGen( Array const & u, E2 const & v ): _u(u), _v(v) 
  {
    _u.isAllocated =false;
    #if ( EMU )
     cout<<" Calling Constructor DivArrayGen "<<endl;
    #endif
        
  }  
     
  ~DivArrayGen()
    {
      
    }    
  __device__   real operator [] (int i) const 
   {  
     return _u[i] / _v[i];
         
   }
  
  

};


DivArrayArray< Array, Array > const operator / ( distArray const   u , distArray const 	  v )
{
   #if ( EMU )
   cout<<"U Offset from DivArrayArray "<<u.offset<<endl; 
   cout<<"V Offset from DivArrayArray "<<v.offset<<endl; 
   #endif

   u.array.isAllocated=false;
   v.array.isAllocated=false;
   return DivArrayArray<Array,Array>(u.array,v.array);
}

template < typename E >
DivGenArray< E, Array > const operator / ( E const  u , distArray const  v )
{
   #if ( EMU )
   cout<<"V Offset from DivGenArray "<<v.offset<<endl; 
   #endif

   v.array.isAllocated=false;
   return DivGenArray<E,Array>(u,v.array);
}


template < typename E >
DivArrayGen< Array, E > const operator / ( distArray const  u , E const  v )
{
   #if ( EMU )
    cout<<"U Offset from DivGenArray "<<u.offset<<endl; 
   #endif

   u.array.isAllocated = false;
   return DivArrayGen<Array,E>(u.array,v);
}

template < typename E1, typename E2 >
Div< E1 , E2 > const operator / ( E1 const  u , E2 const  v )
{
  
   return Div<E1,E2>(u,v);
}

