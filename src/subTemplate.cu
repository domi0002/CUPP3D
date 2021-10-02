//! ------------------------------------------------------------------
//! CU--ET or CUPPET
//! 10/12/2010
//! Dominic Chandar
//! University of Wyoming
//! Solving Discrete equations using CUDA and Expression Templates
//! This code generates CUDA kernels automatically at compile time
//!
//! CODE    : SubTemplate.cu
//! Purpose :
//! This section of the code handles the generation of abstract objects
//! at compile time for subtraction of operands
//! 
//! ------------------------------------------------------------------
//!

template < typename E1, typename E2 >
class SubArrayReal
{
    E1  const  _u;
    E2  const  _v;
    

 public:
     
  SubArrayReal( E1 const & u , E2 const & v ) : _u(u), _v(v)
    {
     _u.isAllocated = false;      
    
    #if (EMU)
    cout<<" Calling Constructor SubArrayReal"<<endl;     
    #endif
 
    }  
    
  ~SubArrayReal()
    {
      
    }    
  __device__ real operator [] (int i) const 
   {  
     #if (EMU)
      if ( i == 0 ) printf(" In operator SubArrayReal");
     #endif        

     return _u[i] - _v;  
	  
   }
    

};

template < typename E1, typename E2 >
class SubRealArray
{
    E1  const  _u;
    E2  const  _v;
    

 public:
     
  SubRealArray( E1 const & u , E2 const & v ) : _u(u), _v(v)//, _du(&u), _dv(&v)
    {
     _v.isAllocated = false;
    
    #if ( EMU )
    cout<<" Calling Constructor SubRealArray"<<endl;         
    #endif
        
    }
  
    
  ~SubRealArray()
    {
      
    }    
  __device__ real operator [] (int i) const 
   {  
      #if (EMU)
        if ( i == 0 ) printf(" In operator SubRealArray");
      #endif 
     
      return _u - _v[i];  
	  
   }
  

};

template < typename E1, typename E2 >
class SubGenReal
{
    E1  const  _u;
    E2  const  _v;
    

 public:
     
  SubGenReal( E1 const & u , E2 const & v ) : _u(u), _v(v)
    {
       #if (EMU)
        cout<<" Calling Constructor SubGenReal"<<endl;     
       #endif
       
    }
  
    
  ~SubGenReal()
    {
      
    }    
  __device__ real operator [] (int i) const 
   {  
      #if (EMU)
        if ( i == 0 ) printf(" In operator SubGenReal");
      #endif
     
      return _u[i] - _v;  
	  
   }
  

};

template < typename E1, typename E2 >
class SubRealGen
{
    E1  const  _u;
    E2  const  _v;
    
 public:
     
  SubRealGen( E1 const & u , E2 const & v ) : _u(u), _v(v)
    {
       #if ( EMU )
         cout<<" Calling Constructor SubRealGen"<<endl;     
       #endif
      
    }
  
    
  ~SubRealGen()
    {
      
    }    
  __device__ real operator [] (int i) const 
   {  
      #if (EMU)
       if ( i == 0 ) printf(" In operator SubRealGen");
      #endif
      
      return _u - _v[i];  
	  
   }
  

};


SubArrayReal< Array, real > const operator - ( distArray const  u , double const & v )
{
   u.array.isAllocated = false;  
   return SubArrayReal<Array,real>(u.array,static_cast<real>(v) );
}

SubRealArray< real , Array > const operator - ( double const & u , distArray const  v )
{
   v.array.isAllocated = false;
   return SubRealArray<real,Array>(static_cast<real>(u),v.array);
}

template < typename E >
SubGenReal< E, real> const operator - ( E const  u , double const & v)
{
   return SubGenReal<E,real>(u,static_cast<real>(v));   
}

template < typename E >
SubRealGen<real, E> const operator - ( double const & u, E const  v)
{
   return SubRealGen<real, E>(static_cast<real>(u),v);
}   


//----------------------------------------------------------------
template < typename E1, typename E2 >
class Sub
{
    E1  const  _u;
    E2  const  _v;
    

 public:
     
  Sub( E1 const & u , E2 const & v ) : _u(u), _v(v)
    {
      #if (EMU)
        cout<<" Calling Constructor SubGenGen"<<endl;
      #endif
      
    }
   
  ~Sub()
    {
      
    }    
  
__device__  real operator [] (int i) const 
   {  
  
      #if (EMU)
        if ( i == 0 ) printf(" In operator SubGenGen");
      #endif

       return   _u[i] - _v[i];
  
   }
  

};

template < typename E1, typename E2>
class SubArrayArray
{
    E1 const    _u;
    E2 const    _v;
    
 public:
     
  
  SubArrayArray( Array  const & u, Array  const & v ): _u(u), _v(v) 
  {
    _u.isAllocated = false ; _v.isAllocated =false;
    #if ( EMU )
     cout<<" Calling Constructor SubArrayArray"<<endl;
    #endif
    
  }  
     
  ~SubArrayArray()
    {
      
    }

  __device__  real operator [] (int i) const 
   { 
      
      
      return _u[i] - _v[i]  ;
     
    }


};

template < typename E1, typename E2 >
class SubGenArray
{
    E1  const  _u;
    E2  const  _v;
    
 public:
     
  
  SubGenArray( E1 const & u, Array const & v ): _u(u), _v(v) 
  {
    _v.isAllocated =false;
    #if ( EMU )
     cout<<" Calling Constructor SubGenArray"<<endl; 
    #endif
    
   
  }  
     
  ~SubGenArray()
    {
      
    }    
  __device__  real operator [] (int i) const 
   {
      
      return _u[i] - _v[i];
      
   }
  

};

template < typename E1, typename E2 >
class SubArrayGen
{
    E1  const  _u;
    E2  const  _v;
    
 public:
     
  
  SubArrayGen( Array const & u, E2 const & v ): _u(u), _v(v) 
  {
    _u.isAllocated =false;
    #if ( EMU )
     cout<<" Calling Constructor SubArrayGen "<<endl;
    #endif
        
  }  
     
  ~SubArrayGen()
    {
      
    }    
  __device__   real operator [] (int i) const 
   {  
     return _u[i] - _v[i];
         
   }
  
  

};


SubArrayArray< Array, Array > const operator - ( distArray const   u , distArray const 	  v )
{
   #if ( EMU )
   cout<<"U Offset from SubArrayArray "<<u.offset<<endl; 
   cout<<"V Offset from SubArrayArray "<<v.offset<<endl; 
   #endif

   u.array.isAllocated=false;
   v.array.isAllocated=false;
   return SubArrayArray<Array,Array>(u.array,v.array);
}

template < typename E >
SubGenArray< E, Array > const operator - ( E const  u , distArray const  v )
{
   #if ( EMU )
   cout<<"V Offset from SubGenArray "<<v.offset<<endl; 
   #endif

   v.array.isAllocated=false;
   return SubGenArray<E,Array>(u,v.array);
}


template < typename E >
SubArrayGen< Array, E > const operator - ( distArray const  u , E const  v )
{
   #if ( EMU )
    cout<<"U Offset from SubGenArray "<<u.offset<<endl; 
   #endif

   u.array.isAllocated = false;
   return SubArrayGen<Array,E>(u.array,v);
}

template < typename E1, typename E2 >
Sub< E1 , E2 > const operator - ( E1 const  u , E2 const  v )
{
  
   return Sub<E1,E2>(u,v);
}


