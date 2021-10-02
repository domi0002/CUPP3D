//! ------------------------------------------------------------------
//! CU++ET or CUPPET
//! 10/12/2010
//! Dominic Chandar
//! University of Wyoming
//! Solving Discrete equations using CUDA and Expression Templates
//! This code generates CUDA kernels automatically at compile time
//!
//! CODE    : addTemplate.cu
//! Purpose :
//! This section of the code handles the generation of abstract objects
//! at compile time for addition of operands
//! 
//! ------------------------------------------------------------------
//!

template < typename E1, typename E2 >
class AddArrayReal
{
    E1  const  _u;
    E2  const  _v;
    

 public:
     
  AddArrayReal( E1 const & u , E2 const & v ) : _u(u), _v(v)
    {
     _u.isAllocated = false;      
    
    #if (EMU)
    cout<<" Calling Constructor AddArrayReal"<<endl;     
    #endif
 
    }  
    
  ~AddArrayReal()
    {
      
    }    
  __device__ real operator [] (int i) const 
   {  
     #if (EMU)
      if ( i == 0 ) printf(" In operator AddArrayReal");
     #endif        

     return _u[i] + _v;  
	  
   }
    

};

template < typename E1, typename E2 >
class AddRealArray
{
    E1  const  _u;
    E2  const  _v;
    

 public:
     
  AddRealArray( E1 const & u , E2 const & v ) : _u(u), _v(v)//, _du(&u), _dv(&v)
    {
     _v.isAllocated = false;
    
    #if ( EMU )
    cout<<" Calling Constructor AddRealArray"<<endl;         
    #endif
        
    }
  
    
  ~AddRealArray()
    {
      
    }    
  __device__ real operator [] (int i) const 
   {  
      #if (EMU)
        if ( i == 0 ) printf(" In operator AddRealArray");
      #endif 
     
      return _u + _v[i];  
	  
   }
  

};

template < typename E1, typename E2 >
class AddGenReal
{
    E1  const  _u;
    E2  const  _v;
    

 public:
     
  AddGenReal( E1 const & u , E2 const & v ) : _u(u), _v(v)
    {
       #if (EMU)
        cout<<" Calling Constructor AddGenReal"<<endl;     
       #endif
       
    }
  
    
  ~AddGenReal()
    {
      
    }    
  __device__ real operator [] (int i) const 
   {  
      #if (EMU)
        if ( i == 0 ) printf(" In operator AddGenReal");
      #endif
     
      return _u[i] + _v;  
	  
   }
  

};

template < typename E1, typename E2 >
class AddRealGen
{
    E1  const  _u;
    E2  const  _v;
    
 public:
     
  AddRealGen( E1 const & u , E2 const & v ) : _u(u), _v(v)
    {
       #if ( EMU )
         cout<<" Calling Constructor AddRealGen"<<endl;     
       #endif
      
    }
  
    
  ~AddRealGen()
    {
      
    }    
  __device__ real operator [] (int i) const 
   {  
      #if (EMU)
       if ( i == 0 ) printf(" In operator AddRealGen");
      #endif
      
      return _u + _v[i];  
	  
   }
  

};


AddArrayReal< Array, real > const operator + ( distArray const  u , double const & v )
{
   u.array.isAllocated = false;  
   return AddArrayReal<Array,real>(u.array,static_cast<real>(v) );
}

AddRealArray< real , Array > const operator + ( double const & u , distArray const  v )
{
   v.array.isAllocated = false;
   return AddRealArray<real,Array>(static_cast<real>(u),v.array);
}

template < typename E >
AddGenReal< E, real> const operator + ( E const  u , double const & v)
{
   return AddGenReal<E,real>(u,static_cast<real>(v));   
}

template < typename E >
AddRealGen<real, E> const operator + ( double const & u, E const  v)
{
   return AddRealGen<real, E>(static_cast<real>(u),v);
}   


//----------------------------------------------------------------
template < typename E1, typename E2 >
class Add
{
    E1  const  _u;
    E2  const  _v;
    

 public:
     
  Add( E1 const & u , E2 const & v ) : _u(u), _v(v)
    {
      #if (EMU)
        cout<<" Calling Constructor AddGenGen"<<endl;
      #endif
      
    }
   
  ~Add()
    {
      
    }    
  
__device__  real operator [] (int i) const 
   {  
  
      #if (EMU)
        if ( i == 0 ) printf(" In operator AddGenGen");
      #endif

       return   _u[i] + _v[i];
  
   }
  

};

template < typename E1, typename E2>
class AddArrayArray
{
    E1 const    _u;
    E2 const    _v;
    
 public:
     
  
  AddArrayArray( Array  const & u, Array  const & v ): _u(u), _v(v) 
  {
    _u.isAllocated = false ; _v.isAllocated =false;
    #if ( EMU )
     cout<<" Calling Constructor AddArrayArray"<<endl;
    #endif
    
  }  
     
  ~AddArrayArray()
    {
      
    }

  __device__  real operator [] (int i) const 
   { 
      
      
      return _u[i] + _v[i]  ;
     
    }


};

template < typename E1, typename E2 >
class AddGenArray
{
    E1  const  _u;
    E2  const  _v;
    
 public:
     
  
  AddGenArray( E1 const & u, Array const & v ): _u(u), _v(v) 
  {
    _v.isAllocated =false;
    #if ( EMU )
     cout<<" Calling Constructor AddGenArray"<<endl; 
    #endif
    
   
  }  
     
  ~AddGenArray()
    {
      
    }    
  __device__  real operator [] (int i) const 
   {
      
      return _u[i] + _v[i];
      
   }
  

};

template < typename E1, typename E2 >
class AddArrayGen
{
    E1  const  _u;
    E2  const  _v;
    
 public:
     
  
  AddArrayGen( Array const & u, E2 const & v ): _u(u), _v(v) 
  {
    _u.isAllocated =false;
    #if ( EMU )
     cout<<" Calling Constructor AddArrayGen "<<endl;
    #endif
        
  }  
     
  ~AddArrayGen()
    {
      
    }    
  __device__   real operator [] (int i) const 
   {  
     return _u[i] + _v[i];
         
   }
  
  

};


AddArrayArray< Array, Array > const operator + ( distArray const   u , distArray const 	  v )
{
   #if ( EMU )
   cout<<"U Offset from addArrayArray "<<u.offset<<endl; 
   cout<<"V Offset from addArrayArray "<<v.offset<<endl; 
   #endif

   u.array.isAllocated=false;
   v.array.isAllocated=false;
   return AddArrayArray<Array,Array>(u.array,v.array);
}

template < typename E >
AddGenArray< E, Array > const operator + ( E const  u , distArray const  v )
{
   #if ( EMU )
   cout<<"V Offset from addGenArray "<<v.offset<<endl; 
   #endif

   v.array.isAllocated=false;
   return AddGenArray<E,Array>(u,v.array);
}


template < typename E >
AddArrayGen< Array, E > const operator + ( distArray const  u , E const  v )
{
   #if ( EMU )
    cout<<"U Offset from addGenArray "<<u.offset<<endl; 
   #endif

   u.array.isAllocated = false;
   return AddArrayGen<Array,E>(u.array,v);
}

template < typename E1, typename E2 >
Add< E1 , E2 > const operator + ( E1 const  u , E2 const  v )
{
  
   return Add<E1,E2>(u,v);
}

//--------------------------------------
// Index Shifting operators
//--------------------------------------
Index operator + ( Index & I, int offset)
{
 return Index(I.base+offset,I.bound+offset,I.direction);
}

Index operator + ( int offset, Index & I)
{
 return Index(I.base+offset,I.bound+offset,I.direction);
}

Index operator - ( Index & I, int offset)
{
 return Index(I.base-offset,I.bound-offset,I.direction);
}

Index operator - ( int offset, Index & I)
{
 return Index(offset-I.base,I.bound+offset,I.direction);
}

