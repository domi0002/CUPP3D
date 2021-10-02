template < typename E1, typename E2 >
class MulArrayReal
{
    E1  const  _u;
    E2  const  _v;
    

 public:
     
  MulArrayReal( E1 const & u , E2 const & v ) : _u(u), _v(v)
    {
     _u.isAllocated = false;      
    
    #if (EMU)
    cout<<" Calling Constructor MulArrayReal"<<endl;     
    #endif
 
    }  
    
  ~MulArrayReal()
    {
      
    }    
  __device__ real operator [] (int i) const 
   {  
     #if (EMU)
      if ( i == 0 ) printf(" In operator MulArrayReal");
     #endif        

     return _u[i] * _v;  
	  
   }
    

};

template < typename E1, typename E2 >
class MulRealArray
{
    E1  const  _u;
    E2  const  _v;
    

 public:
     
  MulRealArray( E1 const & u , E2 const & v ) : _u(u), _v(v)//, _du(&u), _dv(&v)
    {
     _v.isAllocated = false;
    
    #if ( EMU )
    cout<<" Calling Constructor MulRealArray"<<endl;         
    #endif
        
    }
  
    
  ~MulRealArray()
    {
      
    }    
  __device__ real operator [] (int i) const 
   {  
      #if (EMU)
        if ( i == 0 ) printf(" In operator MulRealArray");
      #endif 
     
      return _u * _v[i];  
	  
   }
  

};

template < typename E1, typename E2 >
class MulGenReal
{
    E1  const  _u;
    E2  const  _v;
    

 public:
     
  MulGenReal( E1 const & u , E2 const & v ) : _u(u), _v(v)
    {
       #if (EMU)
        cout<<" Calling Constructor MulGenReal"<<endl;     
       #endif
       
    }
  
    
  ~MulGenReal()
    {
      
    }    
  __device__ real operator [] (int i) const 
   {  
      #if (EMU)
        if ( i == 0 ) printf(" In operator MulGenReal");
      #endif
     
      return _u[i] * _v;  
	  
   }
  

};

template < typename E1, typename E2 >
class MulRealGen
{
    E1  const  _u;
    E2  const  _v;
    
 public:
     
  MulRealGen( E1 const & u , E2 const & v ) : _u(u), _v(v)
    {
       #if ( EMU )
         cout<<" Calling Constructor MulRealGen"<<endl;     
       #endif
      
    }
  
    
  ~MulRealGen()
    {
      
    }    
  __device__ real operator [] (int i) const 
   {  
      #if (EMU)
       if ( i == 0 ) printf(" In operator MulRealGen");
      #endif
      
      return _u * _v[i];  
	  
   }
  

};


MulArrayReal< Array, real > const operator * ( distArray const  u , double const & v )
{
  
   return MulArrayReal<Array,real>(u.array,static_cast<real>(v) );
}

MulRealArray< real , Array > const operator * ( double const & u , distArray const  v )
{
  
   return MulRealArray<real,Array>(static_cast<real>(u),v.array);
}

template < typename E >
MulGenReal< E, real> const operator * ( E const  u , double const & v)
{
   return MulGenReal<E,real>(u,static_cast<real>(v));   
}

template < typename E >
MulRealGen<real, E> const operator * ( double const & u, E const  v)
{
   return MulRealGen<real, E>(static_cast<real>(u),v);
}   


//----------------------------------------------------------------
template < typename E1, typename E2 >
class Mul
{
    E1  const  _u;
    E2  const  _v;
    

 public:
     
  Mul( E1 const & u , E2 const & v ) : _u(u), _v(v)
    {
      #if (EMU)
        cout<<" Calling Constructor MulGenGen"<<endl;
      #endif
      
    }
   
  ~Mul()
    {
      
    }    
  
__device__  real operator [] (int i) const 
   {  
  
      #if (EMU)
        if ( i == 0 ) printf(" In operator MulGenGen");
      #endif

       return   _u[i] * _v[i];
  
   }
  

};

template < typename E1, typename E2>
class MulArrayArray
{
    E1 const    _u;
    E2 const    _v;
    
 public:
     
  
  MulArrayArray( Array  const & u, Array  const & v ): _u(u), _v(v) 
  {
    _u.isAllocated = false ; _v.isAllocated =false;
    #if ( EMU )
     cout<<" Calling Constructor MulArrayArray"<<endl;
    #endif
    
  }  
     
  ~MulArrayArray()
    {
      
    }

  __device__  real operator [] (int i) const 
   { 
      
      return _u[i] * _v[i] ;
     
    }


};

template < typename E1, typename E2 >
class MulGenArray
{
    E1  const  _u;
    E2  const  _v;
    
 public:
     
  
  MulGenArray( E1 const & u, Array const & v ): _u(u), _v(v) 
  {
    _v.isAllocated =false;
    #if ( EMU )
     cout<<" Calling Constructor MulGenArray"<<endl; 
    #endif
    
   
  }  
     
  ~MulGenArray()
    {
      
    }    
  __device__  real operator [] (int i) const 
   {
      
      return _u[i] * _v[i];
      
   }
  

};

template < typename E1, typename E2 >
class MulArrayGen
{
    E1  const  _u;
    E2  const  _v;
    
 public:
     
  
  MulArrayGen( Array const & u, E2 const & v ): _u(u), _v(v) 
  {
    _u.isAllocated =false;
    #if ( EMU )
     cout<<" Calling Constructor MulArrayGen "<<endl;
    #endif
        
  }  
     
  ~MulArrayGen()
    {
      
    }    
  __device__   real operator [] (int i) const 
   {  
     return _u[i] * _v[i];
         
   }
  
  

};


MulArrayArray< Array, Array > const operator * ( distArray const   u , distArray const 	  v )
{
   #if ( EMU )
   cout<<"U Offset from MulArrayArray "<<u.offset<<endl; 
   cout<<"V Offset from MulArrayArray "<<v.offset<<endl; 
   #endif

   u.array.isAllocated=false;
   v.array.isAllocated=false;
   return MulArrayArray<Array,Array>(u.array,v.array);
}

template < typename E >
MulGenArray< E, Array > const operator * ( E const  u , distArray const  v )
{
   #if ( EMU )
   cout<<"V Offset from MulGenArray "<<v.offset<<endl; 
   #endif

   v.array.isAllocated=false;
   return MulGenArray<E,Array>(u,v.array);
}


template < typename E >
MulArrayGen< Array, E > const operator * ( distArray const  u , E const  v )
{
   #if ( EMU )
    cout<<"U Offset from MulGenArray "<<u.offset<<endl; 
   #endif

   u.array.isAllocated = false;
   return MulArrayGen<Array,E>(u.array,v);
}

template < typename E1, typename E2 >
Mul< E1 , E2 > const operator * ( E1 const  u , E2 const  v )
{
  
   return Mul<E1,E2>(u,v);
}

