//! SIN
//!-----------------------------------------------------------------------
template < typename E >
class __SIN_G__ 
{
  E const _u;
  public:
  
  __SIN_G__( E const & u ): _u(u){}
 ~__SIN_G__(){}
 __device__ real operator [] ( int i ) const  { return __sinf(_u[i]);}

};

template < typename E >
class __SIN_Array__ 
{
  E const _u;
  public:
  
  __SIN_Array__( E const & u ): _u(u) { _u.isAllocated = false;}
 ~__SIN_Array__(){}
  __device__ real operator [] ( int i ) const { return __sinf(_u[i]);}

};


template < typename E >
__SIN_G__< E > const SIN( E const u )
{
return __SIN_G__<E>(u);
}

__SIN_Array__ < Array > const SIN( distArray const u )
{
  u.array.isAllocated = false;
  return __SIN_Array__<Array>(u.array);
}
//! EXP
//!-----------------------------------------------------------------------
template < typename E >
class __EXP_G__ 
{
  E const _u;
  public:
  
  __EXP_G__( E const & u ): _u(u){}
 ~__EXP_G__(){}
 __device__ real operator [] ( int i ) const  { return __expf(_u[i]);}

};

template < typename E >
class __EXP_Array__ 
{
  E const _u;
  public:
  
  __EXP_Array__( E const & u ): _u(u) { _u.isAllocated = false;}
 ~__EXP_Array__(){}
  __device__ real operator [] ( int i ) const { return __expf(_u[i]);}

};


template < typename E >
__EXP_G__< E > const EXP( E const u )
{
return __EXP_G__<E>(u);
}

__EXP_Array__ < Array > const EXP( distArray const u )
{
  u.array.isAllocated = false;
  return __EXP_Array__<Array>(u.array);
}
//! POW
//!-----------------------------------------------------------------------
template < typename E >
class __POW_G__ 
{
  E    const _u;
  real const _v;
  public:
  
  __POW_G__( E const & u, real const v): _u(u), _v(v){}
 ~__POW_G__(){}
 __device__ real operator [] ( int i ) const  { return __powf(_u[i],_v);}

};

template < typename E >
class __POW_Array__ 
{
  E    const _u;
  real const _v;
  public:
  
  __POW_Array__( E const & u, real const v ): _u(u), _v(v) { _u.isAllocated = false;}
 ~__POW_Array__(){}
  __device__ real operator [] ( int i ) const { return __powf(_u[i],_v);}

};


template < typename E >
__POW_G__< E > const POW( E const u, real v )
{
return __POW_G__<E>(u,v);
}

__POW_Array__ < Array > const POW( distArray const u, real v )
{
  u.array.isAllocated = false;
  return __POW_Array__<Array>(u.array,v);
}
//! ABS
//!-----------------------------------------------------------------------
template < typename E >
class __ABS_G__ 
{
  E const _u;
  public:
  
  __ABS_G__( E const & u ): _u(u){}
 ~__ABS_G__(){}
 __device__ real operator [] ( int i ) const  { return fabsf(_u[i]);}

};

template < typename E >
class __ABS_Array__ 
{
  E const _u;
  public:
  
  __ABS_Array__( E const & u ): _u(u) { _u.isAllocated = false;}
 ~__ABS_Array__(){}
  __device__ real operator [] ( int i ) const { return fabsf(_u[i]);}

};


template < typename E >
__ABS_G__< E > const ABS( E const u )
{
return __ABS_G__<E>(u);
}

__ABS_Array__ < Array > const ABS( distArray const u )
{
  u.array.isAllocated = false;
  return __ABS_Array__<Array>(u.array);
}
//! SQRT
//!-----------------------------------------------------------------------
template < typename E >
class __SQRT_G__ 
{
  E const _u;
  public:
  
  __SQRT_G__( E const & u ): _u(u){}
 ~__SQRT_G__(){}
 __device__ real operator [] ( int i ) const  { return sqrtf(_u[i]);}

};

template < typename E >
class __SQRT_Array__ 
{
  E const _u;
  public:
  
  __SQRT_Array__( E const & u ): _u(u) { _u.isAllocated = false;}
 ~__SQRT_Array__(){}
  __device__ real operator [] ( int i ) const { return sqrtf(_u[i]);}

};


template < typename E >
__SQRT_G__< E > const SQRT( E const u )
{
return __SQRT_G__<E>(u);
}

__SQRT_Array__ < Array > const SQRT( distArray const u )
{
  u.array.isAllocated = false;
  return __SQRT_Array__<Array>(u.array);
}



