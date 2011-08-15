#ifndef COMMON_FIXED_VECTOR_HH_INCLUDED
# define COMMON_FIXED_VECTOR_HH_INCLUDED


#include "common/real_type.hh"


template<typename T, unsigned int N>
struct fixed_vector
{
  enum { size = N };

  T _data[N];

  const T& operator[](unsigned int i) const
  { return _data[i]; }

  T& operator[](unsigned int i)
  { return _data[i]; }
};


#endif // ! COMMON_FIXED_VECTOR_HH_INCLUDED
