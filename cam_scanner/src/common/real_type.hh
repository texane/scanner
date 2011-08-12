#ifndef COMMON_REAL_TYPE_HH_INCLUDED
# define COMMON_REAL_TYPE_HH_INCLUDED

#ifndef REAL_TYPE_IS_DOUBLE
# define REAL_TYPE_IS_DOUBLE 1
#endif

#if REAL_TYPE_IS_DOUBLE
typedef double real_type;
#else
typedef float real_type;
#endif

#endif // ! COMMON_REAL_TYPE_HH_INCLUDED
