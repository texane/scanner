#ifndef COMMON_REAL_TYPE_HH_INCLUDED
# define COMMON_REAL_TYPE_HH_INCLUDED


#include <opencv/cv.h>


#ifndef REAL_TYPE_IS_DOUBLE
# define REAL_TYPE_IS_DOUBLE 1
#endif


#if REAL_TYPE_IS_DOUBLE
typedef double real_type;
static const int real_typeid = CV_64FC1;
#else
typedef float real_type;
static const int real_typeid = CV_32FC1;
#endif


#endif // ! COMMON_REAL_TYPE_HH_INCLUDED
