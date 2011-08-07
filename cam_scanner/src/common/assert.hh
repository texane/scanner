#ifndef COMMON_ASSERT_HH_INCLUDED
# define COMMON_ASSERT_HH_INCLUDED


#include <cstdio>


#define ASSERT_RETURN(__c, __n)						\
if (!(__c))								\
{									\
  ::printf("[ASSERTION_FAILED %s/%u]: %s\n", __FILE__, __LINE__, #__c);	\
  return __n;								\
}


#define ASSERT_GOTO(__c, __l)						\
if (!(__c))								\
{									\
  ::printf("[ASSERTION_FAILED %s/%u]: %s\n", __FILE__, __LINE__, #__c);	\
  goto __l;								\
}


#endif // ! COMMON_ASSERT_HH_INCLUDED
