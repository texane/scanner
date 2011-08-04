/*
** Made by fabien le mentec <texane@gmail.com>
** 
** Started on  Wed Nov 11 16:47:08 2009 texane
** Last update Wed Nov 11 16:47:29 2009 texane
*/



#ifndef DEBUG_H_INCLUDED
# define DEBUG_H_INCLUDED



#if _DEBUG

# include <cstdio>

# define DEBUG_ENTER() ::printf("[>] %s\n", __FUNCTION__)
# define DEBUG_LEAVE() ::printf("[<] %s\n", __FUNCTION__)
# define DEBUG_PRINTF(s, ...) ::printf("[?] %s_%u: " s, __FUNCTION__, __LINE__, ## __VA_ARGS__)
# define DEBUG_ERROR(s, ...) ::printf("[!] %s_%u: " s, __FUNCTION__, __LINE__, ## __VA_ARGS__)

#else /* ! _DEBUG */

# define DEBUG_ENTER()
# define DEBUG_LEAVE()
# define DEBUG_PRINTF(s, ...)
# define DEBUG_ERROR(s, ...)

#endif /* _DEBUG */



#endif /* DEBUG_H_INCLUDED */
