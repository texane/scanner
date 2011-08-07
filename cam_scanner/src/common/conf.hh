#ifndef COMMON_CONF_HH_INCLUDED
# define COMMON_CONF_HH_INCLUDED


#include <string>


// project configuration
typedef struct conf
{
  std::string proj_dirname;
  std::string calib_frames_dirname;
  std::string scan_frames_dirname;

  unsigned int cam_index;

} conf_t;


int load_conf(conf_t&, const std::string&);
int store_conf(const conf_t&, const std::string&);


#endif // ! COMMON_CONF_HH_INCLUDED
