#ifndef COMMON_CONF_HH_INCLUDED
# define COMMON_CONF_HH_INCLUDED


#include <string>


// project configuration
typedef struct conf
{
  // project
  std::string proj_dirname;

  // calibration
  std::string calib_frames_dirname;
  int chess_square_hcount;
  int chess_square_vcount;
  int chess_square_width; // in mm
  int chess_square_height;

  // scanning
  std::string scan_frames_dirname;

  // camera
  int cam_index;
  int cam_width;
  int cam_height;
  int cam_gain;

} conf_t;


int load_conf(conf_t&, const std::string&);
int store_conf(const conf_t&, const std::string&);

// fixme
static inline int conf_load(conf_t& conf, const std::string& filename)
{
  return load_conf(conf, filename);
}

// fixme
static inline int conf_save(const conf_t& conf, const std::string& filename)
{
  return store_conf(conf, filename);
}


#endif // ! COMMON_CONF_HH_INCLUDED
