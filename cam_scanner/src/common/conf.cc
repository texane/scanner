#include "conf.hh"


int load_conf(conf_t& conf, const std::string& dirname)
{
  // load the project configuration
  // dirname the project directory

  conf.proj_dirname = "/tmp/o";
  conf.calib_frames_dirname = "calib_frames";
  conf.scan_frames_dirname = "scan_frames";
  conf.cam_index = 0;

  return 0;
}
