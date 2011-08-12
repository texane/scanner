#include <cstdio>
#include <cstdlib>
#include <list>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <cv.h>
#include <highgui.h>
#include "../common/conf.hh"
#include "../common/assert.hh"


static inline int do_mkdir(const char* dirname)
{
  return mkdir(dirname, 0755);
}

int main(int ac, char** av)
{
  std::string dirname;
  std::string confname;
  std::string subname;
  conf_t conf;

  if (ac < 1) return -1;

  dirname = av[1];
  confname = dirname + "/conf.xml";

  // initialize a default configuration
  conf.proj_dirname = dirname;
  if (do_mkdir(dirname.c_str()) == -1)
  {
    printf("[!] cannot create project directory\n");
    printf("[!] remove manually if it already exists\n");
    return -1;
  }

  // calibration
  conf.calib_frames_dirname = "calib_frames";
  subname = dirname + "/" + conf.calib_frames_dirname.c_str();
  if (do_mkdir(subname.c_str()) == -1) return -1;
  conf.chess_square_hcount = 8;
  conf.chess_square_vcount = 6;
  conf.chess_square_width = 30;
  conf.chess_square_height = 30;

  // scan
  conf.scan_frames_dirname = "scan_frames";
  subname = dirname + "/" + conf.scan_frames_dirname.c_str();
  if (do_mkdir(subname.c_str()) == -1) return -1;

  // camera
  conf.cam_index = 0;
  conf.cam_width = 640;
  conf.cam_height = 480;
  conf.cam_gain = 50;

  store_conf(conf, confname);

  return 0;
}
