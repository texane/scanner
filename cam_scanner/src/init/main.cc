#include <cstdio>
#include <cstdlib>
#include <list>
#include <string>
#include <sys/stat.h>
#include <cv.h>
#include <highgui.h>
#include "../common/conf.hh"
#include "../common/assert.hh"


int main(int ac, char** av)
{
  std::string dirname;
  std::string confname;
  conf_t conf;

  if (ac < 1) return -1;

  dirname = av[1];
  confname = dirname + "/conf.xml";

  if (mkdir(av[1], 0755) == -1) return -1;

  // initialize a default configuration
  conf.proj_dirname = dirname;
  conf.calib_frames_dirname = "calib_frames";
  conf.scan_frames_dirname = "scan_frames";
  conf.cam_index = 0;

  store_conf(conf, confname);

  return 0;
}
