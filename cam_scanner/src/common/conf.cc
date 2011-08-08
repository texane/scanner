#include <cstdlib>
#include <string>
#include <cv.h>
#include "conf.hh"


int load_conf(conf_t& conf, const std::string& confname)
{
  // confname the configuration file name

  // Open file storage for XML-formatted configuration file.
  CvFileStorage* fs = cvOpenFileStorage(confname.c_str(), 0, CV_STORAGE_READ);
  if (fs == NULL) return -1;

  const char* s;
  CvFileNode* m;
  int err = -1;

  m = cvGetFileNodeByName(fs, 0, "project");
  if (m == NULL) goto on_error;
  s = cvReadStringByName(fs, m, "dirname", NULL);
  if (s == NULL) goto on_error;
  conf.proj_dirname = s;

  m = cvGetFileNodeByName(fs, 0, "camera");
  if (m == NULL) goto on_error;
  conf.cam_index = cvReadIntByName(fs, m, "index", -1);
  if (conf.cam_index == -1) goto on_error;
  conf.cam_width = cvReadIntByName(fs, m, "width", -1);
  if (conf.cam_width == -1) goto on_error;
  conf.cam_height = cvReadIntByName(fs, m, "height", -1);
  if (conf.cam_height == -1) goto on_error;
  conf.cam_height = cvReadIntByName(fs, m, "gain", -1);
  if (conf.cam_gain == -1) goto on_error;

  m = cvGetFileNodeByName(fs, 0, "calib");
  if (m == NULL) goto on_error;
  s = cvReadStringByName(fs, m, "frames_dirname", NULL);
  if (s == NULL) goto on_error;
  conf.calib_frames_dirname = s;
  conf.chess_square_hcount = cvReadIntByName(fs, m, "chess_square_hcount", -1);
  if (conf.chess_square_hcount == -1) goto on_error;
  conf.chess_square_vcount = cvReadIntByName(fs, m, "chess_square_vcount", -1);
  if (conf.chess_square_vcount == -1) goto on_error;
  conf.chess_square_width = cvReadIntByName(fs, m, "chess_square_width", -1);
  if (conf.chess_square_width == -1) goto on_error;
  conf.chess_square_height = cvReadIntByName(fs, m, "chess_square_vcount", -1);
  if (conf.chess_square_height == -1) goto on_error;

  m = cvGetFileNodeByName(fs, 0, "scan");
  if (m == NULL) goto on_error;
  s = cvReadStringByName(fs, m, "frames_dirname", NULL);
  if (s == NULL) goto on_error;
  conf.scan_frames_dirname = s;

  // success
  err = 0;

 on_error:
  cvReleaseFileStorage(&fs);
  return err;
}


int store_conf(const conf_t& conf, const std::string& confname)
{
  CvFileStorage* fs = cvOpenFileStorage(confname.c_str(), 0, CV_STORAGE_WRITE);
  if (fs == NULL) return -1;

  cvStartWriteStruct(fs, "project", CV_NODE_MAP);
  cvWriteString(fs, "dirname", conf.proj_dirname.c_str(), 1);
  cvEndWriteStruct(fs);

  cvStartWriteStruct(fs, "camera", CV_NODE_MAP);
  cvWriteInt(fs, "index", conf.cam_index);
  cvWriteInt(fs, "width", conf.cam_width);
  cvWriteInt(fs, "height", conf.cam_height);
  cvWriteInt(fs, "gain", conf.cam_gain);
  cvEndWriteStruct(fs);

  cvStartWriteStruct(fs, "calib", CV_NODE_MAP);
  cvWriteString(fs, "frames_dirname", conf.calib_frames_dirname.c_str(), 1);
  cvWriteInt(fs, "chess_square_hcount", conf.chess_square_hcount);
  cvWriteInt(fs, "chess_square_vcount", conf.chess_square_vcount);
  cvWriteInt(fs, "chess_square_height", conf.chess_square_height);
  cvWriteInt(fs, "chess_square_width", conf.chess_square_width);
  cvEndWriteStruct(fs);

  cvStartWriteStruct(fs, "scan", CV_NODE_MAP);
  cvWriteString(fs, "frames_dirname", conf.scan_frames_dirname.c_str(), 1);
  cvEndWriteStruct(fs);

  cvReleaseFileStorage(&fs);

  return 0;
}
