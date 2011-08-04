#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <sys/stat.h>
#include <string>
#include <cv.h>
#include <highgui.h>
#include "cvStructuredLight.hh"
#include "cvCalibrateProCam.hh"
#include "cvUtilProCam.hh"


static std::string make_indexed_name(const std::string& dirname, unsigned int i)
{
  char namebuf[1024];
  snprintf(namebuf, sizeof(namebuf), "%s/%06u.jpg", dirname.c_str(), i);
  return std::string(namebuf);
}

static int create_avi(const std::string& dirname, const std::string& aviname)
{
  // return -1 on error
  // assume filename: dirname/012345.jpg

  // open the first image to get parameters
  std::string imname;
  imname = make_indexed_name(dirname, 1);
  IplImage* iplimage = cvLoadImage(imname.c_str());
  if (iplimage == NULL) return -1;
  CvSize imsize;
  imsize.width = iplimage->width;
  imsize.height = iplimage->height;
  cvReleaseImage(&iplimage);

  CvVideoWriter* writer = cvCreateVideoWriter
    (aviname.c_str(), CV_FOURCC('P','I','M','1'), 25, imsize, true);

  if (writer == NULL) return -1;

  for (unsigned int i = 1; true; ++i)
  {
    imname = make_indexed_name(dirname, i);
    iplimage = cvLoadImage(imname.c_str());
    if (iplimage == NULL) break ;
    cvWriteFrame(writer, iplimage);
    cvReleaseImage(&iplimage);
  }

  cvReleaseVideoWriter(&writer);

  return 0;
}

static CvCapture* avi_to_capture(const std::string& aviname)
{
  CvCapture* const cap = cvCreateFileCapture(aviname.c_str());
  if (cap == NULL) return NULL;
  return cap;
}

static inline bool is_file(const std::string& name)
{
  struct stat buf;
  return stat(name.c_str(), &buf) != -1;
}

static CvCapture* directory_to_capture(const std::string& dirname)
{
  // create a capture from a directory
  // assume filename: dirname/012345.jpg
  // may create a video dirname/all.avi

  std::string aviname = dirname + std::string("/all.avi");
  if (is_file(aviname) == false)
  {
    if (create_avi(dirname, aviname) == -1)
      return NULL;
  }

  return avi_to_capture(aviname);
}

int main(int ac, char** av)
{
  CvCapture* cap = directory_to_capture(av[1]);
  if (cap == NULL) return -1;

  slParams params;
  slCalib calib;
  readConfiguration("../conf/conf.xml", &params);
  runCameraCalibration(cap, &params, &calib);
  displayCamCalib(&calib);

  cvReleaseCapture(&cap);
  return 0;
}
