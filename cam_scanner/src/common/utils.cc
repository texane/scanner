#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <math.h>
#include <sys/stat.h>
#include <string>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "common/assert.hh"


// CvCapture helpers

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
  ASSERT_RETURN(iplimage != NULL, -1);

  CvSize imsize;
  imsize.width = iplimage->width;
  imsize.height = iplimage->height;
  cvReleaseImage(&iplimage);

  CvVideoWriter* writer = cvCreateVideoWriter
    (aviname.c_str(), CV_FOURCC('P','I','M','1'), 25, imsize, true);

  ASSERT_RETURN(writer != NULL, -1);

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
  ASSERT_RETURN(cap != NULL, NULL);
  return cap;
}

static inline bool is_file(const std::string& name)
{
  struct stat buf;
  return stat(name.c_str(), &buf) != -1;
}

CvCapture* directory_to_capture(const std::string& dirname)
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
