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

  for (unsigned int i = 0; true; ++i)
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

static int allocate_calib(slCalib& sl_calib, const slParams& sl_params)
{
  // return -1 on error

  const int cam_nelems = sl_params.cam_w * sl_params.cam_h;
  const int proj_nelems = sl_params.proj_w * sl_params.proj_h;

  sl_calib.cam_intrinsic_calib = false;
  sl_calib.proj_intrinsic_calib = false;
  sl_calib.procam_extrinsic_calib = false;
  sl_calib.cam_intrinsic = cvCreateMat(3,3,CV_32FC1);
  sl_calib.cam_distortion = cvCreateMat(5,1,CV_32FC1);
  sl_calib.cam_extrinsic = cvCreateMat(2, 3, CV_32FC1);
  sl_calib.proj_intrinsic = cvCreateMat(3, 3, CV_32FC1);
  sl_calib.proj_distortion = cvCreateMat(5, 1, CV_32FC1);
  sl_calib.proj_extrinsic = cvCreateMat(2, 3, CV_32FC1);
  sl_calib.cam_center = cvCreateMat(3, 1, CV_32FC1);
  sl_calib.proj_center = cvCreateMat(3, 1, CV_32FC1);
  sl_calib.cam_rays = cvCreateMat(3, cam_nelems, CV_32FC1);
  sl_calib.proj_rays = cvCreateMat(3, proj_nelems, CV_32FC1);
  sl_calib.proj_column_planes = cvCreateMat(sl_params.proj_w, 4, CV_32FC1);
  sl_calib.proj_row_planes = cvCreateMat(sl_params.proj_h, 4, CV_32FC1);

  // initialize background model
  sl_calib.background_depth_map = cvCreateMat(sl_params.cam_h, sl_params.cam_w, CV_32FC1);
  sl_calib.background_image = cvCreateImage(cvSize(sl_params.cam_w, sl_params.cam_h), IPL_DEPTH_8U, 3);
  sl_calib.background_mask = cvCreateImage(cvSize(sl_params.cam_w, sl_params.cam_h), IPL_DEPTH_8U, 1);
  cvSet(sl_calib.background_depth_map, cvScalar(FLT_MAX));
  cvZero(sl_calib.background_image);
  cvSet(sl_calib.background_mask, cvScalar(255));

  return 0;
} 

int main(int ac, char** av)
{
  slParams params;
  slCalib calib;
  readConfiguration("../conf/conf.xml", &params);
  allocate_calib(calib, params);

#if 1
  // camera calibration
  {
    CvCapture* cap = directory_to_capture(av[1]);
    if (cap == NULL) return -1;
    runCameraCalibration(cap, &params, &calib);
    cvReleaseCapture(&cap);
  }
#endif

#if 0
  // projector and camera calibration
  {
    CvCapture* cap = directory_to_capture(av[1]);
    if (cap == NULL) return -1;
    runProjectorCalibration(cap, &params, &calib, true);
    cvReleaseCapture(&cap);
  }
#endif

#if 0
  // extrinsic calibration
  {
    CvCapture* cap = directory_to_capture(av[1]);
    if (cap == NULL) return -1;
    runProCamExtrinsicCalibration(cap, &params, &calib);
    cvReleaseCapture(&cap);
  }
#endif

  return 0;
}
