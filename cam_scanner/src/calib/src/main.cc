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
#include "common/utils.hh"


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
  readConfiguration("../../../conf/hercules/xml/config.xml", &params);
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
