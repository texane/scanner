#ifndef COMMON_CAM_PARAMS_HH_INCLUDED
# define COMMON_CAM_PARAMS_HH_INCLUDED


#include <string>
#include <opencv/cv.h>


#if 0

typedef struct cam_intrinsics_param
{
  // camera intrinsic params, ie. independent of pose
  // the resulting matrix is really a combination of the
  // vectors and scalars below, so we include them here
  // for the sake of documentation.

  double fc[2];
  double cc[2];
  double alpha_c;
  double kc[5];
  double fc_error[2];
  double cc_error[2];
  double alpha_c_error;
  double kc_error[5];
  double nx;
  double ny;

} cam_intrinsics_param_t;

typedef struct cam_extrinsic_params
{
  // camera extrinsic parameters

  double omc[3];
  double tc[3];
  double omc_error[3];
  double tc_error[3];

} cam_extrinsic_t;

#endif // 0


typedef struct cam_params
{
  CvMat* intrinsic;
  CvMat* distortion;
  CvMat* extrinsic;

  // toremove
  CvMat* roth;
  CvMat* rotv;
  CvMat* transh;
  CvMat* transv;
  // toremove

} cam_params_t;


int cam_params_load(cam_params_t&, const std::string&);
int cam_params_load_ml(cam_params_t&, const std::string&, bool = true);
int cam_params_save(const cam_params_t&, const std::string&);
int cam_params_create(cam_params_t&);
int cam_params_release(cam_params_t&);


#endif // ! COMMON_CAM_PARAMS_HH_INCLUDED
