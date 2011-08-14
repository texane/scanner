#include <stdlib.h>
#include <string>
#include <opencv/cv.h>
#include "common/real_type.hh"
#include "common/assert.hh"
#include "common/cam_params.hh"


// cam_params.xml layout
// <params>
// <intrinsic> matrix </intrinsic>
// <distortion> matrix </distortion>
// <extrinsic> matrix </extrinsic>
// </params>


int cam_params_load(cam_params_t& params, const std::string& filename)
{
  // assume error
  int error = -1;

  CvFileStorage* fs = cvOpenFileStorage(filename.c_str(), 0, CV_STORAGE_READ);
  ASSERT_RETURN(fs != NULL, -1);

  CvFileNode* const n = cvGetFileNodeByName(fs, 0, "params");
  ASSERT_GOTO(n != NULL, on_error);

  params.intrinsic = (CvMat*)cvReadByName(fs, n, "intrinsic");
  params.distortion = (CvMat*)cvReadByName(fs, n, "distortion");
  params.extrinsic = (CvMat*)cvReadByName(fs, n, "extrinsic");

  ASSERT_GOTO(params.intrinsic != NULL, on_error);
  ASSERT_GOTO(params.distortion != NULL, on_error);
  ASSERT_GOTO(params.extrinsic != NULL, on_error);

  // success
  error = 0;

 on_error:
  cvReleaseFileStorage(&fs);
  return error;
}


int cam_params_load_ml
(cam_params_t& params, const std::string& filename, bool is_lowres)
{
  // load params from old matlab version (mlShadowScan)
  // this is hardcoded for dat/man both resolutions. since
  // mlShadowScan does not save extrinsic parameters. on the
  // long term, this function should not be avail.

  int error;

  error = cam_params_create(params);
  ASSERT_RETURN(error == 0, -1);

  real_type fc[2];
  real_type cc[2];
  real_type kc[5];
  real_type r[3];
  real_type t[3];

  if (is_lowres == true)
  {
    // from mlShadowScan/data/calib-lr/Calib_Results.m
    // note: rotation and translation vectors were found
    // by printing omc_ext and Tc_ext in computeExtrinsic.m


    fc[0] = 1029.088963049301100;
    fc[1] = 1028.772845684173700;
    cc[0] = 287.368565122283710;
    cc[1] = 204.931308136716550;
    kc[0] = -0.148348422140938;
    kc[1] = 0.215129139753359;
    kc[2] = 0.004513111567607;
    kc[3] = 0.004877209469556;
    kc[4] = 0;
    r[0] =  2.0345345;
    r[1] = -0.0089891;
    r[2] = -0.0217502;
    t[0] = -326.24;
    t[1] = 222.80;
    t[2] = 1554.56;
  }
  else
  {
    // from mlShadowScan/data/calib/Calib_Results.m

    fc[0] = 2063.624094669945900;
    fc[1] = 2068.246294767563500;
    cc[0] = 564.284599933070470;
    cc[1] = 430.077119141454150;
    kc[0] = -0.189857971274649;
    kc[1] = 1.159545847481371;
    kc[2] = 0.006814823434558;
    kc[3] = 0.003184022300329;
    kc[4] = 0;
    r[0] =  2.0472535;
    r[1] = -0.0044281;
    r[2] = -0.0208858;
    t[0] = -317.12;
    t[1] = 207.65;
    t[2] = 1560.28;
  }

  // refer to byo3d.pdf,p.21 for intrinsic matrix information
  // refer to opencv doc on cvCalibrateCamera2
  CV_MAT_ELEM(*params.intrinsic, real_type, 0, 0) = fc[0];
  CV_MAT_ELEM(*params.intrinsic, real_type, 0, 1) = 0;
  CV_MAT_ELEM(*params.intrinsic, real_type, 0, 2) = cc[0];
  CV_MAT_ELEM(*params.intrinsic, real_type, 1, 0) = 0;
  CV_MAT_ELEM(*params.intrinsic, real_type, 1, 1) = fc[1];
  CV_MAT_ELEM(*params.intrinsic, real_type, 1, 2) = cc[1];
  CV_MAT_ELEM(*params.intrinsic, real_type, 2, 0) = 0;
  CV_MAT_ELEM(*params.intrinsic, real_type, 2, 1) = 0;
  CV_MAT_ELEM(*params.intrinsic, real_type, 2, 2) = 1;

  // distortion
  CV_MAT_ELEM(*params.distortion, real_type, 0, 0) = kc[0];
  CV_MAT_ELEM(*params.distortion, real_type, 1, 0) = kc[1];
  CV_MAT_ELEM(*params.distortion, real_type, 2, 0) = kc[2];
  CV_MAT_ELEM(*params.distortion, real_type, 3, 0) = kc[3];
  CV_MAT_ELEM(*params.distortion, real_type, 4, 0) = kc[4];

  // extrinsic refer to opencv cvFindExtrinsicCameraParams2
  // and cvStructuredLight/cvCalibrateProCam.cpp
  for (unsigned int i = 0; i < 3; ++i)
  {
    CV_MAT_ELEM(*params.extrinsic, real_type, 0, i) = r[i];
    CV_MAT_ELEM(*params.extrinsic, real_type, 1, i) = t[i];
  }

  // toremove
  // coeffs have been measured with lowres calibration data

  CV_MAT_ELEM(*params.transh, real_type, 0, 0) = -326.24;
  CV_MAT_ELEM(*params.transh, real_type, 1, 0) = 222.80;
  CV_MAT_ELEM(*params.transh, real_type, 2, 0) = 1554.56;

  CV_MAT_ELEM(*params.transv, real_type, 0, 0) = -325.598;
  CV_MAT_ELEM(*params.transv, real_type, 1, 0) = -43.396;
  CV_MAT_ELEM(*params.transv, real_type, 2, 0) = 1872.990;

  CV_MAT_ELEM(*params.roth, real_type, 0, 0) = 0.9998060;
  CV_MAT_ELEM(*params.roth, real_type, 0, 1) = 0.0031637;
  CV_MAT_ELEM(*params.roth, real_type, 0, 2) = -0.0194433;
  CV_MAT_ELEM(*params.roth, real_type, 1, 0) = -0.0159738;
  CV_MAT_ELEM(*params.roth, real_type, 1, 1) = -0.4473934;
  CV_MAT_ELEM(*params.roth, real_type, 1, 2) = -0.8941946;
  CV_MAT_ELEM(*params.roth, real_type, 2, 0) = -0.0115278;
  CV_MAT_ELEM(*params.roth, real_type, 2, 1) = 0.8943316;
  CV_MAT_ELEM(*params.roth, real_type, 2, 2) = -0.4472561;

  CV_MAT_ELEM(*params.rotv, real_type, 0, 0) = 0.9997963;
  CV_MAT_ELEM(*params.rotv, real_type, 0, 1) = -0.0199377;
  CV_MAT_ELEM(*params.rotv, real_type, 0, 2) = -0.0031347;
  CV_MAT_ELEM(*params.rotv, real_type, 1, 0) = -0.0163380;
  CV_MAT_ELEM(*params.rotv, real_type, 1, 1) = -0.8907100;
  CV_MAT_ELEM(*params.rotv, real_type, 1, 2) = 0.4542784;
  CV_MAT_ELEM(*params.rotv, real_type, 2, 0) = -0.0118494;
  CV_MAT_ELEM(*params.rotv, real_type, 2, 1) = -0.4541346;
  CV_MAT_ELEM(*params.rotv, real_type, 2, 2) = -0.8908543;

  // toremove

  return 0;
}


int cam_params_save(const cam_params_t& params, const std::string& filename)
{
  CvFileStorage* fs = cvOpenFileStorage(filename.c_str(), 0, CV_STORAGE_WRITE);
  ASSERT_RETURN(fs != NULL, -1);

  cvStartWriteStruct(fs, "params", CV_NODE_MAP);
  cvWrite(fs, "intrinsic", (void*)params.intrinsic, cvAttrList(0, 0));
  cvWrite(fs, "distortion", (void*)params.distortion, cvAttrList(0, 0));
  cvWrite(fs, "extrinsic", (void*)params.extrinsic, cvAttrList(0, 0));
  cvEndWriteStruct(fs);

  cvReleaseFileStorage(&fs);

  return 0;
}


int cam_params_create(cam_params_t& params)
{
  params.intrinsic = cvCreateMat(3, 3, real_typeid);
  params.distortion = cvCreateMat(5, 1, real_typeid);
  params.extrinsic = cvCreateMat(2, 3, real_typeid);

  ASSERT_RETURN(params.intrinsic != NULL, -1);
  ASSERT_RETURN(params.distortion != NULL, -1);
  ASSERT_RETURN(params.extrinsic != NULL, -1);

  // toremove

  params.transh = cvCreateMat(3, 1, real_typeid);
  params.transv = cvCreateMat(3, 1, real_typeid);
  params.roth = cvCreateMat(3, 3, real_typeid);
  params.rotv = cvCreateMat(3, 3, real_typeid);

  ASSERT_RETURN(params.transh != NULL, -1);
  ASSERT_RETURN(params.transv != NULL, -1);
  ASSERT_RETURN(params.roth != NULL, -1);
  ASSERT_RETURN(params.rotv != NULL, -1);

  // toremove

  return 0;
}


int cam_params_release(cam_params_t& params)
{
  ASSERT_RETURN(params.intrinsic != NULL, -1);
  ASSERT_RETURN(params.distortion != NULL, -1);
  ASSERT_RETURN(params.extrinsic != NULL, -1);

  cvReleaseMat(&params.intrinsic);
  cvReleaseMat(&params.distortion);
  cvReleaseMat(&params.extrinsic);

  // toremove

  ASSERT_RETURN(params.transh != NULL, -1);
  ASSERT_RETURN(params.transv != NULL, -1);
  ASSERT_RETURN(params.roth != NULL, -1);
  ASSERT_RETURN(params.rotv != NULL, -1);

  cvReleaseMat(&params.transh);
  cvReleaseMat(&params.transv);
  cvReleaseMat(&params.roth);
  cvReleaseMat(&params.rotv);

  // toremove

  return 0;
}


#if 0 // UNIT

#include <stdio.h>

static const char* filename = "/tmp/cam_params.xml";

__attribute__((unused)) static int do_write(void)
{
  int error;
  cam_params_t params;

  error = cam_params_create(params);
  ASSERT_RETURN(error == 0, -1);

  CvMat* mat = params.intrinsic;
  unsigned int rows = 3;
  unsigned int cols = 3;

 redo_fill:
  for (unsigned int i = 0; i < rows; ++i)
    for (unsigned int j = 0; j < cols; ++j)
      CV_MAT_ELEM(*mat, real_type, i, j) = 42;

  if (mat != params.extrinsic)
  {
    if (mat == params.intrinsic)
    {
      rows = 5;
      cols = 1;
      mat = params.distortion;
    }
    else if (mat == params.distortion)
    {
      rows = 2;
      cols = 3;
      mat = params.extrinsic;
    }
    goto redo_fill;
  }

  error = cam_params_save(params, filename);

  cam_params_release(params);

  return error;
}


__attribute__((unused)) static int do_read(void)
{
  int error;
  cam_params_t params;

  error = cam_params_load(params, filename);
  ASSERT_RETURN(error == 0, -1);

  CvMat* mat = params.intrinsic;
  unsigned int rows = 3;
  unsigned int cols = 3;

 redo_read:
  for (unsigned int i = 0; i < rows; ++i)
    for (unsigned int j = 0; j < cols; ++j)
      if (CV_MAT_ELEM(*mat, real_type, i, j) != 42)
      {
	printf("invalid\n");
	error = -1;
	mat = params.extrinsic;
	break ;
      }

  if (mat != params.extrinsic)
  {
    if (mat == params.intrinsic)
    {
      rows = 5;
      cols = 1;
      mat = params.distortion;
    }
    else if (mat == params.distortion)
    {
      rows = 2;
      cols = 3;
      mat = params.extrinsic;
    }
    goto redo_read;
  }

  cam_params_release(params);

  return error;
}

int main(int ac, char** av)
{
  return do_read();
}

#endif // UNIT
