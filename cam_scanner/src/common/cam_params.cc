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


#if REAL_TYPE_IS_DOUBLE
  static const int mat_type = CV_64FC1;
#else
  static const int mat_type = CV_32FC1;
#endif


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
  params.intrinsic = cvCreateMat(3, 3, mat_type);
  params.distortion = cvCreateMat(3, 3, mat_type);
  params.extrinsic = cvCreateMat(3, 3, mat_type);

  ASSERT_RETURN(params.intrinsic != NULL, -1);
  ASSERT_RETURN(params.distortion != NULL, -1);
  ASSERT_RETURN(params.extrinsic != NULL, -1);

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

  // fixme: is it done by cvReleaseMat
  params.intrinsic = NULL;
  params.distortion = NULL;
  params.extrinsic = NULL;

  return 0;
}


#if 1 // UNIT

#include <stdio.h>

static const char* filename = "/tmp/cam_params.xml";

__attribute__((unused)) static int do_write(void)
{
  int error;
  cam_params_t params;

  error = cam_params_create(params);
  ASSERT_RETURN(error == 0, -1);

  CvMat* mat = params.intrinsic;

 redo_fill:
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      CV_MAT_ELEM(*mat, real_type, i, j) = 42;

  if (mat != params.extrinsic)
  {
    if (mat == params.intrinsic) mat = params.distortion;
    else if (mat == params.distortion) mat = params.extrinsic;
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

 redo_read:
  for (unsigned int i = 0; i < 3; ++i)
    for (unsigned int j = 0; j < 3; ++j)
      if (CV_MAT_ELEM(*mat, real_type, i, j) != 42)
      {
	printf("invalid\n");
	error = -1;
	mat = params.extrinsic;
	break ;
      }

  if (mat != params.extrinsic)
  {
    if (mat == params.intrinsic) mat = params.distortion;
    else if (mat == params.distortion) mat = params.extrinsic;
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
