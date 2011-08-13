#include <stdlib.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "common/assert.hh"
#include "common/utils.hh"
#include "common/real_type.hh"
#include "common/cam_params.hh"


// show an image or a matrix

__attribute__((unused))
static void show_image(const IplImage* image)
{
  static const char* name = "MatrixView";
  cvNamedWindow(name, 1);
  cvShowImage(name, image);
  cvWaitKey(0);
  cvDestroyWindow(name);
}

__attribute__((unused))
static void show_matrix(CvMat* mat)
{
  IplImage header;
  const IplImage* const image = cvGetImage(mat, &header);
  show_image(image);
}


#if 0 // TODO

// intersection routines
//
// notes on representations
// parametric equation: define a relation as a set of eq
// implicit equation: f(x, y) = 0
// explicit equation: f(x) = y
// lines and planes are described by a coeff vector w.
//
// line parametric form:
// p: q + u * v
//
// line explicit form
// w[0] * x + w[1] = w[2]
//
// line implicit form
// w[0] * x + w[1] * y + w[2] = 0
//
// plane explicit form:
// w[0] * x + w[1] * y + w[2] * z = w[3]
//
// plane implicit form:
// w[0] * x + w[1] * y + w[2] * z + w[3] = 0
//


// define vNT, a vector N sized vector of T typed elements.
typedef int[2] v2i_t;
typedef double[2] v2d_t;
typedef double[3] v3d_t;
typedef double[4] v4d_t;

static int intersect_line_line
(v2d_t* res, const v3d_t* la, const v3d_t* lb)
{
  // inertsect 2 co planar lines
  // la and lb the lines explicit forms
  // res the 2d resulting point
}

static int intersect_line_plane
(vec3d_t* res, const vec3_t* q, const vect3_t* v, const vect3_t* w)
{
  // intersect a line with a plane
  // res the resulting point in 3d coordinates
  // q a point of the line
  // v the line vector
  // w a plane in implicit form
}

#endif // TODO


#if 0 // fitting routines

static void fit_line(void);
static void fit_plane(void);

#endif // fitting routines


#if 0 // TODO

static double norm(const double* v, unsigned int n)
{
  double sum = 0;
  for (; n; --n, ++v) sum += (*v) * (*v);
  return sqrt(sum);
}

static void normalize
(double* v, const int* x, double fc, double cc, double kc, double alphac)
{
  // TOOLBOX_calib/normalize.m
  // TODO
}

static void pixel_to_ray
(double* v, const int* x, double fc, double cc, double kc, double alphac)
{
  // compute the camera coorrdinates of the ray starting
  // at the camera point and passing by the pixel x.
  //
  // v the resulting ray vector
  // x the pixel coordinates
  // fc, cc, kc, alphac the camera intrinsic parameters

  normalize(v, fc, cc, kc, alphac);

  v[0] = x[0];
  v[1] = x[1];
  v[2] = 1;
}

#endif // TODO

// video processing

static inline int rewind_capture(CvCapture* cap)
{
  return cvSetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES, 0);
}

static inline unsigned int get_capture_frame_count(CvCapture* cap)
{
  const double count = cvGetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES);
  return (unsigned int)count;
}

static CvSize get_capture_frame_size(CvCapture* cap)
{
  CvSize size;
  IplImage* const frame = cvQueryFrame(cap);
  size.width = 0;
  size.height = 0;
  if (frame == NULL) return size;
  size = cvGetSize(frame);
  return size;
}

static int estimate_shadow_thresholds(CvCapture* cap, CvMat*& thresholds)
{
  // estimate invdiviual pixels shadow threshold
  // algorithm:
  // min_values = matrix(-inf);
  // max_values = matrix(+inf);
  // foreach frame
  // . frame = rgb2gray(frame);
  // . compute_minmax(frame, min_values, max_values);
  // shadowValue = (min_values + max_values) / 2;

  int error = -1;

  IplImage* gray_image = NULL;
  CvMat* gray_mat = NULL;

  CvMat* minvals = NULL;
  CvMat* maxvals = NULL;

  const CvSize frame_size = get_capture_frame_size(cap);
  const unsigned int nrows = frame_size.height;
  const unsigned int ncols = frame_size.width;

  // create and initialize min max matrces
  // rgb2gray formula: 0.299 * R + 0.587 * G + 0.114 * B

  minvals = cvCreateMat(nrows, ncols, CV_32SC1);
  maxvals = cvCreateMat(nrows, ncols, CV_32SC1);
  ASSERT_GOTO(minvals && maxvals, on_error);

  for (unsigned int i = 0; i < nrows; ++i)
    for (unsigned int j = 0; j < ncols; ++j)
    {
      CV_MAT_ELEM(*minvals, int, i, j) = 255;
      CV_MAT_ELEM(*maxvals, int, i, j) = 0;
    }

  // foreach pixel of each frame, get minmax pixels

  while (1)
  {
    IplImage* const frame_image = cvQueryFrame(cap);
    if (frame_image == NULL) break ;

    // create if not yet created
    if (gray_image == NULL)
    {
      CvMat header;
      gray_image = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
      ASSERT_GOTO(gray_image, on_error);
      gray_mat = cvGetMat(gray_image, &header);
    }

    // cvCopyImage(cam_frame, cam_frame_2);
    cvCvtColor(frame_image, gray_image, CV_RGB2GRAY);

    // get per pixel minmax
    for (unsigned int i = 0; i < nrows; ++i)
      for (unsigned int j = 0; j < ncols; ++j)
      {
	const int gray_val = CV_MAT_ELEM(*gray_mat, unsigned char, i, j);

	if (CV_MAT_ELEM(*minvals, int, i, j) > gray_val)
	  CV_MAT_ELEM(*minvals, int, i, j) = gray_val;

	if (CV_MAT_ELEM(*maxvals, int, i, j) < gray_val)
	  CV_MAT_ELEM(*maxvals, int, i, j) = gray_val;
      }

    // dont release frame_image
  }

  // create and get threshold

  thresholds = cvCreateMat(nrows, ncols, CV_8UC1);
  ASSERT_GOTO(thresholds, on_error);

  for (unsigned int i = 0; i < nrows; ++i)
    for (unsigned int j = 0; j < ncols; ++j)
    {
      const int minval = CV_MAT_ELEM(*minvals, int, i, j);
      const int maxval = CV_MAT_ELEM(*maxvals, int, i, j);
      CV_MAT_ELEM(*thresholds, unsigned char, i, j) = (minval + maxval) / 2;
    }

  // success
  error = 0;

 on_error:
  if (minvals) cvReleaseMat(&minvals);
  if (maxvals) cvReleaseMat(&maxvals);
  if (gray_image) cvReleaseImage(&gray_image);

  if ((error == -1) && thresholds)
    cvReleaseMat(&thresholds);

  rewind_capture(cap);

  return error;
}

#if 0 // TODO
static int estimate_shadow_planes(CvCapture* cap)
{
  // estimate shadow plane parameters
  // algorithm:
  // foreach_frame:
  // . frame = rgb2gray(frame);
  // . estimate vline
  // . estimate hline

  rewind_capture(cap);

  return -1;
}
#endif // TODO

#if 0 // TODO
static int estimate_shadow_cross_times(CvCapture* cap)
{
  // estimate the per pixel shadow crossing time
  // where time is the frame index. a pixel is
  // considered entered by a shadow when its gray
  // intensity changes from non shadow to shadow

  rewind_capture(cap);

  return -1;
}
#endif // TODO


#if 0 // TODO
static void show_estimations(CvCapture* cap)
{
}
#endif // TODO


#if 0 // TODO
static void generate_depth_map(void)
{
  // foreach shadowed pixel
  // . generate ray equation with camera pos
  // . intersect with shadow plane
}
#endif // TODO


#if 0 // TODO
static void show_depth_map(CvMat* depth_map)
{
  // depth map a double matrix
}
#endif // TODO


#if 0 // TODO
static void generate_points
(scan_data_t& scan_data, std::list<v3d_t>& scan_points)
{
  // foreach pixel, generate the parametric ray equation
  // and inersect with the corresponding shadow plane
}
#endif // TODO


// scan main routine

static int do_scan(CvCapture* cap, const cam_params_t& params)
{
  CvMat* shadow_thresholds = NULL;
  CvSize frame_size;
  int error = -1;

  frame_size = get_capture_frame_size(cap);
  ASSERT_GOTO(frame_size.width, on_error);

  error = estimate_shadow_thresholds(cap, shadow_thresholds);
  ASSERT_GOTO(error == 0, on_error);

  // show_matrix(shadow_thresholds);

#if 0
  error = estimate_shadow_planes(cap, shadow_planes);
  ASSERT_GOTO(error == 0, on_error);

  error = estimate_shadow_cross_times(cap, shadow_xtimes);
  ASSERT_GOTO(error == 0, on_error);

  show_estimations(cap);

  generate_depth_map(scan_data);
  show_depth_map(depth_map);

  typedef struct scan_data;
  std::list<v3d_t> scan_points;
  generate_points(scan_data, scan_points);
#endif

  error = 0;

 on_error:
  if (shadow_thresholds != NULL) cvReleaseMat(&shadow_thresholds);

  return error;
}


#if 0 // TODO

// shadow plane lines evaluation
static void evaluate_shadow_vline(const rectangle& area)
{
  // area the area to look into for plane for a line
  // note that area comes from a previous user selection

  sub = substract(image, shadow_values, area);
  for (i = 0; i < rows(tmp); ++i)
    for (j = 1; j < cols(tmp); ++j)
    {
      // entering shadow
      if ((sub[i][j] < 0) && (sub[i][j - 1] >= 0))
	entering_cols.append(j);
      // leaving shadow
      else if ((sub[i][j] >= 0) && (sub[i][j - 1] < 0))
	leaving_cols.append(j);
    }
}

#endif // TODO


int main(int ac, char** av)
{
  // av[1] the data directory containting jpg sequence
  // av[2] the configuration directory

  cam_params_t params;
  int error = -1;
  CvCapture* cap = NULL;

  cap = directory_to_capture(av[1]);
  ASSERT_GOTO(cap, on_error);

  error = cam_params_load_ml(params, "fubar.xml");
  ASSERT_GOTO(error == 0, on_error);

  error = do_scan(cap, params);

 on_error:
  if (cap) cvReleaseCapture(&cap);
  cam_params_release(params);

  return error;
}
