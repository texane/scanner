#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <list>
#include <vector>
#include <string>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "common/assert.hh"
#include "common/utils.hh"
#include "common/real_type.hh"
#include "common/cam_params.hh"
#include "common/fixed_vector.hh"


// toremove
#define CONFIG_SKIP_COUNT 0
#define CONFIG_HERCULES 1

#if CONFIG_HERCULES

// frame range used for reconstruction
static const unsigned int first_rec_frame = 0;
static const unsigned int last_rec_frame = 21;

// the length along x (resp. y) axis between checkboards rectangles
static const real_type dx = 250;
static const real_type dy = 200;

#else

// frame range used for reconstruction
static const unsigned int first_rec_frame = 61;
static const unsigned int last_rec_frame = 151;

// the length along x (resp. y) axis between checkboards rectangles
static const real_type dx = 558.8;
static const real_type dy = 303.2125;

#endif

// clipping volume
static const real_type clip_x[2] = { 5, dx - 5 };
static const real_type clip_y[2] = { 5, dy - 5 };
static const real_type clip_z[2] = { 5, dy + 50 };

// shadows cannot be detected on dark areas. use minimum contrast
// to account for those regions.
static const unsigned int min_contrast = 50;


// real type fixed vectors

typedef fixed_vector<real_type, 2> real2;
typedef fixed_vector<real_type, 3> real3;
typedef fixed_vector<real_type, 4> real4;


// show an image or a matrix

__attribute__((unused))
static void show_image(const IplImage* image, const char* name)
{
  cvNamedWindow(name, 0);
  cvShowImage(name, image);
  cvWaitKey(0);
  cvDestroyWindow(name);
}

__attribute__((unused))
static void show_matrix(CvMat* mat)
{
  IplImage header;
  const IplImage* const image = cvGetImage(mat, &header);
  show_image(image, "MatrixView");
}


// matrix helpers

static inline int get_gray_elem(const CvMat& m, unsigned int i, unsigned int j)
{
  // get elem at i, j from a gray matrix (ie. uint8_t)
  return CV_MAT_ELEM(m, unsigned char, i, j);
}


// video processing

static inline int rewind_capture(CvCapture* cap)
{
  return cvSetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES, 0);
}

static inline int seek_capture(CvCapture* cap, unsigned int n)
{
  return cvSetCaptureProperty(cap, CV_CAP_PROP_POS_FRAMES, n);
}

static inline unsigned int get_capture_frame_count(CvCapture* cap)
{
  const double count = cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_COUNT);
  return (unsigned int)count;
}

static CvSize get_capture_frame_size(CvCapture* cap)
{
  CvSize size;
  size.width = (int)cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH);
  size.height = (int)cvGetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT);
  return size;
}

static inline int set_capture_frame_size(CvCapture* cap, CvSize size)
{
  cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_WIDTH, size.width);
  cvSetCaptureProperty(cap, CV_CAP_PROP_FRAME_HEIGHT, size.height);
  return 0;
}


// ask user for a set of points

typedef struct user_points
{
  // middle line
  CvPoint mline[2];

  // vertical and horizontal planes
  CvPoint vplane[2];
  CvPoint hplane[2];

  // vertical and horizontal corners
  CvPoint vcorner[4];
  CvPoint hcorner[4];

} user_points_t;

static void print_point_array(const CvPoint* p, unsigned int n)
{
  for (unsigned int i = 0; i < n; ++i, ++p)
    printf("(%d, %d)\n", p->x, p->y);
}

__attribute__((unused))
static void print_user_points(user_points_t& points)
{
  printf("middle_points:\n");
  print_point_array(points.mline, 2);

  printf("vplane_points:\n");
  print_point_array(points.vplane, 2);

  printf("hplane_points:\n");
  print_point_array(points.hplane, 2);

  printf("vcorner_points:\n");
  print_point_array(points.vcorner, 4);

  printf("hcorner_points:\n");
  print_point_array(points.hcorner, 4);
}

#if 0 // swept_plane scene configuration

__attribute__((unused))
static int get_static_user_points(user_points_t& points)
{
  // got from a previous run, lowres image

  points.mline[0].x = 106;
  points.mline[0].y = 226;
  points.mline[1].x = 439;
  points.mline[1].y = 222;

  points.vplane[0].x = 112;
  points.vplane[0].y = 3;
  points.vplane[1].x = 400;
  points.vplane[1].y = 144;

  points.hplane[0].x = 98;
  points.hplane[0].y = 333;
  points.hplane[1].x = 415;
  points.hplane[1].y = 378;

  points.vcorner[0].x = 110;
  points.vcorner[0].y = 181;
  points.vcorner[1].x = 417;
  points.vcorner[1].y = 178;
  points.vcorner[2].x = 422;
  points.vcorner[2].y = 15;
  points.vcorner[3].x = 94;
  points.vcorner[3].y = 23;

  points.hcorner[0].x = 74;
  points.hcorner[0].y = 352;
  points.hcorner[1].x = 442;
  points.hcorner[1].y = 351;
  points.hcorner[2].x = 421;
  points.hcorner[2].y = 249;
  points.hcorner[3].x = 106;
  points.hcorner[3].y = 254;

  return 0;
}

#elif CONFIG_HERCULES // hercules scene configuration

__attribute__((unused))
static int get_static_user_points(user_points_t& points)
{
  // got from a previous run, lowres image

  points.mline[0].x = 138;
  points.mline[0].y = 297;
  points.mline[1].x = 512;
  points.mline[1].y = 290;

  points.vplane[0].x = 243;
  points.vplane[0].y = 22;
  points.vplane[1].x = 447;
  points.vplane[1].y = 234;

  points.hplane[0].x = 172;
  points.hplane[0].y = 405;
  points.hplane[1].x = 467;
  points.hplane[1].y = 462;

  points.vcorner[0].x = 86;
  points.vcorner[0].y = 441;
  points.vcorner[1].x = 544;
  points.vcorner[1].y = 442;
  points.vcorner[2].x = 497;
  points.vcorner[2].y = 316;
  points.vcorner[3].x = 158;
  points.vcorner[3].y = 319;

  points.hcorner[0].x = 189;
  points.hcorner[0].y = 229;
  points.hcorner[1].x = 496;
  points.hcorner[1].y = 225;
  points.hcorner[2].x = 498;
  points.hcorner[2].y = 38;
  points.hcorner[3].x = 188;
  points.hcorner[3].y = 45;

  return 0;
}

#else // quickcam scene configuration

__attribute__((unused))
static int get_static_user_points(user_points_t& points)
{
  // got from a previous run, lowres image

  points.mline[0].x = 72;
  points.mline[0].y = 145;
  points.mline[1].x = 267;
  points.mline[1].y = 145;

  points.vplane[0].x = 140;
  points.vplane[0].y = 13;
  points.vplane[1].x = 213;
  points.vplane[1].y = 64;

  points.hplane[0].x = 122;
  points.hplane[0].y = 187;
  points.hplane[1].x = 213;
  points.hplane[1].y = 217;

  points.vcorner[0].x = 71;
  points.vcorner[0].y = 212;
  points.vcorner[1].x = 290;
  points.vcorner[1].y = 211;
  points.vcorner[2].x = 254;
  points.vcorner[2].y = 155;
  points.vcorner[3].x = 93;
  points.vcorner[3].y = 155;

  points.hcorner[0].x = 100;
  points.hcorner[0].y = 115;
  points.hcorner[1].x = 244;
  points.hcorner[1].y = 114;
  points.hcorner[2].x = 251;
  points.hcorner[2].y = 22;
  points.hcorner[3].x = 96;
  points.hcorner[3].y = 27;

  return 0;
}

#endif

typedef struct on_mouse_state
{
  IplImage* image;
  const char* name;
  user_points_t* points;
  unsigned int id;
} on_mouse_state_t;

static void on_mouse(int event, int x, int y, int flags, void* p)
{
  if (event != CV_EVENT_LBUTTONDOWN) return ;

  on_mouse_state_t* const state = (on_mouse_state_t*)p;
  IplImage* const image = state->image;
  user_points_t* const points = state->points;
  CvPoint* point;

  // which point
  if (state->id < 2)
  {
    point = points->mline + state->id;
  }
  else if (state->id < 4)
  {
    point = points->vplane + state->id - 2;
  }
  else if (state->id < 6)
  {
    point = points->hplane + state->id - 4;
  }
  else if (state->id < 10)
  {
    point = points->vcorner + state->id - 6;
  }
  else if (state->id < 14)
  {
    point = points->hcorner + state->id - 10;
  }
  else 
  {
    return ;
  }

  point->x = x;
  point->y = y;

  // draw the point
  static const CvScalar color = CV_RGB(0xff, 0x00, 0xff);
  cvCircle(image, *point, 3, color, 1);
  cvShowImage(state->name, image);

  ++state->id;
}

__attribute__((unused))
static int get_user_points(CvCapture* cap, user_points_t& points)
{
  // get the first frame for reference

  static const char* name = "UserPoints";

  IplImage* frame = NULL;
  IplImage* cloned_image = NULL;
  int error = -1;
  on_mouse_state_t state;

  rewind_capture(cap);

  frame = cvQueryFrame(cap);
  ASSERT_GOTO(frame, on_error);
  cloned_image = cvCloneImage(frame);
  ASSERT_GOTO(cloned_image, on_error);

  cvNamedWindow(name, 0);
  cvShowImage(name, cloned_image);
  state.id = 0;
  state.points = &points;
  state.image = cloned_image;
  state.name = name;
  cvSetMouseCallback(name, on_mouse, &state);

  while (state.id < 14)
  {
    const char c = cvWaitKey(0);
    if (c == 27)
    {
      error = -1;
      goto on_error;
    }
  }

  error = 0;

 on_error:
  if (cloned_image) cvReleaseImage(&cloned_image);
  cvDestroyWindow(name);

  return error;
}


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

static int intersect_line_line
(const real3& p, const real3& q, real2& r)
{
  // intersect 2 coplanar lines
  // p and q the two lines in explicit forms
  //rp the resulting intersection point, if any
  // express the 2 lines equations in the form y = ax + b
  // and solve a0 * x + b0 = a1 * x + b1

  ASSERT_RETURN(fabs(p[1]) > 0.001, -1);
  ASSERT_RETURN(fabs(q[1]) > 0.001, -1);

  const real_type a0 = -p[0] / p[1];
  const real_type a1 = -q[0] / q[1];

  ASSERT_RETURN(fabs(a0 - a1) > 0.001, -1);

  const real_type b0 = p[2] / p[1];
  const real_type b1 = q[2] / q[1];

  // assume r.size() >= 2
  r[0] = (b1 - b0) / (a0 - a1);
  r[1] = a0 * r[0] + b0;

  return 0;
}

static int intersect_line_plane
(const real3& q, const real3& v, const real4& w, real3& p)
{
  // given the line passing by the point q, spanning vector
  // v, and a plane w, find p the intersecting point.

  // inner products

  real_type dotq = 0;
  real_type dotv = 0;

  for (unsigned int i = 0; i < 3; ++i)
  {
    dotq += w[i] * q[i];
    dotv += w[i] * v[i];
  }

  // intersect
  
  const real_type depth = (w[3] - dotq) / dotv;
  for (unsigned int i = 0; i < 3; ++i)
    p[i] = q[i] + depth * v[i];

  return 0;
}


static int pixel_to_ray
(const real2& pixel, const cam_params_t& params, real3& ray)
{
  // compute the camera coorrdinates of the ray starting
  // at the camera point and passing by the given pixel.
  // ray is the normalized 3d vector.

  // todo: need optimization, dont undistort one pixel at a time

  int error = -1;
  real_type norm;
  CvScalar scalar;
  CvMat* src = NULL;
  CvMat* dst = NULL;

#if REAL_TYPE_IS_DOUBLE
  src = cvCreateMat(1, 1, CV_64FC2);
#else
  src = cvCreateMat(1, 1, CV_32FC2);
#endif
  ASSERT_GOTO(src, on_error);

  dst = cvCreateMat(1, 1, src->type);
  ASSERT_GOTO(dst, on_error);

#define CONFIG_ENABLE_UNDISTORT 0

#if CONFIG_ENABLE_UNDISTORT

  scalar.val[0] = pixel[0];
  scalar.val[1] = pixel[1];

  // undistort
  cvSet1D(src, 0, scalar);
  cvUndistortPoints(src, dst, params.intrinsic, params.distortion);
  scalar = cvGet1D(dst, 0);

//   real2 fc;
//   real2 cc;

//   fc[0] = CV_MAT_ELEM(*params.intrinsic, real_type, 0, 0);
//   fc[1] = CV_MAT_ELEM(*params.intrinsic, real_type, 1, 1);
//   cc[0] = CV_MAT_ELEM(*params.intrinsic, real_type, 0, 2);
//   cc[1] = CV_MAT_ELEM(*params.intrinsic, real_type, 1, 2);

//   scalar.val[0] = (scalar.val[0] - cc[0]) / fc[0];
//   scalar.val[1] = (scalar.val[1] - cc[1]) / fc[1];

#else // disable undistort

  real2 fc;
  real2 cc;

  fc[0] = CV_MAT_ELEM(*params.intrinsic, real_type, 0, 0);
  fc[1] = CV_MAT_ELEM(*params.intrinsic, real_type, 1, 1);
  cc[0] = CV_MAT_ELEM(*params.intrinsic, real_type, 0, 2);
  cc[1] = CV_MAT_ELEM(*params.intrinsic, real_type, 1, 2);

  scalar.val[0] = (pixel[0] - cc[0]) / fc[0];
  scalar.val[1] = (pixel[1] - cc[1]) / fc[1];

#endif // CONFIG_ENABLE_UNDISTORT

  // assign z == 1 and normalize
  scalar.val[2] = 1;

  norm = 0;
  for (unsigned int i = 0; i < 3; ++i)
    norm += scalar.val[i] * scalar.val[i];
  norm = sqrt(norm);

  for (unsigned int i = 0; i < 3; ++i)
    ray[i] = scalar.val[i] / norm;

  error = 0;

 on_error:
  if (src) cvReleaseMat(&src);
  if (dst) cvReleaseMat(&dst);
  return error;
}


static inline int pixel_to_ray
(const CvPoint& pixel, const cam_params_t& params, real3& ray)
{
  real2 tmp;
  tmp[0] = (real_type)pixel.x;
  tmp[1] = (real_type)pixel.y;
  return pixel_to_ray(tmp, params, ray);
}


// shadow threshold estimtation

static int estimate_shadow_thresholds
(
 CvCapture* cap,
 CvMat*& thresholds,
 CvMat*& contrasts
)
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

  CvMat header;

  const CvSize frame_size = get_capture_frame_size(cap);
  const unsigned int nrows = frame_size.height;
  const unsigned int ncols = frame_size.width;

  rewind_capture(cap);

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

    // create if does not yet exist
    if (gray_image == NULL)
    {
      gray_image = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
      ASSERT_GOTO(gray_image, on_error);
      gray_mat = cvGetMat(gray_image, &header);
    }

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

  // create threshold and contrast matrices

  thresholds = cvCreateMat(nrows, ncols, CV_8UC1);
  ASSERT_GOTO(thresholds, on_error);

  contrasts = cvCreateMat(nrows, ncols, CV_8UC1);
  ASSERT_GOTO(contrasts, on_error);

  for (unsigned int i = 0; i < nrows; ++i)
    for (unsigned int j = 0; j < ncols; ++j)
    {
      const int minval = CV_MAT_ELEM(*minvals, int, i, j);
      const int maxval = CV_MAT_ELEM(*maxvals, int, i, j);
#if CONFIG_HERCULES
      unsigned char tmp = (minval + maxval) / 2;
      if (tmp < (255 - 10)) tmp += 10;
      else tmp = 255;
      CV_MAT_ELEM(*thresholds, unsigned char, i, j) = tmp;
#else
      CV_MAT_ELEM(*thresholds, unsigned char, i, j) = (minval + maxval) / 2;
#endif
      CV_MAT_ELEM(*contrasts, unsigned char, i, j) = maxval - minval;
    }

  // success
  error = 0;

 on_error:
  if (minvals) cvReleaseMat(&minvals);
  if (maxvals) cvReleaseMat(&maxvals);
  if (gray_image) cvReleaseImage(&gray_image);

  if (error == -1)
  {
    if (thresholds) cvReleaseMat(&thresholds);
    if (contrasts) cvReleaseMat(&contrasts);
  }

  return error;
}

static int estimate_shadow_xtimes
(
 CvCapture* cap,
 const CvMat* thr_mat,
 const CvMat* contrast_mat,
 CvMat* xtime_mat[2]
)
{
  // estimate the per pixel shadow crossing times, where time is
  // (roughly) the frame index. a pixel is considered entered
  // (resp. left) by a shadow when its gray intensity changes
  // from non shadow to shadow (resp. from non shadow to shadow)
  // between the previous and current frame.

  static const real_type not_found = -1;

  IplImage* curr_image = NULL;
  IplImage* prev_image = NULL;
  CvMat* prev_mat = NULL;
  CvMat* curr_mat = NULL;
  CvMat prev_header;
  CvMat curr_header;
  int error = -1;

  // allocate images
  const CvSize frame_size = get_capture_frame_size(cap);
  const unsigned int nrows = frame_size.height;
  const unsigned int ncols = frame_size.width;

  curr_image = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
  ASSERT_GOTO(curr_image, on_error);
  prev_image = cvCreateImage(frame_size, IPL_DEPTH_8U, 1);
  ASSERT_GOTO(prev_image, on_error);

  // retrieve corresponding matrices
  prev_mat = cvGetMat(prev_image, &prev_header);
  curr_mat = cvGetMat(curr_image, &curr_header);

  // allocate and initialize xtime matrices
  xtime_mat[0] = cvCreateMat(nrows, ncols, real_typeid);
  ASSERT_GOTO(xtime_mat[0], on_error);
  xtime_mat[1] = cvCreateMat(nrows, ncols, real_typeid);
  ASSERT_GOTO(xtime_mat[1], on_error);
  for (unsigned int i = 0; i < nrows; ++i)
    for (unsigned int j = 0; j < ncols; ++j)
    {
      CV_MAT_ELEM(*xtime_mat[0], real_type, i, j) = not_found;
      CV_MAT_ELEM(*xtime_mat[1], real_type, i, j) = not_found;
    }

  rewind_capture(cap);

  for (unsigned int frame_index = 0; true; ++frame_index)
  {
    IplImage* const frame_image = cvQueryFrame(cap);
    if (frame_image == NULL) break ;

    // swap prev and current before overriding
    // fixme: copy could be avoided by swapping
    // pointers and inside matrices.
    cvCopy(curr_image, prev_image);

    cvCvtColor(frame_image, curr_image, CV_RGB2GRAY);

    // skip first pass
    if (frame_index == 0) continue ;

    // actual algorithm: detect entering and leaving times
    for (unsigned int i = 0; i < nrows; ++i)
      for (unsigned int j = 0; j < ncols; ++j)
      {
	// entering xtime not already found
	if (CV_MAT_ELEM(*xtime_mat[0], real_type, i, j) == not_found)
	{
	  const int prev_val = get_gray_elem(*prev_mat, i, j);
	  const int curr_val = get_gray_elem(*curr_mat, i, j);
	  const int thr_val = get_gray_elem(*thr_mat, i, j);

	  if ((prev_val >= thr_val) && (curr_val < thr_val))
	  {
	    const real_type xtime = (real_type)frame_index +
	      (real_type)(thr_val - prev_val) / (real_type)(curr_val - prev_val);
	    CV_MAT_ELEM(*xtime_mat[0], real_type, i, j) = xtime;
	  }
	}

	// leaving xtime not already found
	if (CV_MAT_ELEM(*xtime_mat[1], real_type, i, j) == not_found)
	{
	  const int prev_val = get_gray_elem(*prev_mat, i, j);
	  const int curr_val = get_gray_elem(*curr_mat, i, j);
	  const int thr_val = get_gray_elem(*thr_mat, i, j);

	  if ((prev_val < thr_val) && (curr_val >= thr_val))
	  {
	    const real_type xtime = (real_type)frame_index +
	      (real_type)(thr_val - prev_val) / (real_type)(curr_val - prev_val);
	    CV_MAT_ELEM(*xtime_mat[1], real_type, i, j) = xtime;
	  }
	}
      }
  }

  // apply minimum contrast filter
  for (unsigned int i = 0; i < nrows; ++i)
    for (unsigned int j = 0; j < ncols; ++j)
      for (unsigned int k = 0; k < 2; ++k)
	if (CV_MAT_ELEM(*xtime_mat[k], real_type, i, j) != not_found)
	{
	  const unsigned int contrast =
	    CV_MAT_ELEM(*contrast_mat, unsigned char, i, j);
	  if (contrast < min_contrast)
	    CV_MAT_ELEM(*xtime_mat[k], real_type, i, j) = not_found;
	}

#if 1 // plot (entering) shadow xtimes
  {
    IplImage* cloned_image = cvCloneImage(curr_image);
    ASSERT_GOTO(cloned_image, on_error);

    CvMat header;
    CvMat* const cloned_mat = cvGetMat(cloned_image, &header);

    const real_type max_value = (real_type)get_capture_frame_count(cap);

    for (unsigned int i = 0; i < nrows; ++i)
      for (unsigned int j = 0; j < ncols; ++j)
      {
	const real_type xtime_value =
	  CV_MAT_ELEM(*xtime_mat[0], real_type, i, j);

	real_type scaled_value = 0;
	if (xtime_value != not_found)
	  scaled_value = xtime_value / max_value * 255;

	CV_MAT_ELEM(*cloned_mat, unsigned char, i, j) = floor(scaled_value);
      }

    show_image(cloned_image, "EnteringShadowCrossTimes");
    cvReleaseImage(&cloned_image);
  }
#endif // plot shadow xtimes

  // success
  error = 0;

 on_error:
  if (curr_image) cvReleaseImage(&curr_image);
  if (prev_image) cvReleaseImage(&prev_image);
  return error;
}


__attribute__((unused)) static void draw_points
(IplImage* image, const std::list<CvPoint>& points, const CvScalar& color)
{
  std::list<CvPoint>::const_iterator pos = points.begin();
  std::list<CvPoint>::const_iterator end = points.end();
  for (; pos != end; ++pos) cvCircle(image, *pos, 1, color, -1);
}


static void get_shadow_points
(
 const CvMat* gray_mat,
 const CvMat* thr_mat,
 const CvRect& bbox,
 std::list<CvPoint> points[2]
)
{
  // get the entering and leaving shaded points (resp. points[0] and points[1])
  // in the bounding box defined by {first,last}_{row,col}. see comments for the
  // definition of entering and leaving points.

  for (int i = bbox.y; i < bbox.y + bbox.height; ++i)
    for (int j = bbox.x; j < bbox.x + bbox.width; ++j)
    {
      const int thr_val = get_gray_elem(*thr_mat, i, j);
      const int gray_val = get_gray_elem(*gray_mat, i, j);
      if (gray_val >= thr_val) continue ;

      const int lthr_val = get_gray_elem(*thr_mat, i, j - 1);
      const int rthr_val = get_gray_elem(*thr_mat, i, j + 1);
      const int lneigh_val = get_gray_elem(*gray_mat, i, j - 1);
      const int rneigh_val = get_gray_elem(*gray_mat, i, j + 1);

      CvPoint point;
      point.x = j;
      point.y = i;

      // current shaded, left not shaded: entering pixel
      if (lneigh_val >= lthr_val) points[0].push_back(point);

      // current shaded, right not shaded: leaving pixel
      if (rneigh_val >= rthr_val) points[1].push_back(point);
    }
}


#if 0
static bool remove_outlier
(std::list<CvPoint>& points, const real3& w)
{
  // y = mx + p;
  const real_type m = -w[0] / w[1];
  const real_type p = w[2] / w[1];

  // get the maximum error
  real_type err = 0;
  std::list<CvPoint>::iterator pos = points.begin();
  std::list<CvPoint>::iterator i = points.begin();
  for (; i != points.end(); ++i)
  {
    const real_type y = m * (real_type)i->x + p;
    const real_type e = y - i->y;
    const real_type ee = e * e;
    if (ee > err)
    {
      err = ee;
      pos = i;
    }
  }

  // error too small, done
  if (err < 16) return false;

  // delete the outlier
  points.erase(pos);

  return true;
}
#else

static int fit_line(const std::list<CvPoint>& points, real3& w);

static bool remove_outlier(std::list<CvPoint>& points)
{
  real_type min_sum = 42000000; // large enough for infinity
  real_type min_ee = 0;
  std::list<CvPoint>::iterator min_pos = points.begin();

  std::list<CvPoint>::iterator pos = points.begin();
  unsigned int i = 0;

  for (; pos != points.end(); ++pos, ++i)
  {
    // remove the current point by overwritting it with another
    CvPoint saved_point = *pos;
    std::list<CvPoint>::const_iterator copied_pos = points.begin();
    if (pos == points.begin()) ++copied_pos;
    *pos = *copied_pos;

    // lse line fitting
    real3 w;
    fit_line(points, w);

    // y = mx + p;
    const real_type m = -w[0] / w[1];
    const real_type p = w[2] / w[1];

    // compute the error sum
    real_type sum = 0;
    std::list<CvPoint>::iterator j = points.begin();
    for (; j != points.end(); ++j)
    {
      const real_type x = (real_type)j->x;
      const real_type y = m * x + p;
      const real_type e = (real_type)j->y - y;
      const real_type ee = e * e;
      sum += ee;
    }

    // restore the saved point
    *pos = saved_point;

    // save if it minimizes
    if (sum < min_sum)
    {
      const real_type y = m * (real_type)pos->x + p;
      const real_type e = (real_type)pos->y - y;
      min_ee = e * e;
      min_sum = sum;
      min_pos = pos;
    }
  }

  // error too small, done
  if (min_ee < 16) return false;

  // delete the outlier
  points.erase(min_pos);

  return true;
}
#endif


static int fit_line(const std::list<CvPoint>& points, real3& w)
{
  // find a line that best fits points. use least square error method.
  // w contains the resulting line explicit coefficients such that:
  // w[0] * x + w[1] * y = w[2]
  // http://mariotapilouw.blogspot.com/2011/04/least-square-using-opencv.html

  // undetermined, cannot fit
  if (points.size() < 2)
  {
    w[0] = 0;
    w[1] = 0;
    w[2] = 0;
    return -1;
  }

  CvMat* y = cvCreateMat(points.size(), 1, real_typeid);
  CvMat* x = cvCreateMat(points.size(), 2, real_typeid);
  CvMat* res = cvCreateMat(2, 1, real_typeid);

  ASSERT_RETURN(y, -1);
  ASSERT_RETURN(x, -1);
  ASSERT_RETURN(res, -1);

  // fill data
  unsigned int i = 0;
  std::list<CvPoint>::const_iterator pos = points.begin();
  std::list<CvPoint>::const_iterator end = points.end();
  for (; pos != end; ++i, ++pos)
  {
    CV_MAT_ELEM(*y, real_type, i, 0) = pos->y;
    CV_MAT_ELEM(*x, real_type, i, 0) = pos->x;
    CV_MAT_ELEM(*x, real_type, i, 1) = 1;
  }

  // solve and make explicit form
  cvSolve(x, y, res, CV_SVD);

  const real_type a = CV_MAT_ELEM(*res, real_type, 0, 0);
  const real_type b = CV_MAT_ELEM(*res, real_type, 1, 0);

  w[0] = a * -1;
  w[1] = 1;
  w[2] = b;

  cvReleaseMat(&y);
  cvReleaseMat(&x);
  cvReleaseMat(&res);

  return 0;
}


static int fit_line(const CvPoint points[2], real3& w)
{
  std::list<CvPoint> point_list;
  point_list.push_back(points[0]);
  point_list.push_back(points[1]);
  return fit_line(point_list, w);
}


static int fit_plane(const CvMat* points, real4& plane)
{
  // from cvStructuredLight/cvUtilProCam.cpp
  // points is a n x m matrix where m the dimensionality

  int error = -1;

  CvMat* centroid = NULL;
  CvMat* points2 = NULL;
  CvMat* A = NULL;
  CvMat* W = NULL;
  CvMat* V = NULL;

  // estimate geometric centroid

  const int nrows = points->rows;
  const int ncols = points->cols;
  const int type = points->type;

  centroid = cvCreateMat(1, ncols, type);
  ASSERT_GOTO(centroid, on_error);

  cvSet(centroid, cvScalar(0));
  for (int c = 0; c < ncols; ++c)
  {
    for (int r = 0; r < nrows; ++r)
    {
      CV_MAT_ELEM(*centroid, real_type, 0, c) +=
	CV_MAT_ELEM(*points, real_type, r, c);
    }

    CV_MAT_ELEM(*centroid, real_type, 0, c) /= nrows;
  }

  // subtract geometric centroid from each point

  points2 = cvCreateMat(nrows, ncols, type);
  ASSERT_GOTO(points2, on_error);

  for (int r = 0; r < nrows; ++r)
    for (int c = 0; c < ncols; ++c)
    {
      CV_MAT_ELEM(*points2, real_type, r, c) =
	CV_MAT_ELEM(*points, real_type, r, c) -
	CV_MAT_ELEM(*centroid, real_type, 0, c);
    }
	
  // evaluate SVD of covariance matrix

  A = cvCreateMat(ncols, ncols, type);
  ASSERT_GOTO(A, on_error);

  W = cvCreateMat(ncols, ncols, type);
  ASSERT_GOTO(W, on_error);

  V = cvCreateMat(ncols, ncols, type);
  ASSERT_GOTO(V, on_error);

  cvGEMM(points2, points, 1, NULL, 0, A, CV_GEMM_A_T); 
  cvSVD(A, W, NULL, V, CV_SVD_V_T);

  // assign plane coefficients by singular vector
  // corresponding to smallest singular value

  plane[ncols] = 0;
  for (int c = 0; c < ncols; ++c)
  {
    plane[c] = CV_MAT_ELEM(*V, real_type, ncols - 1, c);
    plane[ncols] += plane[c] * CV_MAT_ELEM(*centroid, real_type, 0, c);
  }

  error = 0;

 on_error:

  if (centroid) cvReleaseMat(&centroid);
  if (points2) cvReleaseMat(&points2);
  if (A) cvReleaseMat(&A);
  if (W) cvReleaseMat(&W);
  if (V) cvReleaseMat(&V);

  return error;
}


__attribute__((unused))
static int draw_line(IplImage* image, const real3& w, const CvScalar& color)
{
  // draw the line whose explicit form coefficients are in w[3]
  // x = (w[2] - w[1] * y) / w[0];
  // y = (w[2] - w[0] * x) / w[1];
  // to get the form y = ax + b:
  // a = - w[0] / w[1]
  // b = + w[2] / w[1]

  CvPoint points[2];
  unsigned int i = 0;

  ASSERT_RETURN(fabs(w[1]) >= 0.0001, -1);

  const real_type a = -w[0] / w[1];
  const real_type b =  w[2] / w[1];

  // 0 <= b < image->height, left intersection
  if ((b >= 0) && (b < image->height))
  {
    ASSERT_RETURN(i < 2, -1);
    points[i].x = 0;
    points[i].y = (int)b;
    ++i;
  }

  // 0 <= -b / a < image->width, top intersection
  if (fabs(a) > 0.0001)
  {
    const real_type fu = -b / a;
    if ((fu >= 0) && (fu < image->width))
    {
      ASSERT_RETURN(i < 2, -1);
      points[i].x = floor(fu);
      points[i].y = 0;
      ++i;
    }
  }

  // 0 <= (width - 1) * a + b < height, right intersection
  const real_type bar = (image->width - 1) * a + b;
  if ((bar >= 0) && (bar < image->height))
  {
    ASSERT_RETURN(i < 2, -1);
    points[i].x = image->width - 1;
    points[i].y = floor(bar);
    ++i;
  }

  // 0 <= (height - 1 - b) / a < width, bottom intersection
  if (fabs(a) > 0.0001)
  {
    const real_type baz = (image->height - 1 - b) / a;

    if ((baz >= 0) && (baz < image->width))
    {
      ASSERT_RETURN(i < 2, -1);
      points[i].x = floor(baz);
      points[i].y = image->height - 1;
      ++i;
    }
  }

  ASSERT_RETURN(i == 2, -1);

  cvLine(image, points[0], points[1], color);

  return 0;
}


typedef struct line_eqs
{
  // line explicit equation coefficients

  std::vector<real3> venter;
  std::vector<real3> vleave;
  std::vector<real3> henter;
  std::vector<real3> hleave;

  // todo: bitmap
  std::vector<bool> is_valid;

  real3 middle;
  real3 upper;
  real3 lower;

} line_eqs_t;


typedef struct plane_eqs
{
  // plane explicit equation coefficients

  real4 vplane;
  real4 hplane;

  std::vector<real4> shadow_planes;
  
  // todo: bitmap
  std::vector<bool> is_valid;

} plane_eqs_t;


__attribute__((unused))
static int check_shadow_lines
(const real3& hline, const real3& vline, const real3& mline)
{
  // entering (resp. leaving) lines should intersect on the middle line
  // w the lines explicit form coefficients

  // solve intersection

  real2 point;
  if (intersect_line_line(vline, hline, point) == -1) return -1;

  // compute the corresponding point on middle line
  const real_type a = -mline[0] / mline[1];
  const real_type b = mline[2] / mline[1];
  const real_type y = a * point[0] + b;

  // invalid if more than 10 pixels far
  return fabs(point[1] - y) >= 10 ? -1 : 0;
}

static int estimate_shadow_lines
(
 CvCapture* cap,
 const CvMat* thr_mat,
 const user_points_t& user_points,
 line_eqs_t& line_eqs
)
{
  // estimate shadow plane parameters
  // foreach_frame, estimate {h,v}line

  IplImage* gray_image = NULL;
  CvMat* gray_mat = NULL;
  CvMat header;

  // {vertical,horizontal}_{enter,leave} points
  std::list<CvPoint> shadow_points[4];

  int error = -1;

  // compute bounding boxes according to user inputs
  CvRect vbox;
  vbox.x = user_points.vplane[0].x + 1;
  vbox.width = (user_points.vplane[1].x - 1) - vbox.x;
  vbox.y = user_points.vplane[0].y + 1;
  vbox.height = (user_points.vplane[1].y - 1) - vbox.y;

  CvRect hbox;
  hbox.x = user_points.hplane[0].x + 1;
  hbox.width = (user_points.hplane[1].x - 1) - hbox.x;
  hbox.y = user_points.hplane[0].y + 1;
  hbox.height = (user_points.hplane[1].y - 1) - hbox.y;

  unsigned int frame_index = CONFIG_SKIP_COUNT;
  seek_capture(cap, frame_index);

  // resize the line vectors
  const unsigned int frame_count = get_capture_frame_count(cap);
  line_eqs.venter.resize(frame_count);
  line_eqs.vleave.resize(frame_count);
  line_eqs.henter.resize(frame_count);
  line_eqs.hleave.resize(frame_count);
  line_eqs.is_valid.resize(frame_count);

  while (1)
  {
    // printf("frame_index == %u\n", frame_index);

    IplImage* const frame_image = cvQueryFrame(cap);
    if (frame_image == NULL) break ;

    // create if does not yet exist
    if (gray_image == NULL)
    {
      // todo: move outside the loop
      gray_image = cvCreateImage(cvGetSize(frame_image), IPL_DEPTH_8U, 1);
      ASSERT_GOTO(gray_image, on_error);
      gray_mat = cvGetMat(gray_image, &header);
    }

    cvCvtColor(frame_image, gray_image, CV_RGB2GRAY);

    // find pixels whose value is below the shadow thresold. if the
    // same row left (resp. right) adjacent pixel is not a shadowed
    // one, the the pixel belongs to the entering (resp. leaving)
    // shadow line.
    // limit the search to the bounding boxes indicated by the user
    // input.

    for (unsigned int i = 0; i < 4; ++i) shadow_points[i].clear();
    get_shadow_points(gray_mat, thr_mat, vbox, shadow_points + 0);
    get_shadow_points(gray_mat, thr_mat, hbox, shadow_points + 2);

    // check there are at least 2 points per lines
    for (unsigned int i = 0; i < 4; ++i)
    {
      if (shadow_points[i].size() < 2)
      {
	line_eqs.is_valid[i] = 0;
	goto next_frame;
      }
    }

    line_eqs.is_valid[frame_index] = 1;

    // get lines equation via lse fitting method
    // remove outliers

    {
      for (unsigned int i = 0; i < 4; ++i)
      {
	real3 w;

	// work on a shadow point copy
	std::list<CvPoint>& points = shadow_points[i];

	while (1)
	{
	  if (points.size() <= 2) break ;

	  // if (remove_outlier(points, w) == false) break ;
	  if (remove_outlier(points) == false) break ;

	  fit_line(points, w);
	}

	// copy the line equation coeffs
	if (i == 0) line_eqs.venter[frame_index] = w;
	else if (i == 1) line_eqs.vleave[frame_index] = w;
	else if (i == 2) line_eqs.henter[frame_index] = w;
	else if (i == 3) line_eqs.hleave[frame_index] = w;
      }
    }

#if 1 // plot the lines
    {
      static const CvScalar colors[] =
      {
	CV_RGB(0xff, 0x00, 0x00),
	CV_RGB(0x00, 0xff, 0x00),
	CV_RGB(0x00, 0x00, 0xff),
	CV_RGB(0xff, 0x00, 0xff),
	CV_RGB(0x00, 0x00, 0x00)
      };

      IplImage* cloned_image = cvCloneImage(frame_image);
      ASSERT_GOTO(cloned_image, on_error);

      for (unsigned int i = 0; i < 4; ++i)
	draw_points(cloned_image, shadow_points[i], colors[i]);

      draw_line(cloned_image, line_eqs.venter[frame_index], colors[0]);
      draw_line(cloned_image, line_eqs.vleave[frame_index], colors[1]);
      draw_line(cloned_image, line_eqs.henter[frame_index], colors[2]);
      draw_line(cloned_image, line_eqs.hleave[frame_index], colors[3]);

      const int res = check_shadow_lines
	(line_eqs.venter[frame_index], line_eqs.henter[frame_index], line_eqs.middle);
      if (res == -1)
      {
	printf("invalid shadow lines\n");
	draw_line(cloned_image, line_eqs.middle, colors[4]);
      }

      show_image(cloned_image, "ShadowLines");
      cvReleaseImage(&cloned_image);
    }
#endif // plot the line

  next_frame:
    ++frame_index;
  }

  error = 0;

 on_error:
  if (gray_image) cvReleaseImage(&gray_image);
  return error;
}


static int estimate_reference_planes
(CvCapture* cap, const cam_params_t& params, plane_eqs_t& plane_eqs)
{
  // working matrices

  CvMat* points = NULL;
  CvMat* tsub = NULL;
  CvMat* tmp = NULL;
  CvMat* transposed = NULL;

  // resulting error

  int error = -1;

  // create the four reference points matrix

  points = cvCreateMat(4, 3, real_typeid);
  ASSERT_GOTO(points, on_error);

  CV_MAT_ELEM(*points, real_type, 0, 0) = 0;
  CV_MAT_ELEM(*points, real_type, 0, 1) = 0;
  CV_MAT_ELEM(*points, real_type, 0, 2) = 0;

  CV_MAT_ELEM(*points, real_type, 1, 0) = dx;
  CV_MAT_ELEM(*points, real_type, 1, 1) = 0;
  CV_MAT_ELEM(*points, real_type, 1, 2) = 0;

  CV_MAT_ELEM(*points, real_type, 2, 0) = dx;
  CV_MAT_ELEM(*points, real_type, 2, 1) = dy;
  CV_MAT_ELEM(*points, real_type, 2, 2) = 0;

  CV_MAT_ELEM(*points, real_type, 3, 0) = 0;
  CV_MAT_ELEM(*points, real_type, 3, 1) = dy;
  CV_MAT_ELEM(*points, real_type, 3, 2) = 0;

  error = fit_plane(points, plane_eqs.hplane);
  ASSERT_GOTO(error == 0, on_error);
  error = -1;

  // x = roth' * (rotv * x + repmat(transv - transh, 1, size(x, 2)));

  // tmp = transv - transh

  tmp = cvCreateMat
    (params.transv->rows, params.transv->cols, params.transv->type);
  ASSERT_GOTO(tmp, on_error);
  cvSub(params.transv, params.transh, tmp);

  // tsub = repmat(tmp, 1, size(x, 2));
  // tsub is the translation vector repeated once per point. thus
  // column count is points->row.

  tsub = cvCreateMat(params.transv->rows, points->rows, params.transv->type);
  ASSERT_GOTO(tsub, on_error);

  for (int i = 0; i < tsub->rows; ++i)
    for (int j = 0; j < tsub->cols; ++j)
      CV_MAT_ELEM(*tsub, real_type, i, j) = CV_MAT_ELEM(*tmp, real_type, i, 0);

  cvReleaseMat(&tmp);

  // tmp = rotv * x + tsub;
  // tmp is a m x n matrix where m the dimensionnality and n
  // the point count. note that points have to be transposed
  // here. this is due to fit_plane requiring a differently
  // ordered matrix.

  tmp = cvCreateMat(tsub->rows, tsub->cols, points->type);
  ASSERT_GOTO(tmp, on_error);
  cvGEMM(params.rotv, points, 1, tsub, 1, tmp, CV_GEMM_B_T);

  // x = roth' * tmp;
  // note the points need to be transposed, refer to the above
  // comment, we use the transposed temporary matrix.

  transposed = cvCreateMat(points->cols, points->rows, points->type);
  ASSERT_GOTO(transposed, on_error);

  cvGEMM(params.roth, tmp, 1, NULL, 0, transposed, CV_GEMM_A_T);

  // recast to points
  cvTranspose(transposed, points);

  // fit the plane
  error = fit_plane(points, plane_eqs.vplane);
  ASSERT_GOTO(error == 0, on_error);
  error = -1;

  error = 0;

 on_error:
  if (points) cvReleaseMat(&points);
  if (tmp) cvReleaseMat(&tmp);
  if (tsub) cvReleaseMat(&tsub);
  if (transposed) cvReleaseMat(&transposed);

  return error;
}


static void real3_to_mat(const real3& r, CvMat* m)
{
  CV_MAT_ELEM(*m, real_type, 0, 0) = r[0];
  CV_MAT_ELEM(*m, real_type, 1, 0) = r[1];
  CV_MAT_ELEM(*m, real_type, 2, 0) = r[2];
}


static void mat_to_real3(const CvMat* m, real3& r)
{
  r[0] = CV_MAT_ELEM(*m, real_type, 0, 0);
  r[1] = CV_MAT_ELEM(*m, real_type, 1, 0);
  r[2] = CV_MAT_ELEM(*m, real_type, 2, 0);
}


template<typename type> static void zero(type& v)
{
  for (unsigned int i = 0; i < type::size; ++i) v[i] = 0;
}


template<typename type> static real_type norm(const type& v)
{
  real_type sum = 0;
  for (unsigned int i = 0; i < type::size; ++i) sum += v[i] * v[i];
  return sqrt(sum);
}


template<typename type> static type sub(const type& a, const type& b)
{
  type res;
  for (unsigned int i = 0; i < type::size; ++i) res[i] = a[i] - b[i];
  return res;
}


template<typename type> static type add(const type& a, const type& b)
{
  type res;
  for (unsigned int i = 0; i < type::size; ++i) res[i] = a[i] + b[i];
  return res;
}


template<typename type> static type div(const type& a, real_type b)
{
  type res;
  for (unsigned int i = 0; i < type::size; ++i) res[i] = a[i] / b;
  return res;
}


template<typename type> static real_type dot(const type& a, const type& b)
{
  // compute the dot product
  real_type sum = 0;
  for (unsigned int i = 0; i < type::size; ++i) sum += a[i] * b[i];
  return sum;
}


template<typename type> static type cross(const type& a, const type& b)
{
  // compute the cross product
  real3 p;
  p[0] = a[1] * b[2] - b[1] * a[2];
  p[1] = a[2] * b[0] - b[2] * a[0];
  p[2] = a[0] * b[1] - b[0] * a[1];
  return p;
}


template<typename type> static const char* to_string(const type& v)
{
  static std::string s;

  char buf[64];

  s.clear();

  for (unsigned int i = 0; i < type::size; ++i)
  {
    sprintf(buf, " %lf", v[i]);
    s.append(buf);
  }

  return s.c_str();
}


static int compute_camera_center(const cam_params_t& params, real3& c)
{
  // compute camera center from params

  CvMat* mat = cvCreateMat(3, 1, real_typeid);
  ASSERT_RETURN(mat, -1);

  // c = -roth' * transh;
  cvGEMM(params.roth, params.transh, -1, NULL, 0, mat, CV_GEMM_A_T);
  mat_to_real3(mat, c);

  cvReleaseMat(&mat);

  return 0;
}


static int estimate_shadow_planes
(
 CvCapture* cap,
 const cam_params_t& params,
 const line_eqs_t& line_eqs,
 plane_eqs_t& plane_eqs
)
{
  int error = -1;

  const unsigned int frame_count = get_capture_frame_count(cap);

  unsigned int frame_index = CONFIG_SKIP_COUNT;

  // line line intersection point
  real2 ll_point;

  // shadow plane 4 points
  real3 plane_points[4];

  real3 ray;

  // ray working matrix
  CvMat* ray_mat = NULL;

  // rotation working matrix
  CvMat* rot_mat = NULL;

  // camera center
  real3 c;

  // working vectors
  real3 v;
  real3 vv;
  real3 hv;
  real3 xv;

  // uninitialized warnings
  for (unsigned int i = 0; i < 4; ++i)
    zero(plane_points[i]);
  zero(ll_point);

  // allocate matrices
  ray_mat = cvCreateMat(3, 1, real_typeid);
  ASSERT_GOTO(ray_mat, on_error);

  rot_mat = cvCreateMat(3, 1, real_typeid);
  ASSERT_GOTO(rot_mat, on_error);

  // compute the camera center
  compute_camera_center(params, c);

  // foreach frame
  // determine true position of the lines
  // compute the shadow planes

  seek_capture(cap, frame_index);

  // allocate planes and validity map
  plane_eqs.shadow_planes.resize(frame_count);
  plane_eqs.is_valid.resize(frame_count);

  while (1)
  {
    IplImage* const frame_image = cvQueryFrame(cap);
    if (frame_image == NULL) break ;

    const real3& venter = line_eqs.venter[frame_index];
    const real3& henter = line_eqs.henter[frame_index];
    real4& plane = plane_eqs.shadow_planes[frame_index];

    // assume invalid
    plane_eqs.is_valid[frame_index] = 0;

    // skip if line equations not found
    if (line_eqs.is_valid[frame_index] == 0)
    {
      plane_eqs.is_valid[frame_index] = 0;
      goto next_frame;
    }

#if 1 // reject all other frames
    if (frame_index < first_rec_frame || frame_index > last_rec_frame)
      goto next_frame;
#endif

    plane_eqs.is_valid[frame_index] = 1;

    // get 2 points pairs from both the (entering) vertical and
    // horizontal lines. this is done using middle, lower and upper
    // intersection to derive a ray. then, use intersection between
    // and vplane (resp. hplane) to get points. these 4 points fully
    // determine the shadow plane.

    intersect_line_line(venter, line_eqs.middle, ll_point);
    pixel_to_ray(ll_point, params, ray);
    real3_to_mat(ray, ray_mat);
    cvGEMM(params.roth, ray_mat, 1, NULL, 0, rot_mat, CV_GEMM_A_T);
    mat_to_real3(rot_mat, ray);
    intersect_line_plane(c, ray, plane_eqs.vplane, plane_points[0]);

    intersect_line_line(venter, line_eqs.upper, ll_point);
    pixel_to_ray(ll_point, params, ray);
    real3_to_mat(ray, ray_mat);
    cvGEMM(params.roth, ray_mat, 1, NULL, 0, rot_mat, CV_GEMM_A_T);
    mat_to_real3(rot_mat, ray);
    intersect_line_plane(c, ray, plane_eqs.vplane, plane_points[1]);

    intersect_line_line(henter, line_eqs.middle, ll_point);
    pixel_to_ray(ll_point, params, ray);
    real3_to_mat(ray, ray_mat);
    cvGEMM(params.roth, ray_mat, 1, NULL, 0, rot_mat, CV_GEMM_A_T);
    mat_to_real3(rot_mat, ray);
    intersect_line_plane(c, ray, plane_eqs.hplane, plane_points[2]);

    intersect_line_line(henter, line_eqs.lower, ll_point);
    pixel_to_ray(ll_point, params, ray);
    real3_to_mat(ray, ray_mat);
    cvGEMM(params.roth, ray_mat, 1, NULL, 0, rot_mat, CV_GEMM_A_T);
    mat_to_real3(rot_mat, ray);
    intersect_line_plane(c, ray, plane_eqs.hplane, plane_points[3]);

    // compute the entering plane params from plane_points

    // vertical and horizontal normalized vectors
    v = sub(plane_points[1], plane_points[0]);
    vv = div(v, norm(v));
    v = sub(plane_points[3], plane_points[2]);
    hv = div(v, norm(v));

    // normalized cross product
    v = cross(vv, hv);
    xv = div(v, norm(v));

    // store the explicit plane equation
    for (unsigned int i = 0; i < 3; ++i) plane[i] = xv[i];
    v = add(plane_points[0], plane_points[2]);
    plane[3] = dot(xv, v) / 2;

#if 0 // debugging
    {
      printf("frame_index: %u\n", frame_index);
      for (unsigned int i = 0; i < 4; ++i)
	printf("%s\n", to_string(plane_points[i]));
      printf("plane_eq: %s\n", to_string(plane));
      printf("--\n");
    }
#endif // debugging

    // next frame
  next_frame:
    ++frame_index;
  }

  // success
  error = 0;

 on_error:
  if (rot_mat) cvReleaseMat(&rot_mat);
  if (ray_mat) cvReleaseMat(&ray_mat);

  return error;
}


// 3d point reconstruction

static inline real2 make_real2(real_type x, real_type y)
{
  real2 v;
  v[0] = x;
  v[1] = y;
  return v;
}

static inline real3 make_real3(real_type x, real_type y, real_type z)
{
  real3 v;
  v[0] = x;
  v[1] = y;
  v[2] = z;
  return v;
}

static int reconstruct_points
(
 const cam_params_t& params,
 const plane_eqs_t& plane_eqs,
 CvMat* xtimes[2],
 std::list<real3>& points
)
{
  // reconstruct the 3 dimensionnal points using
  // the previously derived shadow crossing times
  // and planes.
  // xtimes the entering and leaving crossing times
  // points the resulting point list

  static const real_type not_found = -1;

  int error = -1;

  // keep a reference on the entering crossing times
  const CvMat* const xtimes_mat = xtimes[0];

  // ray working matrix
  CvMat* ray_mat = NULL;

  // rotation working matrix
  CvMat* rot_mat = NULL;

  // ray equation
  real3 ray;

  // camera center
  real3 c;

  // reconstructed point
  real3 p;

  ray_mat = cvCreateMat(3, 1, real_typeid);
  ASSERT_GOTO(ray_mat, on_error);

  rot_mat = cvCreateMat(3, 1, real_typeid);
  ASSERT_GOTO(ray_mat, on_error);

  compute_camera_center(params, c);

  // foreach pixel that has a crossing time
  // compute the ray equation
  
  for (int i = 0; i < xtimes_mat->rows; ++i)
    for (int j = 0; j < xtimes_mat->cols; ++j)
    {
      // the entering times, t1 <= t < t2
      const real_type t = CV_MAT_ELEM(*xtimes_mat, real_type, i, j);
      const real_type t1 = floor(t);

      // no xtime for this pixel
      if (t == not_found) continue ;

      const unsigned int frame_index = (unsigned int)t1;

      // possible to get out of bounds
      if ((frame_index + 1) >= plane_eqs.shadow_planes.size())
	continue ;

      // skip frames that have no equation for the shadow plane
      if (plane_eqs.is_valid[frame_index] == 0)
	continue ;
      else if (plane_eqs.is_valid[frame_index + 1] == 0)
	continue ;

      // compute the ray equation
      // warning: pixel.x = j, pixel.y = i
      const real2 pixel = make_real2((real_type)j, (real_type)i);
      pixel_to_ray(pixel, params, ray);
      real3_to_mat(ray, ray_mat);
      cvGEMM(params.roth, ray_mat, 1, NULL, 0, rot_mat, CV_GEMM_A_T);
      mat_to_real3(rot_mat, ray);

      // average plane coefficients
      const real_type alpha = t - t1;
      const real4& t1_plane = plane_eqs.shadow_planes[frame_index + 0];
      const real4& t2_plane = plane_eqs.shadow_planes[frame_index + 1];
      real4 plane;
      for (unsigned int i = 0; i < 4; ++i)
	plane[i] = (1 - alpha) * t1_plane[i] + alpha * t2_plane[i];

      // intersect ray with shadow plane
      if (intersect_line_plane(c, ray, plane, p) != -1)
      {
#if 0 // filter too far
	static const real_type dist_reject = 2000;
	if (norm(p) > dist_reject) continue ;
#endif

#if 0 // filter if not in the clipping volume
	if ((p[0] < clip_x[0]) || (p[0] > clip_x[1])) continue ;
	if ((p[1] < clip_y[0]) || (p[1] > clip_y[1])) continue ;
	if ((p[2] < clip_z[0]) || (p[2] > clip_z[1])) continue ;
#endif

	points.push_back(p);
      }

#if 0 // debugging
      {
	printf("c: "); for (unsigned int i = 0; i < 3; ++i) printf(" %lf", c[i]);
	printf("\n");
	printf("ray: "); for (unsigned int i = 0; i < 3; ++i) printf(" %lf", ray[i]);
	printf("\n");
	printf("plane: "); for (unsigned int i = 0; i < 4; ++i) printf(" %lf", plane[i]);
	printf("\n");
	printf("p: "); for (unsigned int i = 0; i < 4; ++i) printf(" %lf", p[i]);
	printf("\n");
      }
#endif // debugging
    }

  // success
  error = 0;

 on_error:
  if (ray_mat) cvReleaseMat(&ray_mat);
  if (rot_mat) cvReleaseMat(&rot_mat);

  return error;
}


// write points as 3d triples

static int write_points
(const std::string& filename, const std::list<real3>& points)
{
  int error = -1;

  std::list<real3>::const_iterator pos = points.begin();
  std::list<real3>::const_iterator end = points.end();

  FILE* const file = fopen(filename.c_str(), "w+");
  ASSERT_GOTO(file, on_error);

  for (; pos != end; ++pos)
  {
    const real3& p = *pos;
    const int res = fprintf(file, "%lf %lf %lf\n", p[0], p[1], p[2]);
    ASSERT_GOTO(res > 0, on_error);
  }

  error = 0;

 on_error:
  if (file) fclose(file);
  return error;
}


// special lines fitting routines

static inline int fit_middle_line
(const user_points_t& user_points, real3& line_eq)
{
  return fit_line(user_points.mline, line_eq);
}


static int fit_upper_line
(const CvSize& image_size, real3& line_eq)
{
  CvPoint points[2];

  points[0].x = 0;
  points[0].y = image_size.height;
  points[1].x = image_size.width;
  points[1].y = image_size.height;

  return fit_line(points, line_eq);
}


static int fit_lower_line
(const CvSize& image_size, real3& line_eq)
{
  CvPoint points[2];

  points[0].x = 0;
  points[0].y = 0;
  points[1].x = image_size.width;
  points[1].y = 0;

  return fit_line(points, line_eq);
}

// scan main routine

static int do_scan(CvCapture* cap, const cam_params_t& params)
{
  int error = -1;
  CvMat* shadow_contrasts = NULL;
  CvMat* shadow_thresholds = NULL;
  CvMat* shadow_xtimes[2] = { NULL, NULL };
  user_points_t user_points;
  line_eqs_t line_eqs;
  plane_eqs_t plane_eqs;
  CvSize frame_size;
  std::list<real3> points;

  // error = get_user_points(cap, user_points);
  // print_user_points(user_points);
  error = get_static_user_points(user_points);
  ASSERT_GOTO(error == 0, on_error);

  // print_user_points(user_points);

  error = estimate_shadow_thresholds(cap, shadow_thresholds, shadow_contrasts);
  ASSERT_GOTO(error == 0, on_error);

  // show_matrix(shadow_thresholds);

  error = estimate_shadow_xtimes
    (cap, shadow_thresholds, shadow_contrasts, shadow_xtimes);
  ASSERT_GOTO(error == 0, on_error);

  error = fit_middle_line(user_points, line_eqs.middle);
  ASSERT_GOTO(error == 0, on_error);

  frame_size = get_capture_frame_size(cap);

  error = fit_upper_line(frame_size, line_eqs.upper);
  ASSERT_GOTO(error == 0, on_error);

  error = fit_lower_line(frame_size, line_eqs.lower);
  ASSERT_GOTO(error == 0, on_error);

  error = estimate_shadow_lines(cap, shadow_thresholds, user_points, line_eqs);
  ASSERT_GOTO(error == 0, on_error);

  error = estimate_reference_planes(cap, params, plane_eqs);
  ASSERT_GOTO(error == 0, on_error);

  error = estimate_shadow_planes(cap, params, line_eqs, plane_eqs);
  ASSERT_GOTO(error == 0, on_error);

  error = reconstruct_points(params, plane_eqs, shadow_xtimes, points);
  ASSERT_GOTO(error == 0, on_error);

  error = write_points("/tmp/fu.asc", points);
  ASSERT_GOTO(error == 0, on_error);

  error = 0;

 on_error:
  if (shadow_contrasts) cvReleaseMat(&shadow_contrasts);
  if (shadow_thresholds) cvReleaseMat(&shadow_thresholds);
  if (shadow_xtimes[0]) cvReleaseMat(&shadow_xtimes[0]);
  if (shadow_xtimes[1]) cvReleaseMat(&shadow_xtimes[1]);

  return error;
}


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
