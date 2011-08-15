#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <list>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include "common/assert.hh"
#include "common/utils.hh"
#include "common/real_type.hh"
#include "common/cam_params.hh"
#include "common/fixed_vector.hh"


// toremove
#define CONFIG_SKIP_COUNT 0


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

  if ((p[1] == 0) || (q[1] == 0)) return -1;

  const real_type a0 = -p[0] / p[1];
  const real_type a1 = -q[0] / q[1];

  if (fabs(a0 - a1) < 0.0001) return -1;

  const real_type b0 = p[2] / p[1];
  const real_type b1 = q[2] / q[1];

  // assume r.size() >= 2
  r[0] = (b1 - b0) / (a0 - a1);
  r[1] = a0 * r[0] + b0;

  return 0;
}

static int intersect_line_plane(const real3& p, const real3& q, real4& r)
{
  // intersect a line with a plane
  // res the resulting point in 3d coordinates
  // q a point of the line
  // v the line vector
  // w a plane in explicit form

  return 0;
}


static int pixel_to_ray
(const CvPoint& pixel, const cam_params_t& params, real3& ray)
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

  src = cvCreateMat(1, 1, real_typeid);
  ASSERT_GOTO(src, on_error);

  dst = cvCreateMat(1, 1, real_typeid);
  ASSERT_GOTO(dst, on_error);

  scalar.val[0] = (real_type)pixel.x;
  scalar.val[1] = (real_type)pixel.y;
  cvSet1D(src, 0, scalar);
  cvUndistortPoints(src, dst, params.intrinsic, params.distortion);

  scalar = cvGet1D(dst, 0);
  scalar.val[2] = 1;

  // normalize and assign
  norm = 0;
  for (unsigned int i = 0; i < 3; ++i)
    norm += scalar.val[i] * scalar.val[i];

  // assume ray.size() >= 3
  for (unsigned int i = 0; i < 3; ++i) ray[i] = scalar.val[i] / norm;

  error = 0;

 on_error:
  if (src) cvReleaseMat(&src);
  if (dst) cvReleaseMat(&dst);
  return error;
}


// shadow threshold estimtation

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

  CvMat header;

  // fixme
  rewind_capture(cap);
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

  return error;
}

static int estimate_shadow_xtimes
(CvCapture* cap, const CvMat* thr_mat, CvMat* xtime_mat[2])
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
  unsigned int nrows = 0;
  unsigned int ncols = 0;
  int error = -1;

  rewind_capture(cap);

  for (unsigned int frame_index = 0; true; ++frame_index)
  {
    IplImage* const frame_image = cvQueryFrame(cap);
    if (frame_image == NULL) break ;

    // create on first pass
    if (frame_index == 0)
    {
      CvSize size = cvGetSize(frame_image);

      curr_image = cvCreateImage(size, IPL_DEPTH_8U, 1);
      ASSERT_GOTO(curr_image, on_error);

      prev_image = cvCreateImage(size, IPL_DEPTH_8U, 1);
      ASSERT_GOTO(prev_image, on_error);

      // retrieve corresponding matrices
      prev_mat = cvGetMat(prev_image, &prev_header);
      curr_mat = cvGetMat(curr_image, &curr_header);

      // update nrows ncols
      nrows = size.height;
      ncols = size.width;

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
    }

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

#if 0 // plot (entering) shadow xtimes
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

  // printf("y = %lf * x + %lf\n", a, b);

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
  // points is a n x m matrix where M the dimensionality

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
      centroid->data.fl[c] += points->data.fl[ncols * r + c];
    centroid->data.fl[c] /= nrows;
  }

  // subtract geometric centroid from each point

  points2 = cvCreateMat(nrows, ncols, type);
  ASSERT_GOTO(points2, on_error);

  for (int r = 0; r < nrows; ++r)
    for (int c = 0; c < ncols; ++c)
    {
      points2->data.fl[ncols * r + c] =
	points->data.fl[ncols * r + c] - centroid->data.fl[c];
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
    plane[c] = V->data.fl[ncols * (ncols - 1) + c];
    plane[ncols] += plane[c] * centroid->data.fl[c];
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

  std::list<real3> venter;
  std::list<real3> vleave;
  std::list<real3> henter;
  std::list<real3> hleave;

  real3 middle;
  real3 upper;
  real3 lower;

} line_eqs_t;


typedef struct plane_eqs
{
  // plane explicit equation coefficients

  real4 vplane;
  real4 hplane;

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

#if 1 // toremove
  unsigned int frame_index = CONFIG_SKIP_COUNT;
  seek_capture(cap, frame_index);
#else
  rewind_capture(cap);
#endif

  while (1)
  {
    // printf("frame_index == %u\n", frame_index++);

    IplImage* const frame_image = cvQueryFrame(cap);
    if (frame_image == NULL) break ;

    // create if does not yet exist
    if (gray_image == NULL)
    {
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

    // get lines equation via lse fitting method

    real3 line_eq;

    fit_line(shadow_points[0], line_eq);
    line_eqs.venter.push_back(line_eq);

    fit_line(shadow_points[1], line_eq);
    line_eqs.vleave.push_back(line_eq);

    fit_line(shadow_points[2], line_eq);
    line_eqs.henter.push_back(line_eq);

    fit_line(shadow_points[3], line_eq);
    line_eqs.hleave.push_back(line_eq);

#if 0 // plot the lines
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

      draw_line(cloned_image, line_eqs.venter.back(), colors[0]);
      draw_line(cloned_image, line_eqs.vleave.back(), colors[1]);
      draw_line(cloned_image, line_eqs.henter.back(), colors[2]);
      draw_line(cloned_image, line_eqs.hleave.back(), colors[3]);

      const int res = check_shadow_lines
	(line_eqs.venter.back(), line_eqs.henter.back(), line_eqs.middle);
      if (res == -1)
      {
	printf("invalid shadow lines\n");
	draw_line(cloned_image, line_eqs.middle, colors[4]);
      }

      show_image(cloned_image, "ShadowLines");
      cvReleaseImage(&cloned_image);
    }
#endif // plot the line
  }

  error = 0;

 on_error:
  if (gray_image) cvReleaseImage(&gray_image);
  return error;
}


static int estimate_reference_planes
(CvCapture* cap, const cam_params_t& params, plane_eqs_t& plane_eqs)
{
  // the length along x (resp. y) axis between checkboards rectangles

  static const real_type dx = 558.8;
  static const real_type dy = 303.2125;

  // working matrices

  CvMat* points = NULL;
  CvMat* tsub = NULL;
  CvMat* tmp = NULL;

  // resulting error

  int error = -1;

  // create and assign a points matrix

  points = cvCreateMat(3, 3, real_typeid);
  ASSERT_GOTO(points, on_error);

  CV_MAT_ELEM(*points, real_type, 0, 0) = 0;
  CV_MAT_ELEM(*points, real_type, 0, 1) = dx;
  CV_MAT_ELEM(*points, real_type, 0, 2) = dx;

  CV_MAT_ELEM(*points, real_type, 1, 0) = 0;
  CV_MAT_ELEM(*points, real_type, 1, 1) = 0;
  CV_MAT_ELEM(*points, real_type, 1, 2) = dy;

  CV_MAT_ELEM(*points, real_type, 2, 0) = 0;
  CV_MAT_ELEM(*points, real_type, 2, 1) = 0;
  CV_MAT_ELEM(*points, real_type, 2, 2) = 0;

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

  tsub = cvCreateMat(params.transv->rows, points->cols, params.transv->type);
  ASSERT_GOTO(tsub, on_error);

  for (int i = 0; i < tsub->rows; ++i)
    for (int j = 0; j < tsub->cols; ++j)
      CV_MAT_ELEM(*tsub, real_type, i, j) = CV_MAT_ELEM(*tmp, real_type, i, 0);

  cvReleaseMat(&tmp);

  // tmp = rotv * x + tsub;

  tmp = cvCreateMat(params.rotv->rows, points->cols, points->type);
  ASSERT_GOTO(tmp, on_error);
  cvGEMM(params.rotv, points, 1, tsub, 1, tmp, 0);

  // x = roth' * tmp;

  cvGEMM(params.roth, tmp, 1, NULL, 0, points, CV_GEMM_A_T);

  // fit the plane

  error = fit_plane(points, plane_eqs.vplane);
  ASSERT_GOTO(error == 0, on_error);
  error = -1;

  error = 0;

 on_error:
  if (points) cvReleaseMat(&points);
  if (tmp) cvReleaseMat(&tmp);
  if (tsub) cvReleaseMat(&tsub);

  return error;
}


static int estimate_shadow_planes
(
 CvCapture* cap,
 const cam_params_t& params,
 const line_eqs_t& line_eqs,
 plane_eqs_t& plane_eqs
)
{
  static const unsigned int frame_index = CONFIG_SKIP_COUNT;

  int error = -1;

  std::list<real3>::const_iterator venter_pos;
  std::list<real3>::const_iterator vleave_pos;
  std::list<real3>::const_iterator henter_pos;
  std::list<real3>::const_iterator hleave_pos;

  real2 point;
  real3 ray;

  CvMat* c = NULL;

  // c = -roth' * transh;
  c = cvCreateMat(3, 1, real_typeid);
  ASSERT_GOTO(c, on_error);
  cvGEMM(params.roth, params.transh, -1, NULL, 0, c, CV_GEMM_A_T); 

  // foreach frame
  // determine true position of the lines
  // compute the shadow planes

  seek_capture(cap, frame_index);

  venter_pos = line_eqs.venter.begin();
  vleave_pos = line_eqs.vleave.begin();
  henter_pos = line_eqs.henter.begin();
  hleave_pos = line_eqs.hleave.begin();

  for (; true; ++venter_pos, ++vleave_pos, ++henter_pos, ++hleave_pos)
  {
    IplImage* const frame_image = cvQueryFrame(cap);
    if (frame_image == NULL) break ;

    // determine vertical plane

    // find the venter middle intersection
    intersect_line_line(*venter_pos, line_eqs.middle, point);

    // pixel_to_ray(point, params);

    // point = intersect_line_line(venter[i], middle);
    // ray = roth * pixel_to_ray(point, params);
    // vplane = intersect_line_plane(C, ray)

    // determine horizontal plane

    // compute the entering plane params

    // store the params in plane_eqs

    // redo above steps for leaving plane

  }

  // success
  error = 0;

 on_error:
  if (c) cvReleaseMat(&c);

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
  CvMat* shadow_thresholds = NULL;
  CvMat* shadow_xtimes[2] = { NULL, NULL };
  user_points_t user_points;
  line_eqs_t line_eqs;
  plane_eqs_t plane_eqs;
  CvSize frame_size;
  int error = -1;

  // error = get_user_points(cap, user_points);
  error = get_static_user_points(user_points);
  ASSERT_GOTO(error == 0, on_error);

  // print_user_points(user_points);

  error = estimate_shadow_thresholds(cap, shadow_thresholds);
  ASSERT_GOTO(error == 0, on_error);

  // show_matrix(shadow_thresholds);

  error = estimate_shadow_xtimes(cap, shadow_thresholds, shadow_xtimes);
  ASSERT_GOTO(error == 0, on_error);  

  error = fit_middle_line(user_points, line_eqs.middle);
  ASSERT_GOTO(error == 0, on_error);

  // fixme, rewinding should not be needed
  rewind_capture(cap);
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

  // todo: reconstruct_points by pixel rays shadow planes intersection

  error = 0;

 on_error:
  if (shadow_thresholds != NULL) cvReleaseMat(&shadow_thresholds);
  if (shadow_xtimes[0] != NULL) cvReleaseMat(&shadow_xtimes[0]);
  if (shadow_xtimes[1] != NULL) cvReleaseMat(&shadow_xtimes[1]);

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
