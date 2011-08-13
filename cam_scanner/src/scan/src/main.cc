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
  points.hplane[1].x = 425;
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
      CvMat header;
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

static int fit_line(const std::list<CvPoint>& points, real_type w[3])
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

#if REAL_TYPE_IS_DOUBLE
  static const int real_typeid = CV_64FC1;
#else
  static const int real_typeid = CV_32FC1;
#endif

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

  // solve and make implicit form
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

__attribute__((unused))static int draw_implicit_line
(IplImage* image, const real_type w[3], const CvScalar& color)
{
  // draw the line whose implicit form coefficients are in w[3]
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

static int estimate_shadow_planes
(CvCapture* cap, const CvMat* thr_mat, const user_points_t& user_points)
{
  // estimate shadow plane parameters
  // foreach_frame, estimate {h,v}line

  IplImage* gray_image = NULL;
  CvMat* gray_mat = NULL;

  // {vertical,horizontal}_{enter,leave} points
  std::list<CvPoint> shadow_points[4];

  // {vertical,horizontal}_{enter,leave} line equations
  real_type shadow_lineqs[4][3];

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

  // toremove
  seek_capture(cap, 60);
  // rewind_capture(cap);

  while (1)
  {
    IplImage* const frame_image = cvQueryFrame(cap);
    if (frame_image == NULL) break ;

    // create if does not yet exist
    if (gray_image == NULL)
    {
      CvMat header;
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
    for (unsigned int i = 0; i < 4; ++i)
      fit_line(shadow_points[i], shadow_lineqs[i]);

#if 1 // plot the lines
    {
      static const CvScalar colors[4] =
      {
	CV_RGB(0xff, 0x00, 0x00),
	CV_RGB(0x00, 0xff, 0x00),
	CV_RGB(0x00, 0x00, 0xff),
	CV_RGB(0xff, 0x00, 0xff)
      };

      IplImage* cloned_image = cvCloneImage(frame_image);
      ASSERT_GOTO(cloned_image, on_error);

      for (unsigned int i = 0; i < 4; ++i)
      {
	draw_points(cloned_image, shadow_points[i], colors[i]);
	draw_implicit_line(cloned_image, shadow_lineqs[i], colors[i]);
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
  user_points_t user_points;
  int error = -1;

  error = estimate_shadow_thresholds(cap, shadow_thresholds);
  ASSERT_GOTO(error == 0, on_error);

  // show_matrix(shadow_thresholds);

  // error = get_user_points(cap, user_points);
  error = get_static_user_points(user_points);
  ASSERT_GOTO(error == 0, on_error);

  // print_user_points(user_points);

  error = estimate_shadow_planes
    (cap, shadow_thresholds, user_points);
  ASSERT_GOTO(error == 0, on_error);

#if 0

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
