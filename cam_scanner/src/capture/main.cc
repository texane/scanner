#include <cstdio>
#include <cstdlib>
#include <list>
#include <string>
#include <cv.h>
#include <highgui.h>
#include "../common/conf.hh"
#include "../common/assert.hh"


static std::string make_indexed_name(const std::string& dirname, unsigned int i)
{
  char namebuf[1024];
  ::snprintf(namebuf, sizeof(namebuf), "%s/%06u.jpg", dirname.c_str(), i);
  return std::string(namebuf);
}


int main(int ac, char** av)
{
  static const char* const frame_window = "FrameWindow";
  conf_t conf;
  std::list<IplImage*> frames;
  int err = 0;
  std::string projname;
  std::string optname;

  ASSERT_GOTO(ac >= 3, on_error);
  projname = av[1];
  optname = av[2];

  // load project conf
  {
    std::string confname = projname + "/conf.xml";
    err = load_conf(conf, confname);
    ASSERT_GOTO(err == 0, on_error);
  }

  // initialize ui
  {
    // lazy initialization
    // err = cvInitSystem(ac, av);
    err = cvNamedWindow(frame_window, CV_WINDOW_AUTOSIZE);
    ASSERT_GOTO(err == 1, on_error);
  }

  // initialize capture
  {
    CvCapture* cap = cvCaptureFromCAM(conf.cam_index);
    ASSERT_GOTO(cap != NULL, on_error);
    while (1)
    {
      const int key = cvWaitKey(5);
      if (key == 27) break;

      IplImage* const frame = cvQueryFrame(cap);
      // fixme: capture not released
      ASSERT_GOTO(frame != NULL, on_error);

      cvShowImage(frame_window, frame);

      if (key == ' ')
      {
	IplImage* const cloned = cvCloneImage(frame);
	// fixme: capture not released
	ASSERT_GOTO(cloned != NULL, on_error);

	frames.push_back(cloned);
      }
    }
    cvReleaseCapture(&cap);
  }

  // create a directory and store grabbed images
  {
    std::string dirname = conf.proj_dirname + std::string("/");
    if (optname == "calib")
      dirname.append(conf.calib_frames_dirname);
    else if (optname == "scan")
      dirname.append(conf.scan_frames_dirname);
    else ASSERT_GOTO(0, on_error);

    std::list<IplImage*>::iterator pos = frames.begin();
    std::list<IplImage*>::iterator end = frames.end();
    unsigned int i = 0;
    for (; pos != end; ++pos, ++i)
    {
      const std::string filename = make_indexed_name(dirname, i);
      err = cvSaveImage(filename.c_str(), *pos);
      ASSERT_GOTO(err == 1, on_error);
    }
  }

 on_error:
  // release frames
  {
    std::list<IplImage*>::iterator pos = frames.begin();
    std::list<IplImage*>::iterator end = frames.end();
    for (; pos != end; ++pos) cvReleaseImage(&(*pos));
  }

  return err;
}
