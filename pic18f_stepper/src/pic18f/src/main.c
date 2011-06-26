/*
** Made by fabien le mentec <texane@gmail.com>
** 
** Started on  Sun Sep 20 14:08:30 2009 texane
** Last update Wed Nov 11 18:57:28 2009 texane
*/


#include <pic18fregs.h>
#include "config.h"
#include "adc.h"
#include "int.h"
#include "osc.h"
#include "serial.h"


/* ellegro stepper driver */

#define CONFIG_ALLEGRO_STEP_TRIS TRISBbits.TRISB0
#define CONFIG_ALLEGRO_STEP_PORT LATBbits.LATB0
#define CONFIG_ALLEGRO_DIR_TRIS TRISBbits.TRISB1
#define CONFIG_ALLEGRO_DIR_PORT LATBbits.LATB1

static void allegro_setup(void)
{
  CONFIG_ALLEGRO_STEP_TRIS = 0;
  CONFIG_ALLEGRO_STEP_PORT = 0;

  CONFIG_ALLEGRO_DIR_TRIS = 0;
  CONFIG_ALLEGRO_DIR_PORT = 0;
}

static void allegro_set_mode(unsigned int mode)
{
  /* mode in [0:3] */
  mode = mode;
}

static void wait_one_usec(void)
{
  volatile unsigned int i;
  for (i = 0; i < 5; ++i) ;
}

static void allegro_set_dir(unsigned int dir)
{
  /* assume((dir == 0) || (dir == 1)); */
  CONFIG_ALLEGRO_DIR_PORT = dir;
  wait_one_usec();
}

static void allegro_step(void)
{
  /* set high then wait for minimum pulse width
     time. then wait for minimum step low time
  */

  CONFIG_ALLEGRO_STEP_PORT = 1;
  wait_one_usec();

  CONFIG_ALLEGRO_STEP_PORT = 0;
  wait_one_usec();
}


/* pl15s020 stepper motor and screw */

#define CONFIG_PL_SLIDER_UM 1500 /* slider length, in micrometer  */
#define CONFIG_PL_SCREW_UM 5500 /* screw length, in micrometer */
#define CONFIG_PL_STEP_UM 115 /* step length, in micrometer */

#define CONFIG_PL_SW0_TRIS TRISBbits.TRISB2
#define CONFIG_PL_SW0_PORT PORTBbits.RB2
#define CONFIG_PL_SW1_TRIS TRISBbits.TRISB3
#define CONFIG_PL_SW1_PORT PORTBbits.RB3

/* moving direction (clockwise, counter cw) */
#define PL_DIR_CW 0
#define PL_DIR_CCW 1

#if 0 /* perstep length computation */

static inline double dtor(double d)
{ return (2 * M_PI / 360) * d; }

static inline double compute(double alpha, double diam)
{ return 2 * (sin(alpha) * diam / cos(alpha)); }
int main(int ac, char** av)

{ printf("%lf\n", compute(dtor(21), 3) / 20); }

#endif /* perstep length computation */

static void pl_setup(void)
{
  allegro_setup();

  CONFIG_PL_SW0_TRIS = 1;
  CONFIG_PL_SW1_TRIS = 1;
}

static unsigned int pl_move_until
(unsigned int dir, unsigned int mm, unsigned int bits)
{
  /* return the stopping reason, PL_MOVE_XXX */
  /* mm the distance to move in mm */  
  /* bits the stopping condition bits */

#define PL_MOVE_SUCCESS 0
#define PL_MOVE_SW0 (1 << 0)
#define PL_MOVE_FAILURE ((unsigned int)-1)

  unsigned int step_count = 50;

  mm = mm;

  allegro_set_dir(dir);

  for (; step_count; --step_count)
  {
    volatile unsigned int delay = 0;
    for (delay = 0; delay < 1000; ++delay) ;
    allegro_step();

    if (bits & PL_MOVE_SW0)
    {
      if (CONFIG_PL_SW0_PORT == 1)
	return PL_MOVE_SW0;
    }
  }

  return PL_MOVE_SUCCESS;
}

static unsigned int pl_move
(unsigned int dir, unsigned int mm)
{
  return pl_move_until(dir, mm, 0);
}

static int pl_move_initial(void)
{
  /* move to initial position (ie. until switch) */

  unsigned int pass = 0;
  unsigned int status;

#define CONFIG_PL_SW0_DIR PL_DIR_CW

  for (; pass < 50; ++pass)
  {
    status = pl_move_until(CONFIG_PL_SW0_DIR, 1, PL_MOVE_SW0);
    if (status == PL_MOVE_SW0) return 0;
  }

  /* failure */
  return -1;
}

#define pl_reverse_dir(__d) ((__d) ^ 1)


/* debugging led */

static void led_setup(void)
{
#define CONFIG_LED_TRIS TRISBbits.TRISB4
#define CONFIG_LED_PORT LATBbits.LATB4
  CONFIG_LED_TRIS = 0;
}

static void led_toggle(void)
{
  static unsigned char n = 0;
  CONFIG_LED_PORT = n;
  n ^= 1;
}


/* main */

int main(void)
{
  volatile int is_done = 0;
  unsigned int pass = 0;
  unsigned int dir = CONFIG_PL_SW0_DIR;
  unsigned int status;
  unsigned int bits = 0;

  osc_setup();
  int_setup();
  serial_setup();

  led_setup();

  pl_setup();
  pl_move_initial();

  /* todo: check move_initial status */

  /* reverse direction */
  dir = pl_reverse_dir(dir);

  while (is_done == 0)
  {
    /* wait before next move */
    if (++pass == 3000)
    {
      bits = (dir == CONFIG_PL_SW0_DIR) ? PL_MOVE_SW0 : 0;
      status = pl_move_until(dir, 42, bits);

      if (bits && (status == PL_MOVE_SW0))
      {
	/* reverse direction */
	led_toggle();
      }

      dir = pl_reverse_dir(dir);
      pass = 0;
    }

    /* adc_read(); */
  }

  return 0;
}
