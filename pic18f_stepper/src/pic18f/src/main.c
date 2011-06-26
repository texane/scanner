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

#define CONFIG_ALLEGRO_STEP_TRIS TRISCbits.TRISC0
#define CONFIG_ALLEGRO_STEP_PORT LATCbits.LATC0
#define CONFIG_ALLEGRO_DIR_TRIS TRISCbits.TRISC1
#define CONFIG_ALLEGRO_DIR_PORT LATCbits.LATC1

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

/* WARNING: any change on INPUT pin in RB[4 - 7] will cause RBIF
   to be set, which may be considered as a false switch pushed
   if not used carefully.
*/
#define CONFIG_PL_SW0_TRIS TRISBbits.TRISB4
#define CONFIG_PL_SW0_PORT PORTBbits.RB4

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

/* use PORTB interrupt on change feature to track sw0 changes */

static int enable_rbif(void)
{
  /* setup ccp module so that an interrupt is generated on change */
  INTCONbits.RBIF = 0;
  if (CONFIG_PL_SW0_PORT == 1) return -1;
  return 0;
}

#define read_rbif() (INTCONbits.RBIF)
#define reset_rbif() do { INTCONbits.RBIF = 0; } while (0)


static void pl_setup(void)
{
  allegro_setup();

  CONFIG_PL_SW0_TRIS = 1;

  /* exclude from RBIF set */
  TRISBbits.TRISB5 = 0;
  TRISBbits.TRISB6 = 0;
  TRISBbits.TRISB7 = 0;

  INTCONbits.RBIF = 0;
  INTCONbits.RBIE = 0;
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

  unsigned int step_count = mm;

  if (bits & PL_MOVE_SW0)
  {
    if (enable_rbif() == -1)
      return PL_MOVE_SW0;
  }

  allegro_set_dir(dir);

  for (; step_count; --step_count)
  {
    volatile unsigned int delay = 0;
    for (delay = 0; delay < 500; ++delay) ;
    allegro_step();

    if (bits & PL_MOVE_SW0)
    {
      if (read_rbif())
      {
	reset_rbif();
	return PL_MOVE_SW0;
      }
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

  unsigned int pass;
  unsigned int status;

#define CONFIG_PL_SW0_DIR PL_DIR_CCW

  for (pass = 0; pass < 50; ++pass)
  {
    status = pl_move_until(CONFIG_PL_SW0_DIR, 10, PL_MOVE_SW0);
    if (status == PL_MOVE_SW0)
    {
      /* move some steps reverse */
      pl_move_until(PL_DIR_CW, 5, 0);
      return 0;
    }
  }

  /* failure */
  return -1;
}

#define pl_reverse_dir(__d) ((__d) ^ 1)


/* debugging led */

static void led_setup(void)
{
#define CONFIG_LED_TRIS TRISBbits.TRISB2
#define CONFIG_LED_PORT LATBbits.LATB2
  CONFIG_LED_TRIS = 0;
  CONFIG_LED_PORT = 0;
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

#if 0 /* unused */
  enable_rbif();
  while (1)
  {
    if (read_rbif())
    {
      CONFIG_LED_PORT = 1;
      reset_rbif();
    }
    else
      CONFIG_LED_PORT = 0;
  }
#endif /* unused */

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
      status = pl_move_until(dir, 200, bits);

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
