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



/* servo pwm */

#define CONFIG_PWM_TRIS TRISCbits.TRISC1
#define CONFIG_PWM_PERIOD 2000

static void pwm_setup(void)
{
  /* necessary to avoid pin early undefined state */
  CONFIG_PWM_TRIS = 0;
}

static void pwm_start(unsigned int duty_usecs, unsigned int period_usecs)
{
#define CPU_Fosc 8000000
#define TMR2_prescaler 16
  unsigned int dcycle = (CPU_Fosc / 1000000) * (duty_usecs / TMR2_prescaler);

  /* period register */
  PR2 = (CPU_Fosc / 1000000) * (period_usecs / (4 * TMR2_prescaler)) - 1;

  /* duty cycle msbs */
  CCPR2L = (unsigned char)(dcycle >> 2);

  /* pin selected by conf bits */
  CONFIG_PWM_TRIS = 0;

  /* enable timer2, prescaler == 16 */
  T2CONbits.TMR2ON = 1;
  T2CONbits.T2CKPS0 = 0;
  T2CONbits.T2CKPS1 = 1;

  /* configure control register */
#if 0 
  {
    /* pwm mode */
    CCP2CONbits.CCP2M0 = 0;
    CCP2CONbits.CCP2M1 = 0;
    CCP2CONbits.CCP2M2 = 1;
    CCP2CONbits.CCP2M3 = 1;

    /* duty cycle lsbs */
    CCP2CONbits.DC2B0 = ((unsigned char)dcycle >> 0) & 1;
    CCP2CONbits.DC2B1 = ((unsigned char)dcycle >> 1) & 1;
  }
#else
  {
    CCP2CON = (((unsigned char)(dcycle & 0x3)) << 4) | 0x0c;
  }
#endif
}

static void pwm_stop(void)
{
  CCP2CON = 0;
}

static void pwm_next(void)
{
  volatile unsigned int i;
  pwm_start(1200, CONFIG_PWM_PERIOD);
#define CONFIG_PWM_ITER_04_675 2500
#define CONFIG_PWM_ITER_15_000 5000
  for (i = 0; i < CONFIG_PWM_ITER_15_000; ++i) ;
  pwm_stop();
}


/* allegro stepper driver
   warning: rc3 pin not implemented
   warning: rc4, rc5 have no digital output capabilities
 */

#define CONFIG_ALLEGRO_STEP_TRIS TRISCbits.TRISC2
#define CONFIG_ALLEGRO_STEP_PORT LATCbits.LATC2
#define CONFIG_ALLEGRO_DIR_TRIS TRISCbits.TRISC0
#define CONFIG_ALLEGRO_DIR_PORT LATCbits.LATC0

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
  if (CONFIG_PL_SW0_PORT == 0) return -1;
  return 0;
}

#define read_rbif() (INTCONbits.RBIF)
#define reset_rbif() do { INTCONbits.RBIF = 0; } while (0)

#define pl_reverse_dir(__d) (((__d) ^ 1) & 1)

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
  unsigned int retries = 0;

#define CONFIG_PL_SW0_DIR PL_DIR_CCW

  for (pass = 0; pass < 40; ++pass)
  {
    retries = 0;

  retry_move:
    status = pl_move_until(CONFIG_PL_SW0_DIR, 10, PL_MOVE_SW0);
    if (status == PL_MOVE_SW0)
    {
      if ((++retries) < 5) goto retry_move;

      /* move some steps reverse */
      pl_move_until(pl_reverse_dir(CONFIG_PL_SW0_DIR), 5, 0);
      return 0;
    }
  }

  /* failure */
  return -1;
}


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

#define led_set(__n) do { CONFIG_LED_PORT = (__n); } while (0)

/* main */

int main(void)
{
  volatile int is_done = 0;
  unsigned int pass = 0;
  unsigned int dir;
  unsigned int status;
  unsigned int bits = 0;
  unsigned int adc_value;
  unsigned int step_pos;
  unsigned int step_pwm;
  unsigned int i;

  /* early init */
  ADCON1 = 0x0F;
  CMCON = 0x07;

  /* disable portb pullups */
  INTCON2bits.RBPU = 1;

  osc_setup();
  int_setup();
  serial_setup();
  led_setup();
  pwm_setup();
  pl_setup();


#if 0 /* switch unit */
  led_setup();

  enable_rbif();
  while (1)
  {
    if (read_rbif())
    {
      reset_rbif();
      led_toggle();
    }
  }
  
  while (1) ;
#endif /* switch unit */

#if 0 /* serial unit */
  while (1)
  {
    volatile unsigned int i;
    for (i = 0; i < 10000; ++i) ;
    serial_writei(adc_read(CONFIG_ADC_CHANNEL));
  }
#endif /* serial unit */

#if 0 /* servo unit */
  bits = 1;
 redo_pwm:
  if (bits) pwm_start(1200, CONFIG_PWM_PERIOD);
  for (is_done = 0; is_done < CONFIG_PWM_ITER_15_000; ++is_done) ;
  if (bits) pwm_stop();
  for (i = 0; i < 10; ++i)
    for (is_done = 0; is_done < 10000; ++is_done) ;
  bits ^= 1;
  goto redo_pwm;
  while (1) ;
#endif /* servo unit */

#if 0 /* pl unit */
  {
    dir = CONFIG_PL_SW0_DIR;
    while (1)
    {
      led_set(dir);
      pl_move(dir, 100);
      dir = pl_reverse_dir(dir);
    }
  }
  while (1) ;
#endif /* pl unit */

  pl_move_initial();

  /* todo: check move_initial status */

  /* reverse direction */
  dir = pl_reverse_dir(CONFIG_PL_SW0_DIR);

  /* scan */
  step_pos = 0;
  step_pwm = 0;

  while (is_done == 0)
  {
    /* wait before next move */
    ++pass;
#define CONFIG_ADC_PER_STEP 4
    if (pass == (400 / CONFIG_ADC_PER_STEP))
    {
      pass = 0;

      bits = 0;
      /* fixme */
      /* if (dir == CONFIG_PL_SW0_DIR) bits |= PL_MOVE_SW0; */
      /* fixme */
      status = pl_move_until(dir, 1, bits);

      if (status == PL_MOVE_SUCCESS)
      {
	/* update step count or reverse dir */
	if (dir == CONFIG_PL_SW0_DIR)
	{
	  if ((--step_pos) == 0)
	  {
	    /* next angle */
	    ++step_pwm;
	    pwm_next();

	    dir = pl_reverse_dir(dir);
	  }
	}
	else
	{
	  /* assume 190 steps per run */
	  if ((++step_pos) == 190)
	  {
	    /* next angle */
	    ++step_pwm;
	    pwm_next();

	    dir = pl_reverse_dir(dir);
	  }
	}

	for (i = 0; i < CONFIG_ADC_PER_STEP; ++i)
	{
	/* read adc and send pair */
#define CONFIG_ADC_CHANNEL 1
	  adc_value = adc_read(CONFIG_ADC_CHANNEL);
	  serial_writei(step_pos);
	  serial_writei(adc_value);
	  serial_writei(step_pwm);
	}
      }
      else if (bits && (status == PL_MOVE_SW0))
      {
	/* reverse direction */
	led_toggle();
	step_pos = 0;
	dir = pl_reverse_dir(dir);

	/* next angle */
	++step_pwm;
	pwm_next();
      }
    }
  }

  return 0;
}
