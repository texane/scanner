/*
** Made by fabien le mentec <texane@gmail.com>
** 
** Started on  Fri Oct  2 15:48:04 2009 texane
** Last update Fri Oct  2 18:17:37 2009 texane
*/



#include <pic18fregs.h>



unsigned short adc_read(unsigned char chan)
{
  /* assume chan less than 8 */

  unsigned short value;

  /* input pin */

  TRISA = TRISA | (1 << chan);

  PIE1bits.ADIE = 0;

#if 0 /* ADCON1 */

  /* analog inputs */

  ADCON1bits.PCFG0 = 1;
  ADCON1bits.PCFG1 = 1;
  ADCON1bits.PCFG2 = 1;
  ADCON1bits.PCFG3 = 0;

  /* vref-: vss, vref+: an3 */
  ADCON1bits.VCFG0 = 0;
  ADCON1bits.VCFG1 = 1;

#else

  ADCON1 = (1 << 4) | (0xf - (chan + 1));

#endif
  
#if 0 /* select channel */

  ADCON0bits.CHS0 = 0;
  ADCON0bits.CHS1 = 0;
  ADCON0bits.CHS2 = 0;
  ADCON0bits.CHS3 = 0;

#else

  ADCON0 = chan << 2;

#endif

#if 0 /* ADCON2 */

  /* result format right aligned */

  ADCON2bits.ADFM = 1;

  /* acquisition time */

  ADCON2bits.ACQT0 = 1;
  ADCON2bits.ACQT1 = 1;
  ADCON2bits.ACQT2 = 0;

  /* conversion clock */

  ADCON2bits.ADCS0 = 1;
  ADCON2bits.ADCS1 = 0;
  ADCON2bits.ADCS2 = 0;

#else

  ADCON2 = (1 << 7) | (3 << 3) | 1;

#endif

  /* turn on analog module */

  ADCON0bits.ADON = 1;

  /* start the conversion */

  ADCON0bits.GO = 1;

  /* wait for conversion */

  while (ADCON0bits.GO)
    ;

  value = (((unsigned short)ADRESH << 8) | ADRESL);

  /* turn off */

  ADCON0bits.ADON = 0;

  /* 10 bits adc */

  return value & 0x3ff;
}
