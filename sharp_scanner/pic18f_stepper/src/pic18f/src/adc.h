/*
** Made by fabien le mentec <texane@gmail.com>
** 
** Started on  Fri Oct  2 15:46:56 2009 texane
** Last update Sat Oct  3 16:32:37 2009 texane
*/



#ifndef ADC_H_INCLUDED
# define ADC_H_INCLUDED



/* 10 bits adc */


#define ADC_MAX_VALUE (1024 - 1)

#define ADC_QUANTIZE_5_10(V) ((unsigned short)(((V) * 1024.f) / 5.f))


unsigned short adc_read(unsigned char);



#endif /* ! ADC_H_INCLUDED */
