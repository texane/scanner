/*
** Made by fabien le mentec <texane@gmail.com>
** 
** Started on  Wed Nov 11 14:20:06 2009 texane
** Last update Sat Nov 14 11:36:07 2009 texane
*/


#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "serial.h"


static serial_handle_t handle = { -1, };


static int init_serial(void)
{
  static const char* const devname = "/dev/ttyUSB0";

  static serial_conf_t conf =
    { 9600, 8, SERIAL_PARITY_DISABLED, 1 };

  if (serial_open(&handle, devname) == -1)
    goto on_error;

  if (serial_set_conf(&handle, &conf) == -1)
    goto on_error;

  return 0;

 on_error:

  serial_close(&handle);

  return -1;
}

static void fini_serial(void)
{
  serial_close(&handle);
}

static int wait_for_read(void)
{
  int nfds;
  fd_set rds;

  FD_ZERO(&rds);
  FD_SET(handle.fd, &rds);

  nfds = select(handle.fd + 1, &rds, NULL, NULL, NULL);
  if (nfds != 1)
    return -1;

  return 0;
}

typedef struct sum
{
  uint32_t sum;
  uint32_t count;
} sum_t;

#if 0
static void process_pair
(uint16_t pos, uint16_t adc, sum_t* sums, unsigned int count)
{
  uint16_t dist;
  if (pos >= count) return ;
  sums[pos] += adc10_to_dist(adc);
  ++sums[pos].count;
}
#endif

static void average_sums(sum_t* sums, unsigned int count)
{
  unsigned int i;

  for (i = 0; i < count; ++i)
  {
    if (sums[i].count)
      sums[i].sum /= sums[i].count;
  }
}

int main(int ac, char** av)
{
#define CONFIG_SUM_COUNT 2048
#define CONFIG_READ_COUNT 2000
  unsigned int read_count;
  sum_t sums[2048];
  uint8_t buf[4];

  if (init_serial() == -1) return -1;

  memset(sums, 0, sizeof(sums));

#if 0
  for (read_count = 0; read_count < CONFIG_READ_COUNT; ++read_count)
  {
    if (wait_for_read() == -1) break ;
    if (serial_readn(&handle, (void*)buf, sizeof(buf))) break ;
    process_pair((uint16_t));
  }
#else
  uint16_t h, d, a;
  while (1)
  {
    if (wait_for_read() == -1) break ;
    if (serial_readn(&handle, (void*)&h, sizeof(h))) break ;

    if (wait_for_read() == -1) break ;
    if (serial_readn(&handle, (void*)&d, sizeof(d))) break ;

    if (wait_for_read() == -1) break ;
    if (serial_readn(&handle, (void*)&a, sizeof(a))) break ;

    printf("%04x %04x %04x\n", h, d, a);
  }
#endif

  fini_serial();

  return 0;
}
