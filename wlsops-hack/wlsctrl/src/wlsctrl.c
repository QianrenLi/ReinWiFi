#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <sys/mman.h>
#include "wlsctrl.h"

static int fd;
static info_blk *w_blk;
//<little-endian>:           ac-16,     txop-16,   cw_min-16, cw_max-16, aifs-8 if_ind-8
static char tx_params[10] = { 0x00,0x00, 0x00,0x00, 0x00,0x00, 0x00,0x00, 0x00 , 0x00};
// const char tx_prior[9]   = {0x00,0x00, 0x00,0x00, 0x01,0x00, 0x01,0x00, 0x00}; //0,0,1,1,1
// const char tx_normal[9]  = {0x00,0x00, 0x00,0x00, 0x03,0x00, 0x07,0x00, 0x02}; //0,0,3,7,2
// const char tx_last[9]    = {0x00,0x00, 0x00,0x00, 0xFF,0x03, 0xFF,0x07, 0x0F}; //0,0,1023,2047,15

static inline int mmap_write(const char *src, size_t len)
{
    if ((w_blk->byte_ptr[15]&0xFF) == 0x00) //kernel read complete
    {
        memcpy(w_blk, src, len);
        w_blk->byte_ptr[15] |= 0x80; //set kernel readable
        return 1;
    }
    else
    {
        return 0;
    }
}

int w_writer(const char *ptr)
{
    int cnt = 0;
    while(mmap_write(ptr, 10)==0)
    {
        if(++cnt>MAX_TIMEOUT)
        {
            return -1;
        }
    }
    return 0;
}

int set_tx_params(uint16_t ac, uint8_t aifs, uint16_t cw_min, uint16_t cw_max, uint8_t if_ind)
{
    tx_params[0] = ac & 0xFF;
    tx_params[8] = aifs;

    tx_params[4] = cw_min & 0xFF;
    tx_params[5] = (cw_min>>8) & 0xFF;

    tx_params[6] = cw_max & 0xFF;
    tx_params[7] = (cw_max>>8) & 0xFF;

    tx_params[9] = if_ind;
    return w_writer(tx_params);
}

int w_init()
{
    if( (fd=open(DBGFS_FILE, O_RDWR|O_SYNC)) < 0 )
    {
        perror("open call failed");
        return -1;
    }

    if( (w_blk=(info_blk *)mmap(NULL, sizeof(info_blk), PROT_READ|PROT_WRITE, MAP_SHARED, fd, 0)) == MAP_FAILED )
    {
        perror("mmap operation failed");
        return -1;
    }

    return 0;
}

void w_fini()
{
    munmap(w_blk, sizeof(w_blk));
    close(fd);
}