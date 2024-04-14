#ifndef __WLSCTRL_H__
#define __WLSCTRL_H__

#include <inttypes.h>

#define DBGFS_FILE  "/proc/wlsctrl"
#define PAGE_SIZE   4096
#define MAX_TIMEOUT 600000 //2000

typedef union
{
    uint8_t byte_ptr[16];
    uint16_t word_ptr[8];
    uint32_t uint_ptr[4];
    uint64_t long_ptr[2];
}info_blk;

struct mmap_info
{
    info_blk *blk;
};

int w_init(void);
void w_fini(void);
int set_tx_params(uint16_t ac, uint8_t aifs, uint16_t cw_min, uint16_t cw_max, uint8_t if_ind);

#endif