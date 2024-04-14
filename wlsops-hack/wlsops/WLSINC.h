#ifndef __WLSINC_H__
#define __WLSINC_H__

#include <linux/kern_levels.h>
#include <asm/page.h>

#define WLS_KEYWORD "wlsops: "
#define KERN_LOG KERN_NOTICE WLS_KEYWORD
#define printh(...) printk(KERN_LOG __VA_ARGS__)

#define WLS_PAGE_NUM    1
#define WLS_RING_SIZE   (WLS_PAGE_NUM*PAGE_SIZE)
#define WLS_BLK_NUM     (WLS_RING_SIZE/sizeof(info_blk))

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

#endif