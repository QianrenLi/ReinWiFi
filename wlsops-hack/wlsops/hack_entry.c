#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/module.h>
#include <linux/types.h>
#include <linux/version.h>

#include <linux/sched.h>
#include <linux/kthread.h>
#include "hack_mmap.h"
#include "wlsops.h"
#include "WLSINC.h"

static struct task_struct *kThread;
static struct tx_param kParam;

static char *if_name = "default";
module_param(if_name, charp, S_IRUGO | S_IWUSR);

// static char *module_name = "wlsops_hack";
// module_param(module_name, charp, S_IRUGO | S_IWUSR);


int read_loop(void *data)
{
    while(!kthread_should_stop())
    {
        schedule();
        if (mmap_read((char *)&kParam,sizeof(kParam)))
        {
            // printh("%d: (%d, %d, %d)", kParam.ac, kParam.aifs, kParam.cw_min, kParam.cw_max);
            wls_conf_tx(kParam);
        }
    }

    return 0;
}

static int __init wlsops_init(void)
{
    if ( (wls_hack_init(if_name)<0) || (hack_mmap_init()<0) )
    {
        printh("HACK_ENTRY failed.\n");
        return -1;
    }

    kThread = kthread_create(read_loop, NULL, "wlsops_hack");
    wake_up_process(kThread);

    return 0;
}

static void __exit wlsops_fini(void)
{
    kthread_stop(kThread);
    hack_mmap_fini();
    
    printh("Now exit~\n");
    printh("\n");
}


module_init(wlsops_init);
module_exit(wlsops_fini);
MODULE_LICENSE("GPL");
MODULE_AUTHOR("sudo_free");