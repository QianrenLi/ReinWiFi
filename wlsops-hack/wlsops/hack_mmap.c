#include <linux/sched.h>
#include <asm/uaccess.h> /* copy_from_user */
#include <linux/fs.h>
#include <linux/proc_fs.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include "hack_mmap.h"

static struct mmap_info *m_info;

/******************** STRUCT MMAP OPERATIONS ********************/
static void mmap_open(struct vm_area_struct *vma)
{
    // info_blk *tmp_info = (mmap_info *)vma->vm_private_data
}

static void mmap_close(struct vm_area_struct *vma)
{
    // info_blk *tmp_info = (mmap_info *)vma->vm_private_data;
}

static vm_fault_t mmap_fault(struct vm_fault *vmf)
{
    struct page *v_page;

    v_page = virt_to_page(m_info->blk);
    get_page(v_page);
    vmf->page = v_page;
     
    return 0;
}

struct vm_operations_struct mmap_vm_ops =
{
    .open = mmap_open,
    .close = mmap_close,
    .fault = mmap_fault
};

/******************** STRUCT FILE OPERATIONS ********************/
static int fop_open_mmap(struct inode *inode, struct file *fd)
{
    fd->private_data = m_info;    // attach memory to procfs fd
    return 0;
}

static ssize_t default_read_file(struct file *file, char __user *buf, size_t count, loff_t *ppos)
{
    return 0;
}

static ssize_t default_write_file(struct file *file, const char __user *buf, size_t count, loff_t *ppos)
{
    return 0;
}

static int fop_close_mmap(struct inode *inode, struct file *fd)
{
    fd->private_data = NULL;
    return 0;
}

int mmap_ops_mount(struct file *fd, struct vm_area_struct *vma)
{
    vma->vm_ops = &mmap_vm_ops;
    vma->vm_flags |= VM_DONTEXPAND | VM_DONTDUMP;//VM_RESERVED;
    vma->vm_private_data = fd->private_data;
    mmap_open(vma);
    return 0;
}

static const struct proc_ops mmap_fops = {
    .proc_open = fop_open_mmap,
    .proc_read = default_read_file,
    .proc_write = default_write_file,
    .proc_release = fop_close_mmap,
    .proc_mmap = mmap_ops_mount,
};

/******************** CUSTOMIZED FUNCTIONS ********************/
inline int mmap_read(char *dst, size_t len)
{
    if (m_info->blk->byte_ptr[15] & 0x80) //user write complete
    {
        memcpy(dst, m_info->blk, len);
        m_info->blk->byte_ptr[15] = 0x00; //set user writable
        return 1;
    }
    else
    {
        return 0;
    }
}

int hack_mmap_init()
{
    m_info = kmalloc(sizeof(struct mmap_info), GFP_KERNEL);
    m_info->blk = (info_blk *) get_zeroed_page(GFP_KERNEL);
    proc_create(DBGFS_FILE, 0, NULL, &mmap_fops);
    return 0;
}

void hack_mmap_fini()
{
    remove_proc_entry(DBGFS_FILE, NULL);
    free_page((unsigned long)m_info->blk);
    kfree(m_info);
}