MODNAME      := wlsops_hack

obj-m        += $(MODNAME).o
$(MODNAME)-y += hack_entry.o hack_mmap.o wlsops.o

ccflags-y   += -fno-stack-protector -fomit-frame-pointer
# ldflags-y   += -T$(src)

# LDFLAGS := $(ldflags-y) # legacy
KBUILD_CFLAGS := $(filter-out -pg,$(KBUILD_CFLAGS))
KBUILD_CFLAGS := $(filter-out -mfentry,$(KBUILD_CFLAGS))
