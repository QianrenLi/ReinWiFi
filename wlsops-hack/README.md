# Wi-Fi Driver Real-time Hack
> Target for Linux Kernel v5.15 LTS.

An alternative to `iw` which uses `nl80211` interface.

Use `procfs + mmap` to access `ieee80211_ops` within ***extremely low latency*** from userspace.

<!--
The memory r/w is currently implemented in synchronized block writing/reading (to be lockless ring-buffer impl).
-->

### Build
- build the kernel module
    > We use a custom `iwlmvm` and `iwldvm` build in this repo. You can disable it with `cmake -DBUILD_IWLMVM=OFF -DBUILD_IWLDVM=TRUE ..` to activate the dvm module or `cmake -DBUILD_IWLMVM=TRUE -DBUILD_IWLDVM=OFF ..` mvm module.

    ```bash
    mkdir build
    cd build && cmake ..
    make
    ```

- build and install the python controller package
    > It is recommended to install the package globally which could be found by root user. If `master` module is built before, one may need to recompile the wheel by `sudo pip3 uninstall ./dist/*.whl`.
    ```bash
    cd wlsctrl
    python3 setup.py bdist_wheel
    sudo pip3 install ./dist/*.whl
    ```

### How to Use
> Make sure that you have at least one wireless NIC enabled.

1. To build the hack for realtek 8812au, one might include specfic include requirements to the folder wlsops or compile the module with makefile(.bak) in aircrack-ng/rtl8812au. (or directly use https://github.com/QianrenLi/rtl8812au)

2. Run `sudo insmod build/wlsops/wlsops_hack.ko` to install the built kernel modules, where `wlsops_hack` will use the first wireless NIC found in the system;

3. Run the test in `wlsctrl/tests` with root permission.
