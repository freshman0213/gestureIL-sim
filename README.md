# Prerequisites

This code is tested with Python 3.8 on Ubuntu 20.04.


# Install

```Shell
git clone
cd gestureIL-sim
git submodule update --init --recursive
```

# Setup python environment

- create conda environment
- install easysim
    ```Shell
    cd modified_easysim
    pip install -e .
    ```
- install gestureIL
    ```Shell
    cd gestureIL-sim
    pip install -e .
    ```


# Acknowledgements

The code is adopted from [handover-sim](https://github.com/NVlabs/handover-sim).
