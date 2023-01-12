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


# Demo

```Shell
python examples/demo_panda_trajectory.py SIM.RENDER True
python examples/demo_hand_gesture.py SIM.RENDER True
```

# Acknowledgements

The code is adopted from [handover-sim](https://github.com/NVlabs/handover-sim).
