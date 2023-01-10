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


# Download files

- download [assets.zip](https://drive.google.com/file/d/1JGmKU6ICuFOdnAvRtFJ09sUcvhq_v4HK/view?usp=sharing), place the file under `gestureIL/data`, and extract it
- download [mano_poses.zip](https://drive.google.com/file/d/134-6Ql0AmiRo6mDQNsJ-CsITr8sDbbTA/view?usp=sharing), place the file under `gestureIL`, and extract it

# Acknowledgements

The code is adopted from [handover-sim](https://github.com/NVlabs/handover-sim).
