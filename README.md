# FusionNet
Official implementation of FusionNet

The code is modified from the official implementation of epro-pnp(https://github.com/tjiiv-cprg/EPro-PnP/tree/main/EPro-PnP-6DoF), and is used for benchmarking only.

Environment

The code has been tested in the environment described as follows:

Linux (tested on Ubuntu 18.04)
Python 3.7
PyTorch 1.7.0

Find the details in environment.yaml

Dataset setting

Please refer to this link for instructions(CDPN:https://github.com/LZGMatrix/CDPN_ICCV2019_ZhigangLi)

Train
To start training, enter the directory EPro-PnP-6DoF/tools, and run:
python main.py --cfg /PATH/TO/CONFIG  # configs are located in EPro-PnP-6DoF/tools/exp_cfg

Test
To test and evaluate on the LineMOD test split, please edit the config file and

set the load_model option to the path of the checkpoint file,
change the test option from False to True

python main.py --cfg /PATH/TO/CONFIG
