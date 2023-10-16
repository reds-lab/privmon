# Decision-based (label-only) Membership Inference Attacks (Original)

This repository contains the implementation for Membership Leakage in Label-Only Exposures (ACM CCS 2021)

Our attack is also already available on [Adversarial Robustness Toolbox (ART)](https://adversarial-robustness-toolbox.readthedocs.io/en/latest/modules/attacks/inference/membership_inference.html#membership-inference-label-only-decision-boundary), which is a popular Python packages for machine learning security.

## Citation
```bibtex
@inproceedings{LZ21,
author = {Zheng Li and Yang Zhang},
title = {{Membership Leakage in Label-Only Exposures}},
booktitle = {{ACM SIGSAC Conference on Computer and Communications Security (CCS)}},
publisher = {ACM},
year = {2021}
}
```

# Boundary Distance Attack

Modify the parameters in main function.

Run:
```shell
python main.py
```

## NOTE

Run this code may cause:
```
File "/home/xinyuyang/miniconda3/envs/py4ml/lib/python3.9/site-packages/torch/utils/tensorboard/__init__.py", line 4, in <module>
      LooseVersion = distutils.version.LooseVersion
AttributeError: module 'distutils' has no attribute 'version'
```

Following [this instruction](https://github.com/pytorch/pytorch/issues/69894), just downgrading `setuptools` by `pip install setuptools==59.5.0` could resolve this problem.


