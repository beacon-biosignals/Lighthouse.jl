<div align="center">
  <img src="docs/src/assets/logo.svg"
       alt="Lighthouse.jl"
       height="128"
       width="128">
</div>

# Lighthouse.jl

[![Build Status](https://travis-ci.com/beacon-biosignals/Lighthouse.jl.svg?token=m6ugwCxo8pHBxpMeiNTu&branch=main)](https://travis-ci.com/beacon-biosignals/Lighthouse.jl)
[![codecov](https://codecov.io/gh/beacon-biosignals/Lighthouse.jl/branch/main/graph/badge.svg?token=8DnNEbLw2x)](https://codecov.io/gh/beacon-biosignals/Lighthouse.jl)
[![Docs: stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://beacon-biosignals.github.io/Lighthouse.jl/stable)
[![Docs: development](https://img.shields.io/badge/docs-dev-blue.svg)](https://beacon-biosignals.github.io/Lighthouse.jl/dev)


Lighthouse.jl is a Julia package that standardizes and automates performance evaluation for multiclass, multirater classification models. By implementing a minimal interface, your classifier automagically gains a thoroughly instrumented training/testing harness (`Lighthouse.learn!`) that computes and logs tons of meaningful performance metrics to TensorBoard in real-time, including:

- test set loss
- inter-rater agreement (e.g. Cohen's Kappa)
- PR curves
- ROC curves
- calibration curves

Lighthouse itself is framework-agnostic; end-users should use whichever extension package matches their desired framework (e.g. https://github.com/beacon-biosignals/LighthouseFlux.jl).

This package follows the [YASGuide](https://github.com/jrevels/YASGuide).

## Installation

To install Lighthouse for development, run:

```
julia -e 'using Pkg; Pkg.develop(PackageSpec(url="https://github.com/beacon-biosignals/Lighthouse.jl"))'
```

This will install Lighthouse to the default package development directory, `~/.julia/dev/Lighthouse`.

### TensorBoard

Note that Lighthouse logs metrics to a user-specified path in [TensorBoard's](https://github.com/tensorflow/tensorboard) `logdir` format. TensorBoard can be installed via `python3 -m pip install tensorboard` (note: if you have `tensorflow>=1.14`, you should already have `tensorboard`). Once TensorBoard is installed, you can view Lighthouse-generated metrics via `tensorboard --logdir path` where `path` is the path specified by `Lighthouse.LearnLogger`. From there, TensorBoard itself can be used/configured however you like; see https://github.com/tensorflow/tensorboard for more information.
