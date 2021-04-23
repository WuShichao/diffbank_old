# diffbank: easy gravitational wave template bank generation with automatic differentiation

This package uses the [jax](https://github.com/google/jax) automatic
differentiation package to automate the calculation of the parameter space
metric for many gravitational waveforms. The metric can be used to efficiently
generate template banks using the random template bank algorithm from
[Messenger, Prix & Papa (2009)](https://arxiv.org/abs/0809.5223), which works
in curved spaces.

It's also possible to compute the scalar curvature for the metric with
[diffjeom](https://github.com/adam-coogan/diffjeom) package.

## Installation

Clone the repo, `cd` to this directory, and install with `pip install .` (or
`pip install -e .` for an editable install).
