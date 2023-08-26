<h5 align="center">
<img width="200px" src="assets/logo.svg"> <br>

<br>

[**Installation**](#installation)
|[**Examples**](#examples)

![Tests](https://github.com/ASEM000/pytreeclass/actions/workflows/tests.yml/badge.svg)
![pyver](https://img.shields.io/badge/python-3.8%203.9%203.10%203.11_-red)
![pyver](https://img.shields.io/badge/jax-0.4+-red)
![codestyle](https://img.shields.io/badge/codestyle-black-black)
[![Downloads](https://static.pepy.tech/badge/FiniteDiffX)](https://pepy.tech/project/FiniteDiffX)
[![codecov](https://codecov.io/github/ASEM000/FiniteDiffX/branch/main/graph/badge.svg?token=VD45Y4HLWV)](https://codecov.io/github/ASEM000/FiniteDiffX)  
[![Documentation Status](https://readthedocs.org/projects/finitediffx/badge/?version=latest)](https://finitediffx.readthedocs.io/en/latest/?badge=latest)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ASEM000/FiniteDiffX/blob/main/FiniteDiffX%20Examples.ipynb)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/ASEM000/FiniteDiffX)
![PyPI](https://img.shields.io/pypi/v/FiniteDiffX)
[![CodeFactor](https://www.codefactor.io/repository/github/asem000/finitediffx/badge)](https://www.codefactor.io/repository/github/asem000/finitediffx)

</h5>

Differentiable finite difference tools in `jax`

## 🛠️ Installation<a id="installation"></a>

```python
pip install FiniteDiffX
```

**Install development version**

```python
pip install git+https://github.com/ASEM000/FiniteDiffX
```

**If you find it useful to you, consider giving it a star! 🌟**

<br>

## ⏩ Quick Example<a id="examples"></a>

```python
import jax.numpy as jnp
import finitediffx as fdx

# lets first define a vector valued function F: R^3 -> R^3
# F = F1, F2
# F1 = x^2 + y^3
# F2 = x^4 + y^3
# F3 = 0
# F = [x**2 + y**3, x**4 + y**3, 0]

x, y, z = [jnp.linspace(0, 1, 100)] * 3
dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
F1 = X**2 + Y**3
F2 = X**4 + Y**3
F3 = jnp.zeros_like(F1)
F = jnp.stack([F1, F2, F3], axis=0)

# ∇.F : the divergence of F
divF = fdx.divergence(
    F,
    step_size=(dx, dy, dz),
    keepdims=False,
    accuracy=6,
    method="central",
)
```

`jax.grad`, `jax.value_and_grad` finite difference counterpart to be used on
[unimplemented rules in `jax`](https://github.com/google/jax/discussions/16584) or
[on non-traceable `numpy` code](https://github.com/google/jax/issues/15425)

```python

import jax
from jax import numpy as jnp
import numpy as onp  # Not jax-traceable
import finitediffx as fdx
import functools as ft
from jax.experimental import enable_x64

with enable_x64():

    @fdx.fgrad
    @fdx.fgrad
    def np_rosenbach2_fdx_style_1(x, y):
        """Compute the Rosenbach function for two variables in numpy."""
        return onp.power(1-x, 2) + 100*onp.power(y-onp.power(x, 2), 2)

    @ft.partial(fdx.fgrad, derivative=2)
    def np2_rosenbach2_fdx_style2(x, y):
        """Compute the Rosenbach function for two variables."""
        return onp.power(1-x, 2) + 100*onp.power(y-onp.power(x, 2), 2)

    @jax.grad
    @jax.grad
    def jnp_rosenbach2(x, y):
        """Compute the Rosenbach function for two variables."""
        return jnp.power(1-x, 2) + 100*jnp.power(y-jnp.power(x, 2), 2)

    print(np_rosenbach2_fdx_style_1(1.,2.))
    print(np2_rosenbach2_fdx_style2(1.,2.))
    print(jnp_rosenbach2(1., 2.))
# 402.0000951997936
# 402.0000000002219
# 402.0

```
