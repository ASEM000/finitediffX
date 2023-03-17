<h5 align="center">
<img width="200px" src="assets/finitediffx_logo.svg"> <br>

<br>

[**Installation**](#Installation)
|[**Examples**](#Examples)

![Tests](https://github.com/ASEM000/pytreeclass/actions/workflows/tests.yml/badge.svg)
![pyver](https://img.shields.io/badge/python-3.8%203.8%203.9%203.11-red)
![codestyle](https://img.shields.io/badge/codestyle-black-lightgrey)
[![Downloads](https://pepy.tech/badge/FiniteDiffX)](https://pepy.tech/project/FiniteDiffX)
[![codecov](https://codecov.io/github/ASEM000/FiniteDiffX/branch/main/graph/badge.svg?token=VD45Y4HLWV)](https://codecov.io/github/ASEM000/FiniteDiffX)    
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ASEM000/FiniteDiffX/blob/main/FiniteDiffX%20Examples.ipynb) 
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/ASEM000/FiniteDiffX)
![PyPI](https://img.shields.io/pypi/v/FiniteDiffX)

</h5>

Differentiable finite difference tools in `jax`
Implements :

- `difference(array, axis, accuracy, step_size, method, derivative)`
- `gradient(array, accuracy, method, step_size)`
- `jacobian(array, accuracy, method, step_size)`
- `divergence(array, accuracy, step_size, method, keepdims)`
- `hessian(array, accuracy, method, step_size)`
- `laplacian(array, accuracy, method, step_size)`
- `curl(array, step_size, method, keep_dims)`

## 🛠️ Installation<a id="Installation"></a>

```python
pip install FiniteDiffX
```

**Install development version**

```python
pip install git+https://github.com/ASEM000/FiniteDiffX
```

## ⏩ Examples<a id="Examples"></a>

```python

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy.testing as npt

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

# ∂F1/∂x : differentiate F1 with respect to x (i.e axis=0)
dF1dx = fdx.difference(F1, axis=0, step_size=dx, accuracy=6, method="central")
dF1dx_exact = 2 * X
npt.assert_allclose(dF1dx, dF1dx_exact, atol=1e-7)

# ∂F2/∂y : differentiate F2 with respect to y (i.e axis=1)
dF2dy = fdx.difference(F2, axis=1, step_size=dy, accuracy=6)
dF2dy_exact = 3 * Y**2
npt.assert_allclose(dF2dy, dF2dy_exact, atol=1e-7)

# ∇.F : the divergence of F
divF = fdx.divergence(F, step_size=(dx, dy, dz), keepdims=False, accuracy=6, method="central")
divF_exact = 2 * X + 3 * Y**2
npt.assert_allclose(divF, divF_exact, atol=1e-7)

# ∇F1 : the gradient of F1
gradF1 = fdx.gradient(F1, step_size=(dx, dy, dz), accuracy=6, method="central")
gradF1_exact = jnp.stack([2 * X, 3 * Y**2, 0 * X], axis=0)
npt.assert_allclose(gradF1, gradF1_exact, atol=1e-7)

# ΔF1 : laplacian of F1
lapF1 = fdx.laplacian(F1, step_size=(dx, dy, dz), accuracy=6, method="central")
lapF1_exact = 2 + 6 * Y
npt.assert_allclose(lapF1, lapF1_exact, atol=1e-7)

# ∇xF : the curl of F
curlF = fdx.curl(F, step_size=(dx, dy, dz), accuracy=6, method="central")
curlF_exact = jnp.stack([F1 * 0, F1 * 0, 4 * X**3 - 3 * Y**2], axis=0)
npt.assert_allclose(curlF, curlF_exact, atol=1e-7)

# Jacobian of F
JF = fdx.jacobian(F, accuracy=4, step_size=(dx, dy, dz), method="central")
JF_exact = jnp.array(
    [
        [2 * X, 3 * Y**2, jnp.zeros_like(X)],
        [4 * X**3, 3 * Y**2, jnp.zeros_like(X)],
        [jnp.zeros_like(X), jnp.zeros_like(X), jnp.zeros_like(X)],
    ]
)
npt.assert_allclose(JF, JF_exact, atol=1e-7)

# Hessian of F1
HF1 = fdx.hessian(F1, accuracy=4, step_size=(dx, dy, dz), method="central")
HF1_exact = jnp.array(
    [
        [
            2 * jnp.ones_like(X),  # ∂2F1/∂x2
            0 * jnp.ones_like(X),  # ∂2F1/∂xy
            0 * jnp.ones_like(X),  # ∂2F1/∂xz
        ],
        [
            0 * jnp.ones_like(X),  # ∂2F1/∂yx
            6 * Y**2,              # ∂2F1/∂y2
            0 * jnp.ones_like(X),  # ∂2F1/∂yz
        ],
        [
            0 * jnp.ones_like(X),  # ∂2F1/∂zx
            0 * jnp.ones_like(X),  # ∂2F1/∂zy
            0 * jnp.ones_like(X),  # ∂2F1/∂z2
        ],
    ]
)
npt.assert_allclose(JF, JF_exact, atol=1e-7)




```
