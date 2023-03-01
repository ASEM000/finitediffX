# import jax
import jax.numpy as jnp
import numpy.testing as npt
import pytest
from jax.experimental import enable_x64

from finitediffx import (
    curl,
    difference,
    divergence,
    gradient,
    hessian,
    jacobian,
    laplacian,
)
from finitediffx._src.utils import (
    _generate_backward_offsets,
    _generate_central_offsets,
    _generate_forward_offsets,
    generate_finitediff_coeffs,
)


def test_difference():
    with enable_x64():
        x, y = [jnp.linspace(0, 1, 1000)] * 2
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        F = jnp.sin(X) * jnp.cos(Y)
        dFdX = difference(F, step_size=dx, axis=0, accuracy=3)
        npt.assert_allclose(dFdX, jnp.cos(X) * jnp.cos(Y), rtol=1e-3)

        dFdY = difference(F, step_size=dy, axis=1, accuracy=3)
        npt.assert_allclose(dFdY, -jnp.sin(X) * jnp.sin(Y), atol=1e-7)

        # 1 4 6 3 2
        # left = (4-1) = 3 -> forward difference
        # center = (6-1)/2, (3-4)/2 -> central difference
        # right = (2-3) = -1 -> backward difference
        npt.assert_allclose(
            difference(jnp.array([1, 4, 6, 3, 2]), accuracy=1, axis=0, derivative=1),
            jnp.array([3, 2.5, -0.5, -2, -1]),
        )

        # 1 4 6 3 2 4
        # [-1.5  2.  -0.5] (0, 1, 2)
        # [-0.5  0.   0.5] (-1, 0, 1)
        # [ 0.5 -2.   1.5] (-2, -1, 0)
        npt.assert_allclose(
            difference(jnp.array([1, 4, 6, 3, 2, 4]), accuracy=2, axis=0, derivative=1),
            jnp.array([3.5, 2.5, -0.5, -2.0, 0.5, 3.5]),
        )

        # no central difference possible
        x = jnp.array([1.2, 1.4])
        npt.assert_allclose(
            difference(x, accuracy=1, axis=0, derivative=1), jnp.array([0.2, 0.2])
        )


def test_divergence():
    with enable_x64():
        x, y = [jnp.linspace(0, 1, 100)] * 2
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        F1 = X**2 + Y**3
        F2 = X**4 + Y**3
        F = jnp.stack([F1, F2], axis=0)  # 2D vector field F = (F1, F2)
        divZ = divergence(F, step_size=(dx, dy), accuracy=7, keepdims=False)
        divZ_true = 2 * X + 3 * Y**2  # (dF1/dx) + (dF2/dy)
        npt.assert_allclose(divZ, divZ_true, atol=5e-7)

        divZ = divergence(F, step_size=(dx, dy), accuracy=7, keepdims=True)
        divZ_true = 2 * X + 3 * Y**2  # (dF1/dx) + (dF2/dy)
        npt.assert_allclose(divZ, divZ_true[None], atol=5e-7)


def test_gradient():
    with enable_x64():
        x, y = [jnp.linspace(0, 1, 100)] * 2
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        F = X**2 + Y**3
        gradF = gradient(F, step_size=(dx, dy), accuracy=7)
        gradF_true = jnp.stack([2 * X, 3 * Y**2], axis=0)
        npt.assert_allclose(gradF, gradF_true, atol=5e-7)


def test_laplacian():
    with enable_x64():
        # needs float64 for higher accuracy
        x, y = [jnp.linspace(0, 1, 100)] * 2
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        Z = X**4 + Y**3
        laplacianZ = laplacian(Z, step_size=(dx, dy), accuracy=10)
        laplacianZ_true = 12 * X**2 + 6 * Y
        npt.assert_allclose(laplacianZ, laplacianZ_true, atol=1e-7)


def test_curl():
    with enable_x64():
        x, y, z = [jnp.linspace(0, 1, 100)] * 3
        dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
        X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        F1 = X**2 + Y**2
        F2 = X**4 + Y**3
        F3 = 0 * Z
        F = jnp.stack([F1, F2, F3], axis=0)
        curl_Z = curl(F, step_size=(dx, dy, dz), accuracy=5)

        F1 = 0 * X
        F2 = 0 * Y
        F3 = 4 * X**3 - 2 * Y
        curl_Z_true = jnp.stack([F1, F2, F3], axis=0)

        npt.assert_allclose(curl_Z, curl_Z_true, atol=1e-7)

        x, y = [jnp.linspace(-1, 1, 50)] * 2
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        F1 = jnp.sin(Y)
        F2 = jnp.cos(X)
        F = jnp.stack([F1, F2], axis=0)
        res = curl(F, accuracy=4, step_size=dx, keepdims=False)
        npt.assert_allclose(res, -jnp.sin(X) - jnp.cos(Y), atol=1e-4)

        res = curl(F, accuracy=4, step_size=dx, keepdims=True)
        npt.assert_allclose(res[0], -jnp.sin(X) - jnp.cos(Y), atol=1e-4)

    with pytest.raises(ValueError):
        curl(F[None], accuracy=4, step_size=dx, keepdims=True)


def test_jacobian():
    with enable_x64():
        x, y = [jnp.linspace(-1, 1, 100)] * 2
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        F1 = X**2 * Y
        F2 = 5 * X + jnp.sin(Y)
        F = jnp.stack([F1, F2], axis=0)
        JF = jacobian(F, accuracy=4, step_size=(dx, dy))
        JF_true = jnp.array([[2 * X * Y, X**2], [5 * jnp.ones_like(X), jnp.cos(Y)]])
        npt.assert_allclose(JF, JF_true, atol=1e-7)

        x1, x2, x3 = [jnp.linspace(-1, 1, 100)] * 3
        dx1, dx2, dx3 = x1[1] - x1[0], x2[1] - x2[0], x3[1] - x3[0]
        X1, X2, X3 = jnp.meshgrid(x1, x2, x3, indexing="ij")
        Y1 = X1
        Y2 = 5 * X3
        Y3 = 4 * X2**2 - 2 * X3
        Y4 = X3 * jnp.sin(X1)
        Y = jnp.stack([Y1, Y2, Y3, Y4], axis=0)
        JY = jacobian(Y, accuracy=5, step_size=(dx1, dx2, dx3))

        JY_true = jnp.array(
            [
                [jnp.ones_like(X1), jnp.zeros_like(X1), jnp.zeros_like(X1)],
                [jnp.zeros_like(X1), jnp.zeros_like(X1), 5 * jnp.ones_like(X1)],
                [jnp.zeros_like(X1), 8 * X2, -2 * jnp.ones_like(X1)],
                [X3 * jnp.cos(X1), jnp.zeros_like(X1), jnp.sin(X1)],
            ]
        )

        npt.assert_allclose(JY, JY_true, rtol=1e-3, atol=1e-5)


def test_hessian():
    with enable_x64():
        x, y = [jnp.linspace(-1, 1, 100)] * 2
        dx, dy = x[1] - x[0], y[1] - y[0]
        X, Y = jnp.meshgrid(x, y, indexing="ij")

        F = X**2 * Y
        H = hessian(F, accuracy=4, step_size=(dx, dy))
        H_true = jnp.array([[2 * Y, 2 * X], [2 * X, jnp.zeros_like(X)]])
        npt.assert_allclose(H, H_true, atol=1e-7)

        F = X**3 + Y**3
        H = hessian(F, accuracy=4, step_size=(dx, dy))
        H_true = jnp.array([[6 * X, jnp.zeros_like(X)], [jnp.zeros_like(X), 6 * Y]])
        npt.assert_allclose(H, H_true, atol=1e-7)

        F = jnp.sin(X) + jnp.cos(Y)
        H = hessian(F, accuracy=4, step_size=(dx, dy))
        H_true = jnp.array(
            [[-jnp.sin(X), jnp.zeros_like(X)], [jnp.zeros_like(X), -jnp.cos(Y)]]
        )
        npt.assert_allclose(H, H_true, atol=1e-7)

        # x, y, z = [jnp.linspace(0, 0.5, 100)] * 3
        # dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
        # X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")
        # F = jnp.sin(X * Y * Z)
        # H = hessian(F, accuracy=5, step_size=(dx, dy, dz))
        # H_true = jnp.array(
        #     [
        #         [
        #             -jnp.sin(X * Y * Z) * Y**2 * Z**2,
        #             -jnp.sin(X * Y * Z) * X * Y * Z**2 + Z * jnp.cos(X * Y * Z),
        #             -jnp.sin(X * Y * Z) * X * Y**2 * Z + Y * jnp.cos(X * Y * Z),
        #         ],
        #         [
        #             -jnp.sin(X * Y * Z) * X * Y * Z**2 + Z * jnp.cos(X * Y * Z),
        #             -jnp.sin(X * Y * Z) * X**2 * Z**2,
        #             -jnp.sin(X * Y * Z) * X**2 * Y * Z + X * jnp.cos(X * Y * Z),
        #         ],
        #         [
        #             -jnp.sin(X * Y * Z) * X * Y**2 * Z + Y * jnp.cos(X * Y * Z),
        #             -jnp.sin(X * Y * Z) * X**2 * Y * Z + X * jnp.cos(X * Y * Z),
        #             -jnp.sin(X * Y * Z) * X**2 * Y**2,
        #         ],
        #     ]
        # )

        # npt.assert_allclose(H, H_true, atol=1e-7)


def test_generate_coeffs():
    with pytest.raises(ValueError):
        _generate_backward_offsets(derivative=0, accuracy=1)
    with pytest.raises(ValueError):
        _generate_backward_offsets(derivative=1, accuracy=0)

    with pytest.raises(ValueError):
        _generate_forward_offsets(derivative=0, accuracy=1)
    with pytest.raises(ValueError):
        _generate_forward_offsets(derivative=1, accuracy=0)

    with pytest.raises(ValueError):
        _generate_central_offsets(derivative=0, accuracy=1)
    with pytest.raises(ValueError):
        _generate_central_offsets(derivative=1, accuracy=0)

    with pytest.raises(ValueError):
        generate_finitediff_coeffs(offsets=[1, 2], derivative=2)


def test_difference_error():
    x = jnp.ones([2])
    with pytest.raises(ValueError):
        difference(x, accuracy=2)
