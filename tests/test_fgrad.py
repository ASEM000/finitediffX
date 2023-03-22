import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64

from finitediffx import fgrad, generate_finitediff_coeffs


def test_generate_finitediff_coeffs():
    with enable_x64():
        DF = lambda N: generate_finitediff_coeffs(N, 1)
        all_correct = lambda x, y: np.testing.assert_allclose(x, y, atol=1e-2)

        # https://en.wikipedia.org/wiki/fgraderence_coefficient
        all_correct(DF((0, 1)), jnp.array([-1.0, 1.0]))
        all_correct(DF((0, 1, 2)), jnp.array([-3 / 2.0, 2.0, -1 / 2]))
        all_correct(DF((0, 1, 2, 3)), jnp.array([-11 / 6, 3.0, -3 / 2, 1 / 3]))
        all_correct(DF((0, 1, 2, 3, 4)), jnp.array([-25 / 12, 4.0, -3.0, 4 / 3, -1 / 4]))  # fmt: skip
        all_correct(DF((0, 1, 2, 3, 4, 5)), jnp.array([-137 / 60, 5.0, -5, 10 / 3, -5 / 4, 1 / 5]),)  # fmt: skip
        all_correct(
            DF((0, 1, 2, 3, 4, 5, 6)),
            jnp.array([-49 / 20, 6.0, -15 / 2, 20 / 3, -15 / 4, 6 / 5, -1 / 6]),
        )

        # https://web.media.mit.edu/~crtaylor/calculator.html
        all_correct(DF((-2.2, 3.2, 5)), jnp.array([-205 / 972, 280 / 972, -75 / 972]))

        # https://web.njit.edu/~jiang/math712/fornberg.pdf  Table 4.
        DF = lambda N: generate_finitediff_coeffs(N, 0)
        all_correct(DF((-0.5,)), jnp.array([1]))
        all_correct(DF((-0.5, 0.5)), jnp.array([0.5, 0.5]))
        all_correct(DF((-0.5, 0.5, 1.5)), jnp.array([3 / 8, 3 / 4, -1 / 8]))
        all_correct(
            DF((-0.5, 0.5, 1.5, 2.5)), jnp.array([5 / 16, 15 / 16, -5 / 16, 1 / 16])
        )


def test_fgrad_args():
    with enable_x64():
        all_correct = lambda lhs, rhs: np.testing.assert_allclose(lhs, rhs, atol=0.05)

        for func in [
            lambda x: x,
            lambda x: x**2,
            lambda x: x + 2,
            lambda x: jnp.sin(x),
            lambda x: jnp.cos(x) * jnp.sin(x),
        ]:
            f1 = jax.grad(func)
            f2 = fgrad(func)
            args = (1.5,)
            F1, F2 = f1(*args), f2(*args)
            all_correct(F1, F2)

        for func in [
            lambda x, y: x + y,
            lambda x, y: x**2 + y**3,
            lambda x, y: x + 2 + y * x,
        ]:
            f1 = jax.grad(func)
            f2 = fgrad(func)
            args = (1.5, 2.5)
            F1, F2 = f1(*args), f2(*args)
            all_correct(F1, F2)

        for func in [
            lambda x, y, z: x + y + z,
            lambda x, y, z: x**2 + y**3 * z,
            lambda x, y, z: x + 2 + y * x + z,
        ]:
            f1 = jax.grad(func)
            f2 = fgrad(func)
            args = (1.5, 2.5, 3.5)
            F1, F2 = f1(*args), f2(*args)
            all_correct(F1, F2)


def test_fgrad_second_derivative():
    with enable_x64():
        all_correct = lambda lhs, rhs: np.testing.assert_allclose(lhs, rhs, atol=0.05)

        for func in [lambda x: x, lambda x: x**2, lambda x: x + 2]:
            f1 = jax.grad(jax.grad(func))
            f2 = fgrad(func, derivative=2)
            f3 = fgrad(fgrad(func))
            args = (1.5,)
            F1, F2, F3 = f1(*args), f2(*args), f3(*args)
            all_correct(F1, F2)
            all_correct(F1, F3)

        for func in [
            lambda x, y: x + y,
            lambda x, y: x**2 + y**3,
            lambda x, y: x + 2 + y * x,
        ]:
            f1 = jax.grad(jax.grad(func))
            f2 = fgrad(func, derivative=2)
            f3 = fgrad(fgrad(func))
            args = (1.5, 2.5)
            F1, F2, F3 = f1(*args), f2(*args), f3(*args)
            all_correct(F1, F2)
            all_correct(F1, F3)

        for func in [
            lambda x, y, z: x + y + z,
            lambda x, y, z: x**2 + y**3 * z,
            lambda x, y, z: x + 2 + y * x + z,
        ]:
            f1 = jax.grad(jax.grad(func))
            f2 = fgrad(func, derivative=2)
            f3 = fgrad(fgrad(func))
            args = (1.5, 2.5, 3.5)
            F1, F2, F3 = f1(*args), f2(*args), f3(*args)
            all_correct(F1, F2)
            all_correct(F1, F3)


def test_fgrad_argnum():
    with enable_x64():
        all_correct = lambda lhs, rhs: np.testing.assert_allclose(lhs, rhs, atol=0.05)

        # test argnums
        func = lambda x, y, z: x**2 + y**3 + z**4
        f1 = jax.grad(func, argnums=(0,))(1.0, 1.0, 1.0)
        f2 = fgrad(func, argnums=0)(1.0, 1.0, 1.0)
        all_correct(f1, f2)

        f1 = jax.grad(func, argnums=(1,))(1.0, 1.0, 1.0)
        f2 = fgrad(func, argnums=1)(1.0, 1.0, 1.0)
        all_correct(f1, f2)

        f1 = jax.grad(func, argnums=(2,))(1.0, 1.0, 1.0)
        f2 = fgrad(func, argnums=2)(1.0, 1.0, 1.0)
        all_correct(f1, f2)

        # multiple argnums
        f1l, f1r = jax.grad(func, argnums=(0, 1))(1.0, 1.0, 1.0)
        f2l, f2r = fgrad(func, argnums=(0, 1))(1.0, 1.0, 1.0)
        all_correct(f1l, f2l)
        all_correct(f1r, f2r)


def test_fgrad_error():
    with pytest.raises(ValueError):
        fgrad(lambda x: x, argnums=-1)

    with pytest.raises(ValueError):
        fgrad(lambda x: x, argnums=1.0)

    with pytest.raises(ValueError):
        fgrad(lambda x: x, accuracy=1)

    with pytest.raises(ValueError):
        fgrad(lambda x: x, argnums=[1, 2])
