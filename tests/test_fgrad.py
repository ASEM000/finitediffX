import functools as ft

import jax
import jax.numpy as jnp
import numpy as np
import numpy as onp
import numpy.testing as npt
import pytest
from jax.experimental import enable_x64

from finitediffx import (
    Offset,
    define_fdjvp,
    fgrad,
    generate_finitediff_coeffs,
    value_and_fgrad,
)


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


def test_fgrad_multiple_step_sizes():
    with enable_x64():
        # test multiple step sizes
        func = lambda x, y: (x + y) ** 2
        dfunc = fgrad(
            func,
            step_size=(None, 1e-3),
            offsets=(jnp.array([-1, 1.0]), jnp.array([-2, 2.0])),
            argnums=(0, 1),
        )

        dval = dfunc(1.0, 1.0)
        assert dval[0] != dval[1]  # different step sizes
        npt.assert_allclose(dval[0], 4.0, atol=1e-3)
        npt.assert_allclose(dval[1], 4.0, atol=1e-3)

    with pytest.raises(TypeError):
        dfunc = fgrad(
            func,
            step_size=(None, 1e-3),
            offsets=(jnp.array([-1, 1.0]), jnp.array([-2, 2.0])),
            argnums=0,
        )(
            1.0, 2.0
        )  # non-tuple argnums with tuple step_size

    with pytest.raises(AssertionError):
        # mismatched argnums length and step_size length
        dfunc = fgrad(func, step_size=(None,), argnums=(0, 1))

    with pytest.raises(AssertionError):
        # mismatched argnums length and step_size length
        dfunc = fgrad(func, offsets=(jnp.array([-1, 1.0]),), argnums=(0, 1))

    with pytest.raises(ValueError):
        # wrong accuracy
        dfunc = fgrad(
            func,
            offsets=(Offset(accuracy=0), Offset(accuracy=1)),
            argnums=(0, 1),
        )(1.0, 1.0)

    with pytest.raises(ValueError):
        # wrong accuracy
        dfunc = fgrad(
            func, offsets=(Offset(accuracy=2), Offset(accuracy=1)), argnums=(0, 1)
        )
        dfunc(1.0, 1.0)


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


def test_has_aux():
    def func(x):
        return x**2, "value"

    v, a = fgrad(func, has_aux=True)(1.0)

    assert v == 2.0
    assert a == "value"


def test_fgrad_error():
    with pytest.raises(TypeError):
        fgrad(lambda x: x, argnums=1.0)

    with pytest.raises(ValueError):
        fgrad(lambda x: x, offsets=Offset(accuracy=1))(1.0)

    with pytest.raises(TypeError):
        fgrad(lambda x: x, argnums=[1, 2])(1.0)


def test_fdjvp():
    def wrap_pure_callback(func):
        @ft.wraps(func)
        def wrapper(*args, **kwargs):
            args = [jnp.asarray(arg) for arg in args]
            func_ = lambda *a, **k: func(*a, **k).astype(a[0].dtype)
            dtype_ = jax.ShapeDtypeStruct(
                jnp.broadcast_shapes(*[ai.shape for ai in args]),
                args[0].dtype,
            )
            return jax.pure_callback(func_, dtype_, *args, **kwargs, vectorized=True)

        return wrapper

    @jax.jit
    @define_fdjvp
    @wrap_pure_callback
    def numpy_func(x, y):
        return onp.power(x, 2) + onp.power(y, 2)

    npt.assert_allclose(
        jax.grad(numpy_func, argnums=0)(1.0, 2.0),
        jnp.array(2.0),
        atol=1e-3,
    )

    npt.assert_allclose(
        jax.grad(numpy_func, argnums=1)(1.0, 2.0),
        jnp.array(4.0),
        atol=1e-3,
    )


def test_value_and_fgrad():
    def func(x):
        return x**2, "value"

    assert value_and_fgrad(func, has_aux=True)(1.0) == ((1.0, "value"), 2.0)

    with pytest.raises(TypeError):
        value_and_fgrad(func, has_aux="")(1.0)

    v, g = value_and_fgrad(func, has_aux=True, argnums=(0,))(1.0)

    assert v == (1.0, "value")
    assert g[0] == jnp.array(2.0)


def test_fgrad_pytree():
    params = {"a": 1.0, "b": 2.0, "c": 3.0}

    def f(params):
        return params["a"] ** 2 + params["b"]

    dparams_fdx = fgrad(f)(params)
    dparams_jax = jax.grad(f)(params)

    npt.assert_allclose(dparams_fdx["a"], dparams_jax["a"], atol=1e-3)
    npt.assert_allclose(dparams_fdx["b"], dparams_jax["b"], atol=1e-3)
    npt.assert_allclose(dparams_fdx["c"], dparams_jax["c"], atol=1e-3)

    step_size = {"a": 1, "b": 1, "c": 1}
    offsets = {
        "a": jnp.array([-1, 1]),
        "b": jnp.array([-1, 1]),
        "c": jnp.array([-1, 1]),
    }

    dparams_fdx = fgrad(f, step_size=step_size, offsets=offsets)(params)
    dparams_jax = jax.grad(f)(params)

    npt.assert_allclose(dparams_fdx["a"], dparams_jax["a"], atol=1e-3)
    npt.assert_allclose(dparams_fdx["b"], dparams_jax["b"], atol=1e-3)
    npt.assert_allclose(dparams_fdx["c"], dparams_jax["c"], atol=1e-3)

    step_size = {"a": 1, "b": 1, "c": 0}
    dparams_fdx = fgrad(f, step_size=step_size)(params)

    # divide by zero
    assert jnp.isnan(dparams_fdx["c"])

    offsets = {"a": jnp.array([0, 0]), "b": jnp.array([-1, 1]), "c": jnp.array([-1, 1])}
    dparams_fdx = fgrad(f, offsets=offsets)(params)

    # generating coefficients for a will fail
    assert jnp.isnan(dparams_fdx["a"])

    (dparams_fdx,) = fgrad(f, argnums=(0,))(params)
    (dparams_jax,) = jax.grad(f, argnums=(0,))(params)

    npt.assert_allclose(dparams_fdx["a"], dparams_jax["a"], atol=1e-3)
    npt.assert_allclose(dparams_fdx["b"], dparams_jax["b"], atol=1e-3)
    npt.assert_allclose(dparams_fdx["c"], dparams_jax["c"], atol=1e-3)

    step_size = {"a": 1, "b": 1, "c": 1}
    offsets = {
        "a": jnp.array([-1, 1]),
        "b": jnp.array([-1, 1]),
        "c": jnp.array([-1, 1]),
    }

    (dparams_fdx,) = fgrad(f, step_size=(step_size,), offsets=(offsets,), argnums=(0,))(
        params
    )
    dparams_jax = jax.grad(f)(params)

    npt.assert_allclose(dparams_fdx["a"], dparams_jax["a"], atol=1e-3)
    npt.assert_allclose(dparams_fdx["b"], dparams_jax["b"], atol=1e-3)
    npt.assert_allclose(dparams_fdx["c"], dparams_jax["c"], atol=1e-3)
