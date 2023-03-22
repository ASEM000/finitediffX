# credits to Mahmoud Asem 2022 @KAIST

# This file defines the fgrad function which is a finite difference approximation of jax.grad
# for first order derivative, finite difference and automatic difference is approximated as the following
# f(x + Δx) = f(x) + f'(x) Δx + f''(x) Δx^2 / 2 + f'''(x) Δx^3 / 6 + ...
# f(x + ɛ Δx) = f(x) + f'(x) ɛ Δx + f''(x) ɛ^2 Δx^2 / 2 + f'''(x) ɛ^3 Δx^3 / 6 + ... , where ɛ^2 := 0 and ɛ != 0


from __future__ import annotations

import functools as ft
from typing import Callable

import jax
import jax.numpy as jnp

from finitediffx._src.utils import _generate_central_offsets, generate_finitediff_coeffs


def _evaluate_func_at_shifted_steps_along_argnum(
    func: Callable,
    *,
    coeffs: jax.Array,
    offsets: tuple[float | int, ...],
    argnum: int,
    step_size: float,
    derivative: int,
):
    if not isinstance(argnum, int) or argnum < 0:
        raise ValueError(f"argnum must be a non-negative integer, got {argnum}")

    DX = jnp.array(offsets) * step_size

    def wrapper(*args, **kwargs):
        # yield function output at shifted points
        for coeff, dx in zip(coeffs, DX):
            shifted_args = list(args)
            shifted_args[argnum] += dx
            yield coeff * func(*shifted_args, **kwargs) / (step_size**derivative)

    return wrapper


def fgrad(
    func: Callable,
    *,
    argnums: int = 0,
    step_size: float = None,
    offsets: tuple[float | int, ...] = None,
    derivative: int = 1,
    accuracy: int = 3,
) -> Callable:
    """Finite difference derivative of a function with respect to one of its arguments.
    similar to jax.grad but with finite difference approximation

    This function could be useful in certain situations for example
    >>> import jax
    >>> jax.config.update("jax_enable_x64", True)
    >>> f = lambda x : x**10
    >>> def repeated_grad(f, n):
    ...     return f if n==0 else jax.grad(repeated_grad(f, n-1))

    >>> df_grad = jax.jit(repeated_grad(f,10))
    >>> %timeit df_grad(1.)
    4.83 µs ± 4.3 µs per loop (mean ± std. dev. of 7 runs, 1 loop each)

    >>> df_fgrad = jax.jit(fgrad(f, argnums=0, derivative=10, accuracy=2))
    >>> %timeit df_fgrad(1.)
    2.45 µs ± 63.7 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)

    Args:
        func: function to differentiate
        argnums:
            argument number to differentiate. Defaults to 0.
            If a tuple is passed, the function is differentiated with respect to all the arguments in the tuple.
        step_size: step size for the finite difference stencil. Defaults to None.
        offsets: offsets for the finite difference stencil. Defaults to None.
        derivative: derivative order. Defaults to 1.
        accuracy: accuracy of the finite difference stencil. Defaults to 2. used to generate offsets if not provided.

    Returns:
        Callable: derivative of the function

    Example:
        >>> def f(x):
        ...     return x**2
        >>> df = fgrad(f)
        >>> df(2.0)
        DeviceArray(4., dtype=float32)
    """
    if offsets is None:
        # if offsets is not provided, generate them based on accuracy
        # otherwise, use the provided offsets and discard accuracy
        if accuracy < 2:
            raise ValueError(f"accuracy must be >= 2, got {accuracy}")
        # generate central offsets based on accuracy
        offsets = _generate_central_offsets(derivative, accuracy=accuracy)

    if step_size is None:
        # generate step size based on accuracy
        # the best step size = 2**(-23 / (2 * derivative))
        step_size = (2) ** (-23 / (2 * derivative))

    # finite difference coefficients
    coeffs = generate_finitediff_coeffs(offsets, derivative)

    # TODO: edit docstring of the differentiated function

    if isinstance(argnums, int):
        # fgrad(func, argnums=0)
        dfunc = _evaluate_func_at_shifted_steps_along_argnum(
            func,
            coeffs=coeffs,
            offsets=offsets,
            argnum=argnums,
            step_size=step_size,
            derivative=derivative,
        )
        return ft.wraps(func)(lambda *a, **k: sum(dfunc(*a, **k)))

    if isinstance(argnums, tuple):
        # fgrad(func, argnums=(0,1))
        # return a tuple of derivatives evaluated at each argnum
        # this behavior is similar to jax.grad(func, argnums=(...))
        dfuncs = [
            _evaluate_func_at_shifted_steps_along_argnum(
                func,
                coeffs=coeffs,
                offsets=offsets,
                argnum=argnum,
                step_size=step_size,
                derivative=derivative,
            )
            for argnum in argnums
        ]
        return ft.wraps(func)(lambda *a, **k: tuple(sum(df(*a, **k)) for df in dfuncs))

    raise ValueError(f"argnums must be an int or a tuple of ints, got {argnums}")
