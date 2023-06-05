# Copyright 2023 FiniteDiffX authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import functools as ft
from typing import Callable, NamedTuple, TypeVar, Union

import jax
import jax.numpy as jnp
from typing_extensions import ParamSpec

from finitediffx._src.utils import _generate_central_offsets, generate_finitediff_coeffs

P = ParamSpec("P")
T = TypeVar("T")


class Offset(NamedTuple):
    """Convinience class for finite difference offsets used inside `fgrad`"""

    accuracy: int


OffsetType = Union[jax.Array, Offset]
StepsizeType = Union[jax.Array, float]


def _evaluate_func_at_shifted_steps_along_argnum(
    func: Callable[P, T],
    *,
    coeffs: jax.Array,
    offsets: jax.Array,
    argnum: int,
    step_size: StepsizeType,
    derivative: int,
):
    if not isinstance(argnum, int) or argnum < 0:
        raise ValueError(f"argnum must be a non-negative integer, got {argnum}")

    dxs = offsets * step_size

    def wrapper(*args, **kwargs):
        # yield function output at shifted points
        for coeff, dx in zip(coeffs, dxs):
            shifted_args = list(args)
            shifted_args[argnum] += dx
            yield coeff * func(*shifted_args, **kwargs) / (step_size**derivative)

    return wrapper


def _canonicalize_step_size(
    step_size: StepsizeType | tuple[StepsizeType, ...] | None,
    length: int | None,
    derivative: int,
) -> tuple[StepsizeType, ...] | StepsizeType:
    # return non-tuple values if length is None
    if isinstance(step_size, (jax.Array, float)):
        return ((step_size,) * length) if length else step_size
    if step_size is None:
        default = (2) ** (-23 / (2 * derivative))
        return ((default,) * length) if length else default

    if isinstance(step_size, tuple) and length:
        assert len(step_size) == length, f"step_size must be a tuple of length {length}"
        step_size = list(step_size)
        for i, s in enumerate(step_size):
            if s is None:
                step_size[i] = (2) ** (-23 / (2 * derivative))
            elif not isinstance(s, (jax.Array, float)):
                raise TypeError(f"{type(s)} not in {(jax.Array, float)}")
        return tuple(step_size)

    raise TypeError(
        f"`step_size` must be of type:\n"
        f"- `jax.Array`\n"
        f"- `float`\n"
        f"- tuple of `jax.Array` or `float` for tuple argnums.\n"
        f"but got {type(step_size)=}"
    )


def _canonicalize_offsets(
    offsets: tuple[OffsetType | None, ...] | OffsetType | None,
    length: int,
    derivative: int,
) -> tuple[OffsetType, ...] | OffsetType:
    # single value
    if isinstance(offsets, Offset):
        if offsets.accuracy < 2:
            raise ValueError(f"offset accuracy must be >=2, got {offsets.accuracy}")
        offsets = jnp.array(_generate_central_offsets(derivative, offsets.accuracy))
        return ((offsets,) * length) if length else offsets
    if isinstance(offsets, jax.Array):
        return ((offsets,) * length) if length else offsets

    if isinstance(offsets, tuple) and length:
        assert len(offsets) == length, f"`offsets` must be a tuple of length {length}"
        offsets = list(offsets)
        for i, o in enumerate(offsets):
            if isinstance(o, Offset):
                if o.accuracy < 2:
                    raise ValueError(f"offset accuracy must be >=2, got {o.accuracy}")
                o = jnp.array(_generate_central_offsets(derivative, o.accuracy))
                offsets[i] = o
            elif not isinstance(o, (Offset, jax.Array)):
                raise TypeError(f"{type(o)} not an Offset")

        return tuple(offsets)

    raise TypeError(
        f"`offsets` must be of type:\n"
        f"- `Offset`\n"
        f"- `jax.Array`\n"
        f"- tuple of `Offset` or `jax.Array` for tuple argnums.\n"
        f"but got {type(offsets)=}"
    )


def fgrad(
    func: Callable,
    *,
    argnums: int | tuple[int, ...] = 0,
    step_size: StepsizeType | None = None,
    offsets: OffsetType = Offset(accuracy=3),
    derivative: int = 1,
) -> Callable:
    """Finite difference derivative of a function with respect to one of its arguments.
    similar to jax.grad but with finite difference approximation

    Args:
        func: function to differentiate
        argnums: argument number to differentiate. Defaults to 0.
            If a tuple is passed, the function is differentiated with respect to
            all the arguments in the tuple.
        step_size: step size for the finite difference stencil. If `None`, the step size
            is set to `(2) ** (-23 / (2 * derivative))`
        offsets: offsets for the finite difference stencil. Accepted types are:
            - `jax.Array` defining location of function evaluation points.
            - `Offset` with accuracy field to automatically generate offsets.
        derivative: derivative order. Defaults to 1.

    Returns:
        Callable: derivative of the function

    Example:
        >>> import finitediffx as fdx
        >>> import jax
        >>> import jax.numpy as jnp

        >>> def f(x, y):
        ...    return x**2 + y**2

        >>> df=fdx.fgrad(
        ...    func=f,
        ...    argnums=1,  # differentiate with respect to y
        ...    offsets=fdx.Offset(accuracy=2)  # use 2nd order accurate finite difference
        ... )

        >>> df(2.0, 3.0)
        Array(6., dtype=float32)

    """
    func.__doc__ = (
        f"Finite difference derivative of {func.__name__} w.r.t {argnums=}"
        f"\n\n{func.__doc__}"
    )

    if isinstance(argnums, int):
        # fgrad(func, argnums=0)
        kwargs = dict(length=None, derivative=derivative)
        step_size = _canonicalize_step_size(step_size, **kwargs)
        offsets = _canonicalize_offsets(offsets, **kwargs)
        dfunc = _evaluate_func_at_shifted_steps_along_argnum(
            func=func,
            coeffs=generate_finitediff_coeffs(offsets, derivative),
            offsets=offsets,
            step_size=step_size,
            derivative=derivative,
            argnum=argnums,
        )

        return ft.wraps(func)(lambda *a, **k: sum(dfunc(*a, **k)))

    if isinstance(argnums, tuple):
        # fgrad(func, argnums=(0,1))
        # return a tuple of derivatives evaluated at each argnum
        # this behavior is similar to jax.grad(func, argnums=(...))
        kwargs = dict(length=len(argnums), derivative=derivative)

        dfuncs = [
            _evaluate_func_at_shifted_steps_along_argnum(
                func=func,
                coeffs=generate_finitediff_coeffs(offsets_i, derivative),
                offsets=offsets_i,
                step_size=step_size_i,
                derivative=derivative,
                argnum=argnum_i,
            )
            for offsets_i, step_size_i, argnum_i in zip(
                _canonicalize_offsets(offsets, **kwargs),
                _canonicalize_step_size(step_size, **kwargs),
                argnums,
            )
        ]
        return ft.wraps(func)(lambda *a, **k: tuple(sum(df(*a, **k)) for df in dfuncs))

    raise ValueError(f"argnums must be an int or a tuple of ints, got {argnums}")


def define_fdjvp(
    func: Callable[P, T],
    offsets: tuple[OffsetType, ...] | OffsetType = Offset(accuracy=2),
    step_size: tuple[float, ...] | float | None = None,
) -> Callable[P, T]:
    """Define the JVP rule for a function using finite difference.

    Args:
        func: function to define the JVP rule for
        offsets: offsets for the finite difference stencil. Accepted types are:
            - `jax.Array` defining location of function evaluation points.
            - `Offset` with accuracy field to automatically generate offsets.
            - tuple of `Offset` or `jax.Array` defining offsets for each argument.
        step_size: step size for the finite difference stencil. Accepted types are:
            - `float` defining the step size for all arguments.
            - `tuple` of `float` defining the step size for each argument.
            - `None` to use the default step size for each argument.

    Returns:
        Callable: function with JVP rule defined using finite difference.

    Note:
        - This function is motivated by [`JEP`](https://github.com/google/jax/issues/15425)
        - This function can be used with `jax.pure_callback` to define the JVP
            rule for a function that is not differentiable by JAX.

        Example:
            >>> import jax
            >>> import jax.numpy as jnp
            >>> import finitediffx as fdx
            >>> import functools as ft
            >>> import numpy as onp
            >>> def wrap_pure_callback(func):
            ...     @ft.wraps(func)
            ...     def wrapper(*args, **kwargs):
            ...         args = [jnp.asarray(arg) for arg in args]
            ...         func_ = lambda *a, **k: func(*a, **k).astype(a[0].dtype)
            ...         dtype_ = jax.ShapeDtypeStruct(
            ...             jnp.broadcast_shapes(*[ai.shape for ai in args]),
            ...             args[0].dtype,
            ...         )
            ...         return jax.pure_callback(func_, dtype_, *args, **kwargs, vectorized=True)
            ...     return wrapper

            >>> @jax.jit
            ... @jax.grad
            ... @fdx.define_fdjvp
            ... @wrap_pure_callback
            ... def numpy_func(x, y):
            ...     return onp.sin(x) + onp.cos(y)
            >>> print(numpy_func(1., 2.))
            0.5402466
    """
    func = jax.custom_jvp(func)

    @func.defjvp
    def _(primals, tangents):
        kwargs = dict(length=len(primals), derivative=1)
        step_size_ = _canonicalize_step_size(step_size, **kwargs)
        offsets_ = _canonicalize_offsets(offsets, **kwargs)
        primal_out = func(*primals)
        tangent_out = sum(
            fgrad(func, argnums=i, step_size=si, offsets=oi)(*primals) * ti
            for i, (si, oi, ti) in enumerate(zip(step_size_, offsets_, tangents))
        )
        return jnp.array(primal_out), jnp.array(tangent_out)

    return func
