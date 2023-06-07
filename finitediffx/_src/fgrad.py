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
from typing import Any, Callable, NamedTuple, Sequence, TypeVar, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from typing_extensions import ParamSpec

from finitediffx._src.utils import _generate_central_offsets, generate_finitediff_coeffs

__all__ = ("fgrad", "Offset", "define_fdjvp", "value_and_fgrad")

P = ParamSpec("P")
T = TypeVar("T")
constant_treedef = jtu.tree_structure(0)
PyTree = Any


class Offset(NamedTuple):
    """Convinience class for finite difference offsets used inside `fgrad`

    Args:
        accuracy: The accuracy of the finite difference scheme. Must be >=2.

    Example:
        >>> import finitediffx as fdx
        >>> fdx.fgrad(lambda x: x**2, offsets=fdx.Offset(accuracy=2))(1.0)
        Array(2., dtype=float32)
    """

    accuracy: int


OffsetType = Union[jax.Array, Offset, PyTree]
StepsizeType = Union[jax.Array, float, PyTree]


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
        result = []
        for coeff, dx in zip(coeffs, dxs):
            args_ = list(args)
            args_[argnum] += dx
            result += [coeff * func(*args_, **kwargs) / (step_size**derivative)]
        return sum(result)

    return wrapper


def resolve_step_size(
    step_size: StepsizeType | Sequence[StepsizeType] | None,
    treedef: jtu.PyTreeDef,
    derivative: int,
) -> Sequence[StepsizeType] | StepsizeType:
    # return non-tuple values if length is None
    is_leaf = jtu.treedef_is_leaf(treedef)
    length = treedef.num_leaves

    if isinstance(step_size, (jax.Array, float)):
        return (step_size,) * length

    if step_size is None:
        default = (2) ** (-23 / (2 * derivative))
        return (default,) * length

    if isinstance(step_size, Sequence) and not is_leaf:
        assert len(step_size) == length, f"step_size must be a tuple of length {length}"
        step_size = list(step_size)
        for i, s in enumerate(step_size):
            if s is None:
                step_size[i] = (2) ** (-23 / (2 * derivative))
            elif not isinstance(s, (jax.Array, float)):
                raise TypeError(f"{type(s)} not in {(jax.Array, float)}")
        return tuple(step_size)

    step_size_leaves, step_size_treedef = jtu.tree_flatten(step_size)

    if step_size_treedef == treedef:
        # step_size is a pytree with the same structure as the input
        return step_size_leaves

    raise TypeError(
        f"`step_size` must be of type:\n"
        f"- `jax.Array`\n"
        f"- `float`\n"
        f"- tuple of `jax.Array` or `float` for tuple argnums.\n"
        f"- pytree with the same structure as the desired arg.\n"
        f"but got {type(step_size)=}"
    )


def resolve_offsets(
    offsets: Sequence[OffsetType | None] | OffsetType | None,
    treedef: jax.tree_util.PyTreeDef,
    derivative: int,
) -> tuple[OffsetType, ...] | OffsetType:
    # single value

    is_leaf = jtu.treedef_is_leaf(treedef)
    length = treedef.num_leaves

    if isinstance(offsets, Offset):
        if offsets.accuracy < 2:
            raise ValueError(f"offset accuracy must be >=2, got {offsets.accuracy}")
        offsets = jnp.array(_generate_central_offsets(derivative, offsets.accuracy))
        return (offsets,) * length

    if isinstance(offsets, jax.Array):
        return (offsets,) * length

    if isinstance(offsets, Sequence) and not is_leaf:
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

    offsets_leaves, offsets_treedef = jtu.tree_flatten(offsets)

    if offsets_treedef == treedef:
        # offsets is a pytree with the same structure as the input
        return offsets_leaves

    raise TypeError(
        f"`offsets` must be of type:\n"
        f"- `Offset`\n"
        f"- `jax.Array`\n"
        f"- tuple of `Offset` or `jax.Array` for tuple argnums.\n"
        f"- pytree with the same structure as the desired arg.\n"
        f"but got {type(offsets)=}"
    )


def _fgrad_along_argnum(
    func: Callable,
    *,
    argnum: int = 0,
    step_size: StepsizeType | None = None,
    offsets: OffsetType = Offset(accuracy=3),
    derivative: int = 1,
):
    if not isinstance(argnum, int):
        raise TypeError(f"argnum must be an integer, got {type(argnum)}")

    def wrapper(*args, **kwargs):
        arg_leaves, arg_treedef = jtu.tree_flatten(args[argnum])
        args_ = list(args)

        def func_wrapper(*leaves):
            args_[argnum] = jtu.tree_unflatten(arg_treedef, leaves)
            return func(*args_, **kwargs)

        spec = dict(treedef=arg_treedef, derivative=derivative)
        step_size_ = resolve_step_size(step_size, **spec)
        offsets_ = resolve_offsets(offsets, **spec)

        flat_result = [
            _evaluate_func_at_shifted_steps_along_argnum(
                func=func_wrapper,
                coeffs=generate_finitediff_coeffs(oi, derivative),
                offsets=oi,
                step_size=si,
                derivative=derivative,
                argnum=i,
            )(*arg_leaves)
            for i, (oi, si) in enumerate(zip(offsets_, step_size_))
        ]

        return jtu.tree_unflatten(arg_treedef, flat_result)

    return wrapper


def value_and_fgrad(
    func: Callable,
    *,
    argnums: int | tuple[int, ...] = 0,
    step_size: StepsizeType | None = None,
    offsets: OffsetType = Offset(accuracy=3),
    derivative: int = 1,
    has_aux: bool = False,
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
        has_aux: whether the function returns an auxiliary output. Defaults to False.
            If True, the derivative function will return a tuple of the form:
            ((value,aux), derivative) otherwise (value, derivative)

    Returns:
        Value and derivative of the function if `has_aux` is False.
        If `has_aux` is True, the derivative function will return a tuple of the form:
        ((value,aux), derivative)

    Example:
        >>> import finitediffx as fdx
        >>> import jax
        >>> import jax.numpy as jnp
        >>> def f(x, y):
        ...    return x**2 + y**2
        >>> df=fdx.value_and_fgrad(
        ...    func=f,
        ...    argnums=1,  # differentiate with respect to y
        ...    offsets=fdx.Offset(accuracy=2)  # use 2nd order accurate finite difference
        ... )
        >>> df(2.0, 3.0)
        (13.0, Array(6., dtype=float32))

    """
    func.__doc__ = (
        f"Finite difference derivative of {getattr(func,'__name__', func)}"
        f" w.r.t {argnums=}\n\n{func.__doc__}"
    )
    if not isinstance(has_aux, bool):
        raise TypeError(f"{type(has_aux)} not a bool")

    func_ = (lambda *a, **k: func(*a, **k)[0]) if has_aux else func

    if isinstance(argnums, int):
        # fgrad(func, argnums=0)
        spec = dict(treedef=constant_treedef, derivative=derivative)

        dfunc = _fgrad_along_argnum(
            func=func_,
            argnum=argnums,
            step_size=step_size,
            offsets=offsets,
            derivative=derivative,
        )

        if has_aux is True:

            @ft.wraps(func)
            def wrapper(*a, **k):
                value, aux = func(*a, **k)
                return (value, aux), dfunc(*a, **k)

            return wrapper

        @ft.wraps(func)
        def wrapper(*a, **k):
            return func(*a, **k), dfunc(*a, **k)

        return wrapper

    if isinstance(argnums, tuple):
        # fgrad(func, argnums=(0,1))
        # return a tuple of derivatives evaluated at each argnum
        # this behavior is similar to jax.grad(func, argnums=(...))
        treedef = jtu.tree_structure(argnums)
        spec = dict(treedef=treedef, derivative=derivative)

        dfuncs = [
            _fgrad_along_argnum(
                func=func_,
                argnum=argnum_i,
                step_size=step_size_i,
                offsets=offsets_i,
                derivative=derivative,
            )
            for offsets_i, step_size_i, argnum_i in zip(
                resolve_offsets(offsets, **spec),
                resolve_step_size(step_size, **spec),
                argnums,
            )
        ]

        if has_aux:

            @ft.wraps(func)
            def wrapper(*a, **k):
                # destructuring the tuple to ensure
                # two item tuple is returned
                value, aux = func(*a, **k)
                return (value, aux), tuple(df(*a, **k) for df in dfuncs)

            return wrapper

        @ft.wraps(func)
        def wrapper(*a, **k):
            return func(*a, **k), tuple(df(*a, **k) for df in dfuncs)

        return wrapper

    raise TypeError(f"argnums must be an int or a tuple of ints, got {argnums}")


def fgrad(
    func: Callable,
    *,
    argnums: int | tuple[int, ...] = 0,
    step_size: StepsizeType | None = None,
    offsets: OffsetType = Offset(accuracy=3),
    derivative: int = 1,
    has_aux: bool = False,
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
        has_aux: whether the function returns an auxiliary output. Defaults to False.
            If True, the derivative function will return a tuple of the form:
            (derivative, aux) otherwise it will return only the derivative.

    Returns:
        Derivative of the function if `has_aux` is False, otherwise a tuple of
        the form: (derivative, aux)

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
    value_and_fgrad_func = value_and_fgrad(
        func=func,
        argnums=argnums,
        step_size=step_size,
        offsets=offsets,
        derivative=derivative,
        has_aux=has_aux,
    )

    if has_aux:

        @ft.wraps(func)
        def wrapper(*a, **k):
            (_, aux), g = value_and_fgrad_func(*a, **k)
            return g, aux

        return wrapper

    @ft.wraps(func)
    def wrapper(*a, **k):
        _, g = value_and_fgrad_func(*a, **k)
        return g

    return wrapper


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
        kwargs = dict(treedef=jtu.tree_structure(primals), derivative=1)
        step_size_ = resolve_step_size(step_size, **kwargs)
        offsets_ = resolve_offsets(offsets, **kwargs)
        primal_out = func(*primals)
        tangent_out = sum(
            fgrad(func, argnums=i, step_size=si, offsets=oi)(*primals) * ti
            for i, (si, oi, ti) in enumerate(zip(step_size_, offsets_, tangents))
        )
        return jnp.array(primal_out), jnp.array(tangent_out)

    return func
