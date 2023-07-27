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

import dataclasses as dc
import functools as ft
from typing import Any, Callable, Sequence, TypeVar, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from typing_extensions import ParamSpec

from finitediffx._src.utils import _generate_central_offsets, generate_finitediff_coeffs

__all__ = ("fgrad", "Offset", "define_fdjvp", "value_and_fgrad")

P = ParamSpec("P")
T = TypeVar("T")
PyTree = Any


@dc.dataclass(frozen=True)
class Offset:
    """Convinience class for finite difference offsets used inside :func:`.fgrad`
    :func:`.value_and_fgrad`.

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


def resolve_step_size(
    step_size: StepsizeType | Sequence[StepsizeType] | None,
    treedef: jtu.PyTreeDef,
    derivative: int,
) -> Sequence[StepsizeType] | StepsizeType:
    # return non-tuple values if length is None
    length = treedef.num_leaves

    if isinstance(step_size, (jax.Array, float)):
        return (step_size,) * length

    if step_size is None:
        default = (2) ** (-23 / (2 * derivative))
        return (default,) * length

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
    length = treedef.num_leaves

    if isinstance(offsets, Offset):
        if offsets.accuracy < 2:
            raise ValueError(f"offset accuracy must be >=2, got {offsets.accuracy}")
        offsets = jnp.array(_generate_central_offsets(derivative, offsets.accuracy))
        return (offsets,) * length

    if isinstance(offsets, jax.Array):
        return (offsets,) * length

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


def _perturb_flat_args(
    *,
    flat_func: Callable,
    coeffs: jax.Array,
    flat_offsets: jax.Array,
    flat_argnum: int,
    flat_step_size: jax.Array,
    derivative: int,
    average_gradients: bool = False,
):
    def flat_args_wrapper(*flat_args):
        def scalar_perturb(*, h: float):
            return flat_func(
                *(
                    flat_args[:flat_argnum]
                    + (flat_args[flat_argnum] + h,)
                    + flat_args[flat_argnum + 1 :]
                )
            )

        def array_perturb(*, h: float) -> jax.Array:
            # should be much slower than jax.grad for large arrays
            # but can be used for non-tracable code where jax.grad fails
            size = flat_args[flat_argnum].size
            indices = jnp.arange(size)
            shape = flat_args[flat_argnum].shape
            flat_array = jnp.array(flat_args[flat_argnum].reshape(-1))

            def perturb_element(index):
                return flat_func(
                    *(
                        flat_args[:flat_argnum]
                        + (flat_array.at[index].add(h).reshape(shape),)
                        + flat_args[flat_argnum + 1 :]
                    )
                )

            try:
                # in case of tracable code (jax code)
                result = jax.vmap(perturb_element)(indices)
            except jax.errors.TracerArrayConversionError:
                # non-tracable code e.g. numpy code
                result = jnp.array([perturb_element(index) for index in indices])

            if result.size > size:
                raise TypeError("Non scalar-output.")

            return result.reshape(shape)

        def array_average_perturb(*, h: float) -> jax.Array:
            # perturb the array all at once and average the result
            # faster than array_perturb for large arrays but gives
            # average gradient
            shape = flat_args[flat_argnum].shape
            size = flat_args[flat_argnum].size
            result = flat_func(
                *(
                    flat_args[:flat_argnum]
                    + ((flat_args[flat_argnum] + h),)
                    + flat_args[flat_argnum + 1 :]
                )
            )

            if result.size > size:
                raise TypeError("Non scalar-output.")

            result = jnp.broadcast_to(result, shape)
            result = result / result.size
            return result

        perturb_func = (
            (array_average_perturb if average_gradients else array_perturb)
            if isinstance(flat_args[flat_argnum], (np.ndarray, jax.Array))
            else scalar_perturb
        )

        return sum(
            coeff * perturb_func(h=dx) / flat_step_size**derivative
            for coeff, dx in zip(coeffs, flat_offsets * flat_step_size)
        )

    return flat_args_wrapper


def _fgrad_along_argnum(
    func: Callable,
    *,
    argnum: int = 0,
    step_size: StepsizeType | None = None,
    offsets: OffsetType = Offset(accuracy=3),
    derivative: int = 1,
    average_gradients: bool = False,
):
    def wrapper(*args, **kwargs):
        # full args/kwargs
        flat_args, flat_treedef = jtu.tree_flatten(args[argnum])

        def flat_func(*flat_args):
            return func(
                *(
                    args[:argnum]
                    + (jtu.tree_unflatten(flat_treedef, flat_args),)
                    + args[argnum + 1 :]
                ),
                **kwargs,
            )

        step_size_ = resolve_step_size(step_size, flat_treedef, derivative)
        offsets_ = resolve_offsets(offsets, flat_treedef, derivative)

        flat_result = (
            _perturb_flat_args(
                flat_func=flat_func,
                coeffs=generate_finitediff_coeffs(oi, derivative),
                flat_offsets=oi,
                flat_step_size=si,
                flat_argnum=i,
                derivative=derivative,
                average_gradients=average_gradients,
            )(*flat_args)
            for i, (oi, si) in enumerate(zip(offsets_, step_size_))
        )

        return jtu.tree_unflatten(flat_treedef, flat_result)

    return wrapper


def value_and_fgrad(
    func: Callable[P, T],
    *,
    argnums: int | tuple[int, ...] = 0,
    step_size: StepsizeType | None = None,
    offsets: OffsetType = Offset(accuracy=3),
    derivative: int = 1,
    has_aux: bool = False,
    average_gradients: bool = False,
):
    """Finite difference derivative of a function with respect to one of its arguments.

    Similar to ``jax.value_and_grad`` but with finite difference approximation

    Args:
        func: function to differentiate
        argnums: argument number to differentiate. Defaults to 0.
            If a tuple is passed, the function is differentiated with respect to
            all the arguments in the tuple.
        step_size: step size for the finite difference stencil. If `None`, the step size
            is set to ``(2) ** (-23 / (2 * derivative))``
        offsets: offsets for the finite difference stencil. Accepted types are:

            - ``jax.Array`` defining location of function evaluation points.
            - :class:`Offset` with accuracy field to automatically generate offsets.
            - pytree of ``jax.Array``/ :class:`.Offset` to define offsets for
              each argument of the same pytree structure as argument defined
              by ``argnums``.

        derivative: derivative order. Defaults to 1.
        has_aux: whether the function returns an auxiliary output. Defaults to
            ``False``. If True, the derivative function will return a tuple of
            the form: ((value,aux), derivative) otherwise (value, derivative)
        average_gradients: whether to average the array gradients. Yields faster
            results. Defaults to ``False``.

    Returns:
        Value and derivative of the function if ``has_aux`` is ``False``.
        If ``has_aux`` is True, the derivative function will return a tuple of the form:
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
        dfunc = _fgrad_along_argnum(
            func=func_,
            argnum=argnums,
            step_size=step_size,
            offsets=offsets,
            derivative=derivative,
            average_gradients=average_gradients,
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
        if not all(isinstance(ai, int) for ai in argnums):
            raise TypeError(f"{argnums=} must be an integer or a tuple of integers")

        if isinstance(offsets, tuple):
            if len(offsets) != len(argnums):
                raise AssertionError("offsets must have the same length as argnums")
        else:
            offsets = (offsets,) * len(argnums)

        if isinstance(step_size, tuple):
            if len(step_size) != len(argnums):
                raise AssertionError("step_size must have the same length as argnums")

        else:
            step_size = (step_size,) * len(argnums)

        dfuncs = [
            _fgrad_along_argnum(
                func=func_,
                argnum=ai,
                step_size=si,
                offsets=oi,
                derivative=derivative,
                average_gradients=average_gradients,
            )
            for oi, si, ai in zip(offsets, step_size, argnums)
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
    average_gradients: bool = False,
) -> Callable:
    """Finite difference derivative of a function with respect to one of its arguments.

    Similar to ``jax.grad`` but with finite difference approximation.

    Args:
        func: function to differentiate
        argnums: argument number to differentiate. Defaults to 0.
            If a tuple is passed, the function is differentiated with respect to
            all the arguments in the tuple.
        step_size: step size for the finite difference stencil. If `None`, the step size
            is set to `(2) ** (-23 / (2 * derivative))`
        offsets: offsets for the finite difference stencil. Accepted types are:

            - ``jax.Array`` defining location of function evaluation points.
            - :class:`.Offset` with accuracy field to automatically generate offsets.
            - pytree of ``jax.Array``/:class:`.Offset` to define offsets for
              each argument of the same pytree structure as argument defined
              by ``argnums``.

        derivative: derivative order. Defaults to 1.
        has_aux: whether the function returns an auxiliary output. Defaults to
            ``False``. If ``True``, the derivative function will return a tuple
            of the form: (derivative, aux) otherwise it will return only the derivative.
        average_gradients: whether to average the array gradients. Yields faster
            results. Defaults to ``False``.

    Returns:
        Derivative of the function if ``has_aux`` is ``False``, otherwise a tuple of
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
        average_gradients=average_gradients,
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
            - ``jax.Array`` defining location of function evaluation points.
            - :class:`.Offset` with accuracy field to automatically generate offsets.
            - tuple of `Offset` or ``jax.Array`` defining offsets for each argument.
        step_size: step size for the finite difference stencil. Accepted types are:
            - ``float`` defining the step size for all arguments.
            - ``tuple`` of ``float`` defining the step size for each argument.
            - ``None`` to use the default step size for each argument.

    Returns:
        Callable: function with JVP rule defined using finite difference.

    Note:
        - This function is motivated by [``JEP``](https://github.com/google/jax/issues/15425)
        - This function can be used with ``jax.pure_callback`` to define the JVP
            rule for a function that is not differentiable by ``jax``.

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
