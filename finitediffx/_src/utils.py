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
from typing import Any

import jax
import jax.numpy as jnp


def _check_and_return(value: Any, ndim: int, name: str):
    if isinstance(value, int):
        return (value,) * ndim
    elif isinstance(value, jax.Array):
        return jnp.repeat(value, ndim)
    elif isinstance(value, tuple):
        assert len(value) == ndim, f"{name} must be a tuple of length {ndim}"
        return tuple(value)
    raise ValueError(f"Expected int or tuple for {name}, got {value}.")


def _generate_forward_offsets(
    derivative: int,
    accuracy: int = 2,
) -> tuple[float | int, ...]:
    """Generate central difference offsets

    Args:
        derivative : derivative order

    Returns:
        tuple[float | int, ...]: central difference offsets
    """
    if derivative < 1:
        msg = f"derivative must be >= 1 for forward offset generation, got {derivative}"
        raise ValueError(msg)
    if accuracy < 1:
        msg = f"accuracy must be >= 2 for forward offset generation, got {accuracy}"
        raise ValueError(msg)

    # ref:https://en.wikipedia.org/wiki/Finite_difference_coefficient
    return tuple(range(0, (derivative + accuracy)))


def _generate_central_offsets(
    derivative: int,
    accuracy: int = 2,
) -> tuple[float | int, ...]:
    """Generate central difference offsets

    Args:
        derivative : derivative order

    Returns:
        tuple[float | int, ...]: central difference offsets
    """
    if derivative < 1:
        msg = f"derivative must be >= 1 for central offset generation, got {derivative}"
        raise ValueError(msg)
    if accuracy < 2:
        msg = f"accuracy must be >= 2 for central offset generation, got {accuracy}"
        raise ValueError(msg)

    # ref:https://en.wikipedia.org/wiki/Finite_difference_coefficient
    left = -((derivative + accuracy - 1) // 2)
    right = (derivative + accuracy - 1) // 2 + 1
    return tuple(range(left, right))


def _generate_backward_offsets(
    derivative: int,
    accuracy: int = 2,
) -> tuple[float | int, ...]:
    """Generate central difference offsets

    Args:
        derivative : derivative order

    Returns:
        tuple[float | int, ...]: central difference offsets
    """
    if derivative < 1:
        msg = f"derivative must be >= 1 for back offset generation, got {derivative}"
        raise ValueError(msg)
    if accuracy < 1:
        msg = f"accuracy must be >= 2 for back offset generation, got {accuracy}"
        raise ValueError(msg)

    return tuple(range(-(derivative + accuracy - 1), 1))


@ft.partial(jax.jit, static_argnums=(1,))
def generate_finitediff_coeffs(
    offsets: tuple[float | int, ...],
    derivative: int,
) -> jax.Array:
    """Generate FD coeffs

    Args:
        offsets: offsets of the finite difference stencil
        derivative: derivative order

    Returns:
        tuple[float]: finite difference coefficients

    Example:
        >>> generate_finitediff_coeffs(offsets=(-1, 0, 1), derivative=1)
        Array([-0.5,  0. ,  0.5], dtype=float32)

        >>> generate_finitediff_coeffs(offsets=(-1, 0, 1), derivative=2)
        Array([ 1., -2.,  1.], dtype=float32)
        >>> # translates to  1*f(x-1) - 2*f(x) + 1*f(x+1)

    See:
        https://en.wikipedia.org/wiki/Finite_difference_coefficient
        https://web.media.mit.edu/~crtaylor/calculator.html
    """

    if derivative >= (N := len(offsets)):
        raise ValueError(f"{len(offsets)=} must be larger than {derivative=}.")

    A = jnp.repeat(jnp.array(offsets)[None, :], repeats=N, axis=0)
    A **= jnp.arange(0, N).reshape(-1, 1)
    index = jnp.arange(N)
    factorial = jnp.prod(jnp.arange(1, derivative + 1))
    B = jnp.where(index == derivative, factorial, 0)[:, None]
    return (jnp.linalg.inv(A) @ B).flatten()
