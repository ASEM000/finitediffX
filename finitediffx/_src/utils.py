from __future__ import annotations

import functools as ft

import jax
import jax.numpy as jnp


def _check_and_return(value, ndim, name):
    if isinstance(value, int):
        return (value,) * ndim
    elif isinstance(value, jnp.ndarray):
        return jnp.repeat(value, ndim)
    elif isinstance(value, tuple):
        assert len(value) == ndim, f"{name} must be a tuple of length {ndim}"
        return tuple(value)
    raise ValueError(f"Expected int or tuple for {name}, got {value}.")


def _generate_forward_offsets(
    derivative: int, accuracy: int = 2
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

    return tuple(range(0, (derivative + accuracy)))


def _generate_central_offsets(
    derivative: int, accuracy: int = 2
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

    return tuple(
        range(-((derivative + accuracy - 1) // 2), (derivative + accuracy - 1) // 2 + 1)
    )


def _generate_backward_offsets(
    derivative: int, accuracy: int = 2
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
    offsets: tuple[float | int, ...], derivative: int
) -> tuple[float]:
    """Generate FD coeffs

    Args:
        offsets: offsets of the finite difference stencil
        derivative: derivative order

    Returns:
        tuple[float]: finite difference coefficients

    Example:
        >>> generate_finitediff_coeffs(offsets=(-1, 0, 1), derivative=1)
        (-0.5, 0.0, 0.5)

        >>> generate_finitediff_coeffs(offsets=(-1, 0, 1), derivative=2)
        (1.0, -2.0, 1.0)
        # translates to  1*f(x-1) - 2*f(x) + 1*f(x+1)

    See: https://en.wikipedia.org/wiki/Finite_difference_coefficient
    """

    N = len(offsets)

    if derivative >= N:
        raise ValueError(
            "Sampling points must be larger than derivative order."
            f" len(offsets)={len(offsets)} must be less than {derivative}"
        )

    A = jnp.repeat(jnp.array(offsets)[None, :], repeats=N, axis=0)
    A **= jnp.arange(0, N).reshape(-1, 1)
    index = jnp.arange(N)
    factorial = jnp.prod(jnp.arange(1, derivative + 1))
    B = jnp.where(index == derivative, factorial, 0)[:, None]
    C = jnp.linalg.inv(A) @ B  # solve Ax = B
    return C.flatten()
