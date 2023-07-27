🛠️ Installation
----------------

Install from github::

   pip install finitediffx


🏃 Quick example
------------------

.. code-block:: python

    import jax.numpy as jnp
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

    # ∇.F : the divergence of F
    divF = fdx.divergence(
        F,
        step_size=(dx, dy, dz),
        keepdims=False,
        accuracy=6,
        method="central",
    )



.. toctree::
    :caption: Examples
    :maxdepth: 1
    
    examples


.. toctree::
    :caption: API Documentation
    :maxdepth: 1

    
    API/api

Apache2.0 License.

Indices
=======

* :ref:`genindex`


