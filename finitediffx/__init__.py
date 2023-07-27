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

from ._src.fgrad import Offset, define_fdjvp, fgrad, value_and_fgrad
from ._src.finite_diff import (
    curl,
    difference,
    divergence,
    gradient,
    hessian,
    jacobian,
    laplacian,
)
from ._src.utils import generate_finitediff_coeffs

__all__ = (
    "curl",
    "divergence",
    "difference",
    "gradient",
    "jacobian",
    "laplacian",
    "hessian",
    "fgrad",
    "value_and_fgrad",
    "Offset",
    "define_fdjvp",
    "generate_finitediff_coeffs",
)


__version__ = "0.1.0"
