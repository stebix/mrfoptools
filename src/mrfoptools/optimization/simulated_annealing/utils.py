from typing import TypeAlias, Hashable, Any
from collections.abc import Mapping

import numpy as np
import jax

from mrfoptools.optimization.simulated_annealing.variables import Designation

InnerKey: TypeAlias = Hashable
OuterKey: TypeAlias = Hashable

def transpose(
    d: Mapping[OuterKey, Mapping[InnerKey, Any]], /
) -> dict[InnerKey, dict[OuterKey, Any]]:
    """
    Transpose a singularly nested mapping by swapping the outer and inner keys.
    """
    transposed_mapping: dict[InnerKey, dict[OuterKey, Any]] = {}
    for outer_key, inner_mapping in d.items():
        for inner_key, value in inner_mapping.items():
            if inner_key not in transposed_mapping:
                transposed_mapping[inner_key] = {}
            transposed_mapping[inner_key][outer_key] = value
    return transposed_mapping


def recursive_cast(
    mapping: Mapping,
    key_cast_types: tuple[type, ...] = (Designation,),
    value_cast_types: tuple[type, ...] = (np.ndarray, jax.Array),
) -> dict:
    """
    Recursively cast all values of `cast_types` in the mapping.
    The values of type `value_cast_types` must support the `tolist()` method.
    The keys of type `key_cast_types` are cast to their `to_string()` representation.
    """
    recast = {}
    for k, v in mapping.items():
        if isinstance(k, key_cast_types):
            k = k.to_string()
        if isinstance(v, Mapping):
            recast[k] = recursive_cast(v, value_cast_types)
        elif isinstance(v, value_cast_types):
            recast[k] = v.tolist()
        else:
            recast[k] = v
    return recast
