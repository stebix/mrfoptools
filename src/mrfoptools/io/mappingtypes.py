from collections.abc import Mapping, Sequence
from typing import NamedTuple, Union, TypeAlias
from numbers import Number

import jax
import numpy as np


Array = jax.Array | np.ndarray
ArrayLike = Array | Sequence[Number]

# DataMapping intended to reference nested dictionaries of numpy arrays
# or sequences of numbers that can be written to zarr arrays directly.
DataMapping: TypeAlias = Mapping[str, Union['DataMapping', ArrayLike]]

# Jsonifiable intended to reference data types that can be serialized to JSON.
# This is utilized for metadata mappings that are stored as attributes in zarr groups.
Jsonifiable = str | int | float | bool | None | Mapping[str, 'Jsonifiable'] | Sequence['Jsonifiable']
MetadataMapping: TypeAlias = Mapping[str, Union['MetadataMapping', Jsonifiable]]


class GroupKeys(NamedTuple):
    array_keys: set[str]
    subgroup_keys: set[str]


ArrayMapping = Mapping[str, Union[np.ndarray, 'ArrayMapping']]  



class MappingKeys(NamedTuple):
    array_keys: set[str]
    submapping_keys: set[str]
    metadata_keys: set[str]


def sort_mapping_keys(
    mapping: Mapping,
    *,
    array_like_types: tuple[type, ...] = (np.ndarray, jax.Array)
) -> MappingKeys:
    """
    Sort the keys of a mapping into three categories based on the value type.
    Mappings (nested elements), array-like objects and metadata-like objects (all else).

    Parameters
    ----------
    mapping : Mapping
        The mapping to sort keys from.
    
    array_like_types : tuple[type, ...], optional
        Types to consider array-like, by default (np.ndarray, jax.Array)

    Returns
    -------
    MappingKeys
        A named tuple with the keys sorted into array keys, submapping keys and metadata keys.
    """
    array_keys = set()
    submapping_keys = set()
    metadata_keys = set()

    for key, value in mapping.items():
        if isinstance(value, Mapping):
            submapping_keys.add(key)
        elif isinstance(value, array_like_types):
            array_keys.add(key)
        else:
            metadata_keys.add(key)

    return MappingKeys(
        array_keys=array_keys,
        submapping_keys=submapping_keys,
        metadata_keys=metadata_keys
    )


def datamappings_are_equal(
    mapping_a: DataMapping,
    mapping_b: DataMapping
) -> bool:
    """
    Compare two data mappings for equality.
    """
    arry_like_types = (np.ndarray, jax.Array)
    if mapping_a.keys() != mapping_b.keys():
        return False

    for key in mapping_a.keys():
        value_a = mapping_a[key]
        value_b = mapping_b[key]

        if isinstance(value_a, Mapping):
            if not datamappings_are_equal(value_a, value_b):
                return False
        elif isinstance(value_a, arry_like_types):
            if not np.allclose(value_a, value_b):
                return False
        else:
            return value_a == value_b

    return True
