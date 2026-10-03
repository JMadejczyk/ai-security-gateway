"""An immutable mapping usable as a Pydantic field.

Frozen Pydantic models only block attribute assignment; a plain ``dict`` field can still be
edited in place. A published policy snapshot must not change without changing its revision,
so every mapping inside it is a `FrozenDict`.
"""

import copy
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, NoReturn, get_args

from pydantic import GetCoreSchemaHandler, SerializerFunctionWrapHandler
from pydantic_core import core_schema


class FrozenDict[K, V](Mapping[K, V]):
    """Read-only mapping: no item or attribute assignment, deep-copyable and picklable."""

    __slots__ = ("_data",)
    _data: Mapping[K, V]

    def __init__(self, data: Mapping[K, V] | None = None) -> None:
        object.__setattr__(self, "_data", MappingProxyType(dict(data or {})))

    def __getitem__(self, key: K) -> V:
        return self._data[key]

    def __iter__(self) -> Iterator[K]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenDict({dict(self._data)!r})"

    def __setattr__(self, name: str, value: object) -> NoReturn:
        msg = f"{type(self).__name__} is immutable"
        raise AttributeError(msg)

    def __delattr__(self, name: str) -> NoReturn:
        msg = f"{type(self).__name__} is immutable"
        raise AttributeError(msg)

    def __deepcopy__(self, memo: dict[int, Any]) -> "FrozenDict[K, V]":
        return FrozenDict(copy.deepcopy(dict(self._data), memo))

    def __reduce__(self) -> tuple[type["FrozenDict[K, V]"], tuple[dict[K, V]]]:
        return (FrozenDict, (dict(self._data),))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: object, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        key_type, value_type = get_args(source_type) or (Any, Any)
        dict_schema = handler.generate_schema(dict[key_type, value_type])
        return core_schema.no_info_after_validator_function(
            cls,
            dict_schema,
            serialization=core_schema.wrap_serializer_function_ser_schema(
                _serialize_as_dict, schema=dict_schema
            ),
        )


def _serialize_as_dict(value: Mapping[Any, Any], handler: SerializerFunctionWrapHandler) -> Any:  # noqa: ANN401 -- returns whatever the dict serializer produces for the mode
    return handler(dict(value))
