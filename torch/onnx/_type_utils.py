"""Utilities for converting and operating on ONNX, JIT and torch types."""
from __future__ import annotations

import enum
from typing import Dict, Optional, Union

from typing_extensions import Literal

import torch
from torch._C import _onnx as _C_onnx
from torch.onnx import errors
from torch.onnx._internal import _beartype

ScalarName = Literal[
    "Byte",
    "Char",
    "Double",
    "Float",
    "Half",
    "Int",
    "Long",
    "Short",
    "Bool",
    "ComplexHalf",
    "ComplexFloat",
    "ComplexDouble",
    "QInt8",
    "QUInt8",
    "QInt32",
    "BFloat16",
    "Undefined",
]

TorchName = Literal[
    "bool",
    "uint8_t",
    "int8_t",
    "double",
    "float",
    "half",
    "int",
    "int64_t",
    "int16_t",
    "complex32",
    "complex64",
    "complex128",
    "qint8",
    "quint8",
    "qint32",
    "bfloat16",
]


class JitScalarType(enum.IntEnum):
    """Scalar types defined in torch.

    Use ``JitScalarType`` to convert from torch and JIT scalar types to ONNX scalar types.

    Examples::
        >>> # xdoctest: +IGNORE_WANT("win32 has different output")
        >>> JitScalarType.from_name("Float").onnx_type()
        TensorProtoDataType.FLOAT
    """

    # Order defined in https://github.com/pytorch/pytorch/blob/344defc9733a45fee8d0c4d3f5530f631e823196/c10/core/ScalarType.h
    UINT8 = 0
    INT8 = enum.auto()  # 1
    INT16 = enum.auto()  # 2
    INT = enum.auto()  # 3
    INT64 = enum.auto()  # 4
    HALF = enum.auto()  # 5
    FLOAT = enum.auto()  # 6
    DOUBLE = enum.auto()  # 7
    COMPLEX32 = enum.auto()  # 8
    COMPLEX64 = enum.auto()  # 9
    COMPLEX128 = enum.auto()  # 10
    BOOL = enum.auto()  # 11
    QINT8 = enum.auto()  # 12
    QUINT8 = enum.auto()  # 13
    QINT32 = enum.auto()  # 14
    BFLOAT16 = enum.auto()  # 15
    UNDEFINED = enum.auto()  # 16

    @classmethod
    @_beartype.beartype
    def from_name(
        cls, name: Union[ScalarName, TorchName, Optional[str]]
    ) -> JitScalarType:
        """Convert a JIT scalar type or torch type name to ScalarType.

        NB: DO NOT USE this API if dtype is comes from a torch._C.Value.type().scalarType() call because
            "RuntimeError: INTERNAL ASSERT FAILED at "../aten/src/ATen/core/jit_type_base.h" can
            be raised in some scenarios. Instead use from_value API which handles all cases

        Args:
            name: JIT scalar type name (Byte) or torch type name (uint8_t).

        Returns:
            JitScalarType

        Raises:
            ValueError: if name is not a valid scalar type name or if it is None.
        """
        if name is None:
            raise ValueError("Scalar type name cannot be None")
        if valid_scalar_name(name):
            return _SCALAR_NAME_TO_TYPE[name]  # type: ignore[index]
        if valid_torch_name(name):
            return _TORCH_NAME_TO_SCALAR_TYPE[name]  # type: ignore[index]

        raise ValueError(f"Unknown torch or scalar type: '{name}'")

    @classmethod
    @_beartype.beartype
    def from_dtype(cls, dtype: torch.dtype) -> JitScalarType:
        """Convert a torch dtype to JitScalarType.

        NB: DO NOT USE this API if dtype is comes from a torch._C.Value.type().dtype() call because
            "RuntimeError: INTERNAL ASSERT FAILED at "../aten/src/ATen/core/jit_type_base.h" can
            be raised in some scenarios. Instead use from_value API which handles all cases

        Args:
            dtype: A torch.dtype to create a JitScalarType from

        Returns:
            JitScalarType

        Raises:
            ValueError: if dtype is not a valid torch.dtype or if it is None.
        """
        if dtype not in _DTYPE_TO_SCALAR_TYPE:
            raise ValueError(f"Unknown dtype: {dtype}")
        return _DTYPE_TO_SCALAR_TYPE[dtype]

    # TODO: `value: torch._C.Value` type annotation raises (CI machines only):
    #       NameError: name '_C' is not defined
    # In both cases, using typing.TypeVar or string literals didn't work
    @classmethod
    @_beartype.beartype
    def from_value(cls, value) -> JitScalarType:
        """Create a JitScalarType from a torch._C.Value underlying scalar type.

        Args:
            value: A `torch._C.Value` object to fetch scalar type from.

        Returns:
            JitScalarType.

        Raises:
            OnnxExporterError: if value is None.
            SymbolicValueError: when value.type().scalarType() does not exist

        """

        if value is None:
            raise errors.OnnxExporterError(
                "Cannot determine scalar type for `None` instance."
            )
        elif isinstance(value, torch.Tensor):
            return JitScalarType.from_dtype(value.dtype)
        elif isinstance(value.type(), torch.ListType):
            return JitScalarType.from_dtype(value.type().getElementType().dtype())
        scalar_type = value.type().scalarType()
        if scalar_type is None:
            raise errors.SymbolicValueError(
                f"Cannot determine scalar type for this '{type(value.type())}' instance.",
                value,
            )
        return JitScalarType.from_name(scalar_type)

    @_beartype.beartype
    def scalar_name(self) -> ScalarName:
        """Convert a JitScalarType to a JIT scalar type name."""
        return _SCALAR_TYPE_TO_NAME[self]

    @_beartype.beartype
    def torch_name(self) -> TorchName:
        """Convert a JitScalarType to a torch type name."""
        return _SCALAR_TYPE_TO_TORCH_NAME[self]

    @_beartype.beartype
    def dtype(self) -> torch.dtype:
        """Convert a JitScalarType to a torch dtype."""
        return _SCALAR_TYPE_TO_DTYPE[self]

    @_beartype.beartype
    def onnx_type(self) -> _C_onnx.TensorProtoDataType:
        """Convert a JitScalarType to an ONNX data type."""
        if self not in _SCALAR_TYPE_TO_ONNX:
            raise ValueError(f"Scalar type {self} cannot be converted to ONNX")
        return _SCALAR_TYPE_TO_ONNX[self]

    @_beartype.beartype
    def onnx_compatible(self) -> bool:
        """Return whether this JitScalarType is compatible with ONNX."""
        return (
            self in _SCALAR_TYPE_TO_ONNX
            and self != JitScalarType.UNDEFINED
            and self != JitScalarType.COMPLEX32
        )


@_beartype.beartype
def valid_scalar_name(scalar_name: Union[ScalarName, str]) -> bool:
    """Return whether the given scalar name is a valid JIT scalar type name."""
    return scalar_name in _SCALAR_NAME_TO_TYPE


@_beartype.beartype
def valid_torch_name(torch_name: Union[TorchName, str]) -> bool:
    """Return whether the given torch name is a valid torch type name."""
    return torch_name in _TORCH_NAME_TO_SCALAR_TYPE


# https://github.com/pytorch/pytorch/blob/344defc9733a45fee8d0c4d3f5530f631e823196/c10/core/ScalarType.h
_SCALAR_TYPE_TO_NAME: Dict[JitScalarType, ScalarName] = {
    JitScalarType.BOOL: "Bool",
    JitScalarType.UINT8: "Byte",
    JitScalarType.INT8: "Char",
    JitScalarType.INT16: "Short",
    JitScalarType.INT: "Int",
    JitScalarType.INT64: "Long",
    JitScalarType.HALF: "Half",
    JitScalarType.FLOAT: "Float",
    JitScalarType.DOUBLE: "Double",
    JitScalarType.COMPLEX32: "ComplexHalf",
    JitScalarType.COMPLEX64: "ComplexFloat",
    JitScalarType.COMPLEX128: "ComplexDouble",
    JitScalarType.QINT8: "QInt8",
    JitScalarType.QUINT8: "QUInt8",
    JitScalarType.QINT32: "QInt32",
    JitScalarType.BFLOAT16: "BFloat16",
    JitScalarType.UNDEFINED: "Undefined",
}

_SCALAR_NAME_TO_TYPE: Dict[ScalarName, JitScalarType] = {
    v: k for k, v in _SCALAR_TYPE_TO_NAME.items()
}

_SCALAR_TYPE_TO_TORCH_NAME: Dict[JitScalarType, TorchName] = {
    JitScalarType.BOOL: "bool",
    JitScalarType.UINT8: "uint8_t",
    JitScalarType.INT8: "int8_t",
    JitScalarType.INT16: "int16_t",
    JitScalarType.INT: "int",
    JitScalarType.INT64: "int64_t",
    JitScalarType.HALF: "half",
    JitScalarType.FLOAT: "float",
    JitScalarType.DOUBLE: "double",
    JitScalarType.COMPLEX32: "complex32",
    JitScalarType.COMPLEX64: "complex64",
    JitScalarType.COMPLEX128: "complex128",
    JitScalarType.QINT8: "qint8",
    JitScalarType.QUINT8: "quint8",
    JitScalarType.QINT32: "qint32",
    JitScalarType.BFLOAT16: "bfloat16",
}

_TORCH_NAME_TO_SCALAR_TYPE: Dict[TorchName, JitScalarType] = {
    v: k for k, v in _SCALAR_TYPE_TO_TORCH_NAME.items()
}

_SCALAR_TYPE_TO_ONNX = {
    JitScalarType.BOOL: _C_onnx.TensorProtoDataType.BOOL,
    JitScalarType.UINT8: _C_onnx.TensorProtoDataType.UINT8,
    JitScalarType.INT8: _C_onnx.TensorProtoDataType.INT8,
    JitScalarType.INT16: _C_onnx.TensorProtoDataType.INT16,
    JitScalarType.INT: _C_onnx.TensorProtoDataType.INT32,
    JitScalarType.INT64: _C_onnx.TensorProtoDataType.INT64,
    JitScalarType.HALF: _C_onnx.TensorProtoDataType.FLOAT16,
    JitScalarType.FLOAT: _C_onnx.TensorProtoDataType.FLOAT,
    JitScalarType.DOUBLE: _C_onnx.TensorProtoDataType.DOUBLE,
    JitScalarType.COMPLEX64: _C_onnx.TensorProtoDataType.COMPLEX64,
    JitScalarType.COMPLEX128: _C_onnx.TensorProtoDataType.COMPLEX128,
    JitScalarType.BFLOAT16: _C_onnx.TensorProtoDataType.BFLOAT16,
    JitScalarType.UNDEFINED: _C_onnx.TensorProtoDataType.UNDEFINED,
    JitScalarType.COMPLEX32: _C_onnx.TensorProtoDataType.UNDEFINED,
    JitScalarType.QINT8: _C_onnx.TensorProtoDataType.INT8,
    JitScalarType.QUINT8: _C_onnx.TensorProtoDataType.UINT8,
    JitScalarType.QINT32: _C_onnx.TensorProtoDataType.INT32,
}

# source of truth is
# https://github.com/pytorch/pytorch/blob/master/torch/csrc/utils/tensor_dtypes.cpp
_SCALAR_TYPE_TO_DTYPE = {
    JitScalarType.BOOL: torch.bool,
    JitScalarType.UINT8: torch.uint8,
    JitScalarType.INT8: torch.int8,
    JitScalarType.INT16: torch.short,
    JitScalarType.INT: torch.int,
    JitScalarType.INT64: torch.int64,
    JitScalarType.HALF: torch.half,
    JitScalarType.FLOAT: torch.float,
    JitScalarType.DOUBLE: torch.double,
    JitScalarType.COMPLEX32: torch.complex32,
    JitScalarType.COMPLEX64: torch.complex64,
    JitScalarType.COMPLEX128: torch.complex128,
    JitScalarType.QINT8: torch.qint8,
    JitScalarType.QUINT8: torch.quint8,
    JitScalarType.QINT32: torch.qint32,
    JitScalarType.BFLOAT16: torch.bfloat16,
}

_DTYPE_TO_SCALAR_TYPE = {v: k for k, v in _SCALAR_TYPE_TO_DTYPE.items()}
