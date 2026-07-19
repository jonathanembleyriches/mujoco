"""ProtoSpec generator toolchain (build-time Python).

Public surface: the IDL parser. ``parse_spec`` turns ``.spec`` schema text (or a
path) into a validated :class:`Schema` whose ``to_json()`` is the canonical AST
that all code emitters consume.
"""

from .idl import Schema, SchemaError, parse_spec

__all__ = ["parse_spec", "Schema", "SchemaError"]
