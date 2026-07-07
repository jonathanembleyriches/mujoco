"""ProtoSpec IDL parser: `.spec` text -> canonical JSON-serializable schema AST.

This module is the single owner of the ProtoSpec IDL. It contains the tokenizer,
the recursive-descent parser, the AST dataclasses, JSON (de)serialization, and
post-parse validation (type resolution, mixin flattening, presence semantics).
The JSON produced by ``Schema.to_json()`` is the stable contract every downstream
emitter consumes (plan Section 3); its shape is documented below.

Grammar
-------
The normative grammar is plan Section 5. It is reproduced here with the additions
and clarifications this implementation commits to (all underspecified points are
resolved conservatively and noted):

    schema     := header (enumdef | structdef | variantdef | uniondef
                          | mixindef | elemdef)*
    header     := "mujoco_version" STRING
    enumdef    := "enum" NAME "{" (NAME "=" STRING)+ "}"
    structdef  := "struct" NAME "{" field* "}"
    variantdef := "variant" NAME "{" (NAME ":" type)+ "}"          # ADDED (see below)
    uniondef   := "union" NAME "=" NAME ("|" NAME)*                 # ADDED (see below)
    mixindef   := "mixin" NAME "{" field* "}"
    elemdef    := "element" NAME annots? "{" (usestmt | field | child)* "}"
    usestmt    := "use" NAME
    child      := "children" NAME ":" NAME cardinality             # NAME: element | union
    field      := NAME ":" type annots? default? comment?
    type       := prim ("[" INT "]" | "[" INT ".." INT "]" | "[]")?
                | "ref" "<" NAME ">" | "variant" NAME | NAME "[]"? # ref<NAME>: element | union
    annots     := "(" (key ("=" value)?),* ")"
    default    := "=" (literal | NAME | "{" literal,* "}")

Grammar decisions beyond the plan
---------------------------------
* variant definition form (ADDED). The plan references variants like
  ``variant Orientation`` by name in field types but never gives a definition
  form. This module adds a top-level ``variant NAME { tag : type ... }`` block:
  each arm names a tag and a value type (a struct name or a primitive). This is
  the "clean top-level definition" the plan recommends and keeps the field-level
  ``variant NAME`` usage untouched. ``variant`` at the top level is a definition;
  ``variant NAME`` after ``field ":"`` is a type reference -- the two never
  collide because they occur in disjoint syntactic positions.
* union definition form (ADDED). MJCF has sections whose *document order across
  different element tags* is semantic (the ids/data-addresses of actuators,
  sensors, equality constraints and tendons all follow interleaved source
  order, and a spatial tendon's routing IS the interleave of site/geom/pulley
  path items). A per-type child list cannot express that. A ``union NAME = A |
  B | C`` names an ordered heterogeneous set of member *elements*; it is usable
  in exactly two positions and nowhere else:
    - ``children NAME : UnionName cardinality`` -- one ordered child list whose
      items may be any member element, source order preserved; and
    - ``ref<UnionName>`` -- a reference that may target any member element's
      namespace (e.g. ``ref<TendonAny>`` = "any tendon", spatial or fixed).
  A union used as a plain field type is a diagnosed error. Members must resolve
  to elements, must be unique, and a union may not contain another union.
* Element body order is relaxed. The plan writes ``("use" NAME)* field* child*``.
  Members are instead dispatched by leading token (``use`` / ``children`` /
  otherwise a field) and may interleave; flattening still injects mixin fields
  before local fields, so semantics are unchanged. Source order of fields and of
  children is preserved (it is significant for serialization/flattening).
* Sized arity (``[N]`` / ``[N..M]``) attaches to primitives only, exactly as the
  grammar's ``prim "[" ... "]"`` productions state. ``NAME[3]`` on an enum/struct
  is a parse error. The one exception is the *unbounded* form: a bare ``NAME[]``
  is permitted on an enum reference and denotes a space-separated keyword set --
  the keywords are OR'd into a bitmask (MJCF reads these via MuJoCo's
  ``MapValues``, e.g. a rangefinder's ``data="dist point normal"``). ``NAME[]`` on
  a struct reference is a validation error; sized arity on any named type is a
  parse error.
* ``required`` is folded into the field's ``optional`` boolean (DR-1: everything
  not ``required`` is presence-tracked); it is not echoed as an annotation.
* The default XML tag for an element/field is ``lower(name)`` and is NOT
  materialized into the AST -- only a divergent ``xml="..."`` annotation is
  recorded. Emitters apply the ``lower(name)`` default. This keeps the AST free
  of derived data (plan Section 5: "the default binding is the lowercased IDL
  name ... the annotation appears on the handful of fields where MJCF disagrees").

Canonical JSON shape
--------------------
``Schema.to_json()`` returns a plain ``dict`` of JSON-native values. Top-level
definition lists are sorted by name (their order is not semantic); everything
inside a definition (fields, enum members, variant arms, children, uses) stays in
source order (which is semantic). Every node carries integer ``line``/``col``
(1-based). Keys that are null/empty/false are omitted; ``from_json`` restores
them, so ``to_json -> from_json -> to_json`` is a fixpoint.

    Schema      {mujoco_version, enums[], structs[], variants[], unions[],
                 mixins[], elements[], line, col}
    EnumDef     {name, members[], line, col}
    EnumMember  {name, value, doc?, line, col}          # value = XML keyword
    StructDef   {name, fields[], line, col}
    VariantDef  {name, arms[], line, col}
    VariantArm  {tag, type, doc?, line, col}
    UnionDef    {name, members[], line, col}            # members[]: element
                                                        #   names, source order
    MixinDef    {name, fields[], line, col}
    ElementDef  {name, annotations?{xml}, uses[], fields[], children[], line, col}
    ChildDef    {name, (element | union), cardinality, line, col}
                                                         # element XOR union;
                                                         #   union -> a union
                                                         #   name (heterogeneous
                                                         #   child list).
                                                         # cardinality:
                                                         #   zero_or_more|zero_or_one|one
    Field       {name, type, optional, annotations?, default?, doc?,
                 source_mixin?, line, col}
    TypeRef     {kind, ..., line, col}
                  kind=prim    -> {prim, arity?}
                    arity      -> {kind=fixed, size} | {kind=range, min, max}
                                | {kind=unbounded}
                  kind=ref     -> {target}            # target: element or union
                  kind=variant -> {target}            # target is a variant name
                  kind=named   -> {name, arity?, category}  # category: enum|struct
                    arity      -> {kind=unbounded}     # enum keyword set only
    DefaultValue {kind, ..., line, col}
                  kind=scalar  -> {value}             # number | bool | str
                  kind=enum    -> {member}            # bare identifier
                  kind=array   -> {values[]}          # brace array of literals

``fields[]`` on an element is the FLATTENED list: mixin-injected fields (in
``use`` order, then mixin field order) precede local fields, and each injected
field carries ``source_mixin`` naming its origin. ``from_json`` does not re-run
validation or flattening -- the JSON is already the resolved AST.

Public API
----------
``parse_spec(path_or_text) -> Schema`` and ``SchemaError`` are the only public
names; everything else is private.
"""

from __future__ import annotations

import copy
import json
import os
import re
from dataclasses import dataclass, field as _dc_field
from typing import Any, Optional

__all__ = ["parse_spec", "Schema", "SchemaError"]


_PRIMS = frozenset({"int32", "uint64", "float", "double", "bool", "string"})
_FIELD_ANNOTS = frozenset(
    {"xml", "unit", "required", "variant_group", "variant_tag", "element_text"}
)
_ELEMENT_ANNOTS = frozenset({"xml"})
_FLAG_ANNOTS = frozenset({"required", "element_text"})
_CARDINALITY = {"*": "zero_or_more", "?": "zero_or_one", "!": "one"}


# --------------------------------------------------------------------------- #
# Errors                                                                       #
# --------------------------------------------------------------------------- #
class SchemaError(Exception):
    """A parse or validation error, located in the source.

    Carries structured ``filename``/``line``/``col``/``message`` plus the rendered
    offending source line with a caret, which is also the exception's ``str()``.
    """

    def __init__(
        self,
        message: str,
        *,
        line: int,
        col: int,
        filename: str = "<string>",
        source_line: Optional[str] = None,
    ) -> None:
        self.message = message
        self.line = line
        self.col = col
        self.filename = filename
        self.source_line = source_line
        super().__init__(self._render())

    def _render(self) -> str:
        out = [f"{self.filename}:{self.line}:{self.col}: error: {self.message}"]
        if self.source_line is not None:
            out.append("    " + self.source_line)
            out.append("    " + " " * (self.col - 1) + "^")
        return "\n".join(out)


@dataclass
class _SourceCtx:
    """Filename + split source lines; mints located :class:`SchemaError`s."""

    filename: str
    lines: list[str]

    def error(self, message: str, line: int, col: int) -> SchemaError:
        src = self.lines[line - 1] if 1 <= line <= len(self.lines) else None
        return SchemaError(
            message, line=line, col=col, filename=self.filename, source_line=src
        )


# --------------------------------------------------------------------------- #
# AST nodes                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class TypeRef:
    kind: str  # 'prim' | 'ref' | 'variant' | 'named'
    line: int
    col: int
    prim: Optional[str] = None
    arity: Optional[dict] = None
    target: Optional[str] = None
    name: Optional[str] = None
    category: Optional[str] = None  # 'enum' | 'struct', set during validation

    def to_json(self) -> dict:
        d: dict[str, Any] = {"kind": self.kind}
        if self.kind == "prim":
            d["prim"] = self.prim
            if self.arity is not None:
                d["arity"] = self.arity
        elif self.kind in ("ref", "variant"):
            d["target"] = self.target
        elif self.kind == "named":
            d["name"] = self.name
            if self.arity is not None:
                d["arity"] = self.arity
            if self.category is not None:
                d["category"] = self.category
        d["line"] = self.line
        d["col"] = self.col
        return d

    @staticmethod
    def from_json(d: dict) -> "TypeRef":
        return TypeRef(
            kind=d["kind"],
            line=d["line"],
            col=d["col"],
            prim=d.get("prim"),
            arity=d.get("arity"),
            target=d.get("target"),
            name=d.get("name"),
            category=d.get("category"),
        )


@dataclass
class DefaultValue:
    kind: str  # 'scalar' | 'enum' | 'array'
    line: int
    col: int
    value: Any = None  # scalar
    member: Optional[str] = None  # enum
    values: Optional[list] = None  # array

    def to_json(self) -> dict:
        d: dict[str, Any] = {"kind": self.kind}
        if self.kind == "scalar":
            d["value"] = self.value
        elif self.kind == "enum":
            d["member"] = self.member
        elif self.kind == "array":
            d["values"] = list(self.values or [])
        d["line"] = self.line
        d["col"] = self.col
        return d

    @staticmethod
    def from_json(d: dict) -> "DefaultValue":
        return DefaultValue(
            kind=d["kind"],
            line=d["line"],
            col=d["col"],
            value=d.get("value"),
            member=d.get("member"),
            values=d.get("values"),
        )


@dataclass
class Field:
    name: str
    type: TypeRef
    line: int
    col: int
    optional: bool = True
    annotations: dict = _dc_field(default_factory=dict)
    default: Optional[DefaultValue] = None
    doc: Optional[str] = None
    source_mixin: Optional[str] = None

    def to_json(self) -> dict:
        d: dict[str, Any] = {
            "name": self.name,
            "type": self.type.to_json(),
            "optional": self.optional,
        }
        if self.annotations:
            d["annotations"] = dict(sorted(self.annotations.items()))
        if self.default is not None:
            d["default"] = self.default.to_json()
        if self.doc is not None:
            d["doc"] = self.doc
        if self.source_mixin is not None:
            d["source_mixin"] = self.source_mixin
        d["line"] = self.line
        d["col"] = self.col
        return d

    @staticmethod
    def from_json(d: dict) -> "Field":
        return Field(
            name=d["name"],
            type=TypeRef.from_json(d["type"]),
            line=d["line"],
            col=d["col"],
            optional=d["optional"],
            annotations=dict(d.get("annotations", {})),
            default=DefaultValue.from_json(d["default"]) if "default" in d else None,
            doc=d.get("doc"),
            source_mixin=d.get("source_mixin"),
        )


@dataclass
class EnumMember:
    name: str
    value: str
    line: int
    col: int
    doc: Optional[str] = None

    def to_json(self) -> dict:
        d: dict[str, Any] = {"name": self.name, "value": self.value}
        if self.doc is not None:
            d["doc"] = self.doc
        d["line"] = self.line
        d["col"] = self.col
        return d

    @staticmethod
    def from_json(d: dict) -> "EnumMember":
        return EnumMember(
            name=d["name"], value=d["value"], line=d["line"], col=d["col"],
            doc=d.get("doc"),
        )


@dataclass
class EnumDef:
    name: str
    line: int
    col: int
    members: list[EnumMember] = _dc_field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "members": [m.to_json() for m in self.members],
            "line": self.line,
            "col": self.col,
        }

    @staticmethod
    def from_json(d: dict) -> "EnumDef":
        return EnumDef(
            name=d["name"], line=d["line"], col=d["col"],
            members=[EnumMember.from_json(m) for m in d["members"]],
        )


@dataclass
class StructDef:
    name: str
    line: int
    col: int
    fields: list[Field] = _dc_field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "fields": [f.to_json() for f in self.fields],
            "line": self.line,
            "col": self.col,
        }

    @staticmethod
    def from_json(d: dict) -> "StructDef":
        return StructDef(
            name=d["name"], line=d["line"], col=d["col"],
            fields=[Field.from_json(f) for f in d["fields"]],
        )


@dataclass
class VariantArm:
    tag: str
    type: TypeRef
    line: int
    col: int
    doc: Optional[str] = None

    def to_json(self) -> dict:
        d: dict[str, Any] = {"tag": self.tag, "type": self.type.to_json()}
        if self.doc is not None:
            d["doc"] = self.doc
        d["line"] = self.line
        d["col"] = self.col
        return d

    @staticmethod
    def from_json(d: dict) -> "VariantArm":
        return VariantArm(
            tag=d["tag"], type=TypeRef.from_json(d["type"]),
            line=d["line"], col=d["col"], doc=d.get("doc"),
        )


@dataclass
class VariantDef:
    name: str
    line: int
    col: int
    arms: list[VariantArm] = _dc_field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "arms": [a.to_json() for a in self.arms],
            "line": self.line,
            "col": self.col,
        }

    @staticmethod
    def from_json(d: dict) -> "VariantDef":
        return VariantDef(
            name=d["name"], line=d["line"], col=d["col"],
            arms=[VariantArm.from_json(a) for a in d["arms"]],
        )


@dataclass
class _UnionMember:
    """A ``union`` member (an element name plus provenance for member errors).

    Serialized as a bare name in the union's ``members`` list; line/col are not
    part of the JSON contract (mirrors :class:`_Use`)."""

    name: str
    line: int
    col: int


@dataclass
class UnionDef:
    name: str
    line: int
    col: int
    members: list[_UnionMember] = _dc_field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "members": [m.name for m in self.members],
            "line": self.line,
            "col": self.col,
        }

    @staticmethod
    def from_json(d: dict) -> "UnionDef":
        return UnionDef(
            name=d["name"], line=d["line"], col=d["col"],
            members=[_UnionMember(name=n, line=0, col=0) for n in d["members"]],
        )


@dataclass
class MixinDef:
    name: str
    line: int
    col: int
    fields: list[Field] = _dc_field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "fields": [f.to_json() for f in self.fields],
            "line": self.line,
            "col": self.col,
        }

    @staticmethod
    def from_json(d: dict) -> "MixinDef":
        return MixinDef(
            name=d["name"], line=d["line"], col=d["col"],
            fields=[Field.from_json(f) for f in d["fields"]],
        )


@dataclass
class ChildDef:
    """An ordered child list. Its items are all one ``element``, or -- when
    ``union`` is set instead -- any member of that union (a heterogeneous list
    preserving source order). Exactly one of ``element``/``union`` is set; at
    parse time the target name lands in ``element`` and validation reclassifies
    it to ``union`` once the symbol resolves."""

    name: str
    cardinality: str
    line: int
    col: int
    element: Optional[str] = None
    union: Optional[str] = None

    def to_json(self) -> dict:
        d: dict[str, Any] = {"name": self.name}
        if self.union is not None:
            d["union"] = self.union
        else:
            d["element"] = self.element
        d["cardinality"] = self.cardinality
        d["line"] = self.line
        d["col"] = self.col
        return d

    @staticmethod
    def from_json(d: dict) -> "ChildDef":
        return ChildDef(
            name=d["name"], cardinality=d["cardinality"],
            line=d["line"], col=d["col"],
            element=d.get("element"), union=d.get("union"),
        )


@dataclass
class _Use:
    """An element's ``use MIXIN`` statement (provenance for flatten errors).

    Serialized as a bare name in the element's ``uses`` list; line/col are not
    part of the JSON contract.
    """

    name: str
    line: int
    col: int


@dataclass
class ElementDef:
    name: str
    line: int
    col: int
    annotations: dict = _dc_field(default_factory=dict)
    uses: list[_Use] = _dc_field(default_factory=list)
    fields: list[Field] = _dc_field(default_factory=list)
    children: list[ChildDef] = _dc_field(default_factory=list)

    def to_json(self) -> dict:
        d: dict[str, Any] = {"name": self.name}
        if self.annotations:
            d["annotations"] = dict(sorted(self.annotations.items()))
        d["uses"] = [u.name for u in self.uses]
        d["fields"] = [f.to_json() for f in self.fields]
        d["children"] = [c.to_json() for c in self.children]
        d["line"] = self.line
        d["col"] = self.col
        return d

    @staticmethod
    def from_json(d: dict) -> "ElementDef":
        return ElementDef(
            name=d["name"],
            line=d["line"],
            col=d["col"],
            annotations=dict(d.get("annotations", {})),
            uses=[_Use(name=n, line=0, col=0) for n in d.get("uses", [])],
            fields=[Field.from_json(f) for f in d["fields"]],
            children=[ChildDef.from_json(c) for c in d["children"]],
        )


@dataclass
class Schema:
    """A parsed, validated, flattened ProtoSpec schema (the AST root)."""

    mujoco_version: str
    line: int
    col: int
    enums: list[EnumDef] = _dc_field(default_factory=list)
    structs: list[StructDef] = _dc_field(default_factory=list)
    variants: list[VariantDef] = _dc_field(default_factory=list)
    unions: list[UnionDef] = _dc_field(default_factory=list)
    mixins: list[MixinDef] = _dc_field(default_factory=list)
    elements: list[ElementDef] = _dc_field(default_factory=list)

    def to_json(self) -> dict:
        def by_name(xs):
            return sorted(xs, key=lambda n: n.name)

        return {
            "mujoco_version": self.mujoco_version,
            "enums": [e.to_json() for e in by_name(self.enums)],
            "structs": [s.to_json() for s in by_name(self.structs)],
            "variants": [v.to_json() for v in by_name(self.variants)],
            "unions": [u.to_json() for u in by_name(self.unions)],
            "mixins": [m.to_json() for m in by_name(self.mixins)],
            "elements": [e.to_json() for e in by_name(self.elements)],
            "line": self.line,
            "col": self.col,
        }

    @staticmethod
    def from_json(d: dict) -> "Schema":
        return Schema(
            mujoco_version=d["mujoco_version"],
            line=d["line"],
            col=d["col"],
            enums=[EnumDef.from_json(x) for x in d["enums"]],
            structs=[StructDef.from_json(x) for x in d["structs"]],
            variants=[VariantDef.from_json(x) for x in d["variants"]],
            unions=[UnionDef.from_json(x) for x in d.get("unions", [])],
            mixins=[MixinDef.from_json(x) for x in d["mixins"]],
            elements=[ElementDef.from_json(x) for x in d["elements"]],
        )

    def to_json_str(self, *, indent: int = 2) -> str:
        """Deterministic pretty JSON string (sorted object keys)."""
        return json.dumps(self.to_json(), indent=indent, sort_keys=True)


# --------------------------------------------------------------------------- #
# Tokenizer                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class _Token:
    type: str  # 'STRING' | 'NUMBER' | 'NAME' | 'PUNCT' | 'RANGE' | 'EOF'
    value: str
    line: int
    col: int


_TOKEN_RE = re.compile(
    r"""
      (?P<WS>[ \t\r\f\v]+)
    | (?P<STRING>"(?:\\.|[^"\\])*")
    | (?P<NUMBER>-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
    | (?P<NAME>[A-Za-z_][A-Za-z0-9_]*)
    | (?P<RANGE>\.\.)
    | (?P<PUNCT>[{}()\[\]<>:=,*?!|])
    """,
    re.VERBOSE,
)


def _tokenize(text: str, ctx: _SourceCtx) -> tuple[list[_Token], dict[int, str]]:
    """Return (tokens, comments-by-line). Comments (``# ...``) are stripped from
    the token stream; a comment is recorded against the line it appears on so the
    parser can attach a trailing comment as a field/member/arm doc."""
    tokens: list[_Token] = []
    comments: dict[int, str] = {}
    for lineno, line in enumerate(ctx.lines, 1):
        col = 0
        n = len(line)
        while col < n:
            ch = line[col]
            if ch == "#":
                comments[lineno] = line[col + 1 :].strip()
                break
            m = _TOKEN_RE.match(line, col)
            if not m:
                raise ctx.error(f"unexpected character {ch!r}", lineno, col + 1)
            kind = m.lastgroup
            start = m.start()
            col = m.end()
            if kind == "WS":
                continue
            tokens.append(_Token(kind, m.group(), lineno, start + 1))
    last_line = len(ctx.lines) if ctx.lines else 1
    last_col = (len(ctx.lines[-1]) + 1) if ctx.lines else 1
    tokens.append(_Token("EOF", "", last_line, last_col))
    return tokens, comments


def _unquote(tok: _Token) -> str:
    try:
        return json.loads(tok.value)
    except (ValueError, json.JSONDecodeError):
        return tok.value[1:-1]


def _parse_number(s: str):
    return float(s) if any(c in s for c in ".eE") else int(s)


# --------------------------------------------------------------------------- #
# Parser                                                                        #
# --------------------------------------------------------------------------- #
class _Parser:
    def __init__(self, tokens: list[_Token], comments: dict[int, str], ctx: _SourceCtx):
        self._toks = tokens
        self._comments = comments
        self._ctx = ctx
        self._i = 0
        self._last: _Token = tokens[0]

    # -- cursor -------------------------------------------------------------- #
    def _peek(self) -> _Token:
        return self._toks[self._i]

    def _at_end(self) -> bool:
        return self._toks[self._i].type == "EOF"

    def _advance(self) -> _Token:
        tok = self._toks[self._i]
        if tok.type != "EOF":
            self._last = tok
            self._i += 1
        return tok

    def _check(self, type_: str, value: Optional[str] = None) -> bool:
        t = self._toks[self._i]
        return t.type == type_ and (value is None or t.value == value)

    def _check_kw(self, value: str) -> bool:
        return self._check("NAME", value)

    def _error(self, message: str, tok: Optional[_Token] = None) -> SchemaError:
        tok = tok or self._peek()
        return self._ctx.error(message, tok.line, tok.col)

    @staticmethod
    def _describe(tok: _Token) -> str:
        return "end of file" if tok.type == "EOF" else repr(tok.value)

    def _expect(self, type_: str, value: Optional[str] = None, what: str = "") -> _Token:
        t = self._peek()
        if t.type == type_ and (value is None or t.value == value):
            return self._advance()
        exp = what or (repr(value) if value is not None else type_.lower())
        raise self._error(f"expected {exp}, got {self._describe(t)}", t)

    def _expect_punct(self, value: str) -> _Token:
        return self._expect("PUNCT", value, repr(value))

    def _expect_name(self, what: str = "a name") -> _Token:
        return self._expect("NAME", None, what)

    def _trailing_doc(self) -> Optional[str]:
        """Comment on the line of the most recently consumed token, if any."""
        doc = self._comments.get(self._last.line)
        return doc if doc else None

    # -- top level ----------------------------------------------------------- #
    def parse(self) -> Schema:
        if self._at_end():
            raise self._error(
                "empty schema: expected a 'mujoco_version' header", self._peek()
            )
        header = self._peek()
        if not self._check_kw("mujoco_version"):
            raise self._error(
                'schema must begin with a header: mujoco_version "<version>"', header
            )
        self._advance()
        ver = self._expect("STRING", what="a version string")
        schema = Schema(
            mujoco_version=_unquote(ver), line=header.line, col=header.col
        )
        while not self._at_end():
            tok = self._peek()
            if tok.type != "NAME":
                raise self._error(
                    "expected a top-level definition "
                    "(enum, struct, variant, union, mixin, element)",
                    tok,
                )
            kw = tok.value
            if kw == "enum":
                schema.enums.append(self._parse_enum())
            elif kw == "struct":
                schema.structs.append(self._parse_struct())
            elif kw == "variant":
                schema.variants.append(self._parse_variant_def())
            elif kw == "union":
                schema.unions.append(self._parse_union_def())
            elif kw == "mixin":
                schema.mixins.append(self._parse_mixin())
            elif kw == "element":
                schema.elements.append(self._parse_element())
            else:
                raise self._error(
                    f"unexpected {kw!r}: expected "
                    "enum, struct, variant, union, mixin, or element",
                    tok,
                )
        return schema

    # -- definitions --------------------------------------------------------- #
    def _parse_enum(self) -> EnumDef:
        kw = self._advance()  # 'enum'
        name = self._expect_name("an enum name")
        node = EnumDef(name=name.value, line=name.line, col=name.col)
        open_brace = self._expect_punct("{")
        while True:
            if self._check("PUNCT", "}"):
                self._advance()
                break
            if self._at_end():
                raise self._error("unbalanced '{': missing closing '}'", open_brace)
            mname = self._expect_name("an enum member name")
            self._expect_punct("=")
            val = self._expect("STRING", what="an XML keyword string")
            node.members.append(
                EnumMember(
                    name=mname.value,
                    value=_unquote(val),
                    line=mname.line,
                    col=mname.col,
                    doc=self._trailing_doc(),
                )
            )
        if not node.members:
            raise self._error(
                f"enum {name.value!r} must declare at least one member", kw
            )
        return node

    def _parse_struct(self) -> StructDef:
        self._advance()  # 'struct'
        name = self._expect_name("a struct name")
        node = StructDef(name=name.value, line=name.line, col=name.col)
        node.fields = self._parse_field_block()
        return node

    def _parse_mixin(self) -> MixinDef:
        self._advance()  # 'mixin'
        name = self._expect_name("a mixin name")
        node = MixinDef(name=name.value, line=name.line, col=name.col)
        node.fields = self._parse_field_block()
        return node

    def _parse_field_block(self) -> list[Field]:
        open_brace = self._expect_punct("{")
        fields: list[Field] = []
        while True:
            if self._check("PUNCT", "}"):
                self._advance()
                break
            if self._at_end():
                raise self._error("unbalanced '{': missing closing '}'", open_brace)
            fields.append(self._parse_field())
        return fields

    def _parse_variant_def(self) -> VariantDef:
        kw = self._advance()  # 'variant'
        name = self._expect_name("a variant name")
        node = VariantDef(name=name.value, line=name.line, col=name.col)
        open_brace = self._expect_punct("{")
        while True:
            if self._check("PUNCT", "}"):
                self._advance()
                break
            if self._at_end():
                raise self._error("unbalanced '{': missing closing '}'", open_brace)
            tag = self._expect_name("a variant tag")
            self._expect_punct(":")
            typ = self._parse_type()
            node.arms.append(
                VariantArm(
                    tag=tag.value,
                    type=typ,
                    line=tag.line,
                    col=tag.col,
                    doc=self._trailing_doc(),
                )
            )
        if not node.arms:
            raise self._error(
                f"variant {name.value!r} must declare at least one arm", kw
            )
        return node

    def _parse_union_def(self) -> UnionDef:
        self._advance()  # 'union'
        name = self._expect_name("a union name")
        node = UnionDef(name=name.value, line=name.line, col=name.col)
        self._expect_punct("=")
        first = self._expect_name("a union member element name")
        node.members.append(
            _UnionMember(name=first.value, line=first.line, col=first.col)
        )
        while self._check("PUNCT", "|"):
            self._advance()
            m = self._expect_name("a union member element name")
            node.members.append(_UnionMember(name=m.value, line=m.line, col=m.col))
        return node

    def _parse_element(self) -> ElementDef:
        self._advance()  # 'element'
        name = self._expect_name("an element name")
        node = ElementDef(name=name.value, line=name.line, col=name.col)
        if self._check("PUNCT", "("):
            raw = self._parse_annots()
            node.annotations, _ = self._normalize_annots(raw, _ELEMENT_ANNOTS, "element")
        open_brace = self._expect_punct("{")
        while True:
            if self._check("PUNCT", "}"):
                self._advance()
                break
            if self._at_end():
                raise self._error("unbalanced '{': missing closing '}'", open_brace)
            tok = self._peek()
            if tok.type == "NAME" and tok.value == "use":
                self._advance()
                mixin = self._expect_name("a mixin name")
                node.uses.append(_Use(name=mixin.value, line=mixin.line, col=mixin.col))
            elif tok.type == "NAME" and tok.value == "children":
                node.children.append(self._parse_child())
            else:
                node.fields.append(self._parse_field())
        return node

    def _parse_child(self) -> ChildDef:
        self._advance()  # 'children'
        fname = self._expect_name("a child-list name")
        self._expect_punct(":")
        elem = self._expect_name("a child element type")
        card = self._peek()
        if card.type != "PUNCT" or card.value not in _CARDINALITY:
            raise self._error(
                "expected a cardinality marker '*', '?', or '!' "
                f"after child element type, got {self._describe(card)}",
                card,
            )
        self._advance()
        # Located at the child element type so resolution errors point there.
        # The target lands in ``element``; validation reclassifies it to
        # ``union`` if the name resolves to a union.
        return ChildDef(
            name=fname.value,
            element=elem.value,
            cardinality=_CARDINALITY[card.value],
            line=elem.line,
            col=elem.col,
        )

    # -- fields -------------------------------------------------------------- #
    def _parse_field(self) -> Field:
        name = self._expect_name("a field name")
        self._expect_punct(":")
        typ = self._parse_type()
        annotations: dict = {}
        required = False
        if self._check("PUNCT", "("):
            raw = self._parse_annots()
            annotations, required = self._normalize_annots(raw, _FIELD_ANNOTS, "field")
        default = None
        if self._check("PUNCT", "="):
            default = self._parse_default()
        return Field(
            name=name.value,
            type=typ,
            line=name.line,
            col=name.col,
            optional=not required,
            annotations=annotations,
            default=default,
            doc=self._trailing_doc(),
        )

    def _parse_type(self) -> TypeRef:
        tok = self._peek()
        if tok.type != "NAME":
            raise self._error(
                f"expected a type, got {self._describe(tok)}", tok
            )
        if tok.value in _PRIMS:
            self._advance()
            arity = self._parse_arity() if self._check("PUNCT", "[") else None
            return TypeRef(
                kind="prim", line=tok.line, col=tok.col, prim=tok.value, arity=arity
            )
        if tok.value == "ref":
            self._advance()
            self._expect_punct("<")
            target = self._expect_name("an element name inside ref<>")
            self._expect_punct(">")
            return TypeRef(
                kind="ref", line=tok.line, col=tok.col, target=target.value
            )
        if tok.value == "variant":
            self._advance()
            target = self._expect_name("a variant name")
            return TypeRef(
                kind="variant", line=tok.line, col=tok.col, target=target.value
            )
        # A bare name: an enum or struct reference (resolved during validation).
        # Sized arities (``[N]`` / ``[N..M]``) are only valid on primitives; a
        # named type may carry the unbounded form ``[]`` alone, denoting a
        # space-separated keyword set (a bitmask over an enum's keywords, e.g.
        # ``data : RayData[]``). The enum-only restriction is enforced during
        # validation, once the name resolves.
        self._advance()
        arity = None
        if self._check("PUNCT", "["):
            open_bracket = self._peek()
            arity = self._parse_arity()
            if arity["kind"] != "unbounded":
                raise self._error(
                    "a sized array arity ('[N]' / '[N..M]') is only valid on "
                    "primitive types (int32, uint64, float, double, bool, string), "
                    f"not {tok.value!r}; an enum reference may only carry the "
                    "unbounded form '[]' (a space-separated keyword set)",
                    open_bracket,
                )
        return TypeRef(
            kind="named", line=tok.line, col=tok.col, name=tok.value, arity=arity
        )

    def _parse_arity(self) -> dict:
        self._expect_punct("[")
        if self._check("PUNCT", "]"):
            self._advance()
            return {"kind": "unbounded"}
        lo = self._expect("NUMBER", what="an integer size")
        if "." in lo.value or "e" in lo.value or "E" in lo.value:
            raise self._error("array size must be an integer", lo)
        lo_n = int(lo.value)
        if lo_n < 0:
            raise self._error("array size must be non-negative", lo)
        if self._check("RANGE"):
            self._advance()
            hi = self._expect("NUMBER", what="an integer upper bound")
            if "." in hi.value or "e" in hi.value or "E" in hi.value:
                raise self._error("array upper bound must be an integer", hi)
            hi_n = int(hi.value)
            self._expect_punct("]")
            if hi_n < lo_n:
                raise self._error(
                    f"array range upper bound {hi_n} is less than lower bound {lo_n}",
                    hi,
                )
            return {"kind": "range", "min": lo_n, "max": hi_n}
        self._expect_punct("]")
        return {"kind": "fixed", "size": lo_n}

    def _parse_default(self) -> DefaultValue:
        eq = self._advance()  # '='
        tok = self._peek()
        if tok.type == "PUNCT" and tok.value == "{":
            return self._parse_array_default(eq)
        if tok.type == "STRING":
            self._advance()
            return DefaultValue(
                kind="scalar", line=tok.line, col=tok.col, value=_unquote(tok)
            )
        if tok.type == "NUMBER":
            self._advance()
            return DefaultValue(
                kind="scalar", line=tok.line, col=tok.col, value=_parse_number(tok.value)
            )
        if tok.type == "NAME":
            self._advance()
            if tok.value in ("true", "false"):
                return DefaultValue(
                    kind="scalar", line=tok.line, col=tok.col, value=(tok.value == "true")
                )
            return DefaultValue(
                kind="enum", line=tok.line, col=tok.col, member=tok.value
            )
        raise self._error(
            "expected a default value: a literal, an enum member name, "
            f"or a brace array, got {self._describe(tok)}",
            tok,
        )

    def _parse_array_default(self, at: _Token) -> DefaultValue:
        self._expect_punct("{")
        values: list = []
        first = True
        while True:
            if self._check("PUNCT", "}"):
                self._advance()
                break
            if self._at_end():
                raise self._error("unbalanced '{': missing closing '}'", at)
            if not first:
                self._expect_punct(",")
            first = False
            tok = self._peek()
            if tok.type == "STRING":
                self._advance()
                values.append(_unquote(tok))
            elif tok.type == "NUMBER":
                self._advance()
                values.append(_parse_number(tok.value))
            elif tok.type == "NAME" and tok.value in ("true", "false"):
                self._advance()
                values.append(tok.value == "true")
            else:
                raise self._error(
                    "brace-array default elements must be literals "
                    f"(number, bool, or string), got {self._describe(tok)}",
                    tok,
                )
        return DefaultValue(kind="array", line=at.line, col=at.col, values=values)

    # -- annotations --------------------------------------------------------- #
    def _parse_annots(self) -> list[tuple]:
        """Parse ``( key (= value)? , ... )`` into raw tuples; validation of the
        key set and value kinds happens in :meth:`_normalize_annots`."""
        self._expect_punct("(")
        raw: list[tuple] = []
        first = True
        while True:
            if self._check("PUNCT", ")"):
                self._advance()
                break
            if self._at_end():
                raise self._error("unbalanced '(': missing closing ')'", self._peek())
            if not first:
                self._expect_punct(",")
            first = False
            key = self._expect_name("an annotation key")
            value = None
            is_string = False
            if self._check("PUNCT", "="):
                self._advance()
                vtok = self._peek()
                if vtok.type == "STRING":
                    self._advance()
                    value = _unquote(vtok)
                    is_string = True
                elif vtok.type == "NAME":
                    self._advance()
                    value = vtok.value
                elif vtok.type == "NUMBER":
                    self._advance()
                    value = vtok.value
                else:
                    raise self._error(
                        "expected an annotation value (string or identifier), "
                        f"got {self._describe(vtok)}",
                        vtok,
                    )
            raw.append((key.value, value, is_string, key))
        return raw

    def _normalize_annots(
        self, raw: list[tuple], allowed: frozenset, context: str
    ) -> tuple[dict, bool]:
        result: dict = {}
        required = False
        for key, value, is_string, tok in raw:
            if key not in _FIELD_ANNOTS:
                raise self._error(f"unknown annotation key {key!r}", tok)
            if key not in allowed:
                raise self._error(
                    f"annotation {key!r} is not valid on {context}s", tok
                )
            if key == "required":
                if required:
                    raise self._error("duplicate annotation 'required'", tok)
                if value is not None:
                    raise self._error("annotation 'required' takes no value", tok)
                required = True
                continue
            if key in result:
                raise self._error(f"duplicate annotation {key!r}", tok)
            if key in _FLAG_ANNOTS:
                if value is not None:
                    raise self._error(f"annotation {key!r} takes no value", tok)
                result[key] = True
            elif key == "xml":
                if value is None or not is_string:
                    raise self._error(
                        "annotation 'xml' requires a string value, e.g. xml=\"class\"",
                        tok,
                    )
                result[key] = value
            elif key == "unit":
                if value is None:
                    raise self._error("annotation 'unit' requires a value", tok)
                if value != "angle":
                    raise self._error(
                        f"annotation 'unit' only accepts 'angle', got {value!r}", tok
                    )
                result[key] = value
            else:  # variant_group / variant_tag
                if value is None:
                    raise self._error(f"annotation {key!r} requires a value", tok)
                result[key] = value
        return result, required


# --------------------------------------------------------------------------- #
# Validation + flattening                                                       #
# --------------------------------------------------------------------------- #
class _Symbols:
    """Top-level name table: name -> (category, node)."""

    def __init__(self) -> None:
        self._by_name: dict[str, tuple[str, Any]] = {}

    def add(self, category: str, node: Any, ctx: _SourceCtx) -> None:
        prev = self._by_name.get(node.name)
        if prev is not None:
            prev_cat = prev[0]
            raise ctx.error(
                f"duplicate definition of {node.name!r} "
                f"(already defined as a {prev_cat})",
                node.line,
                node.col,
            )
        self._by_name[node.name] = (category, node)

    def get(self, name: str) -> Optional[tuple[str, Any]]:
        return self._by_name.get(name)


def _validate(schema: Schema, ctx: _SourceCtx) -> None:
    symbols = _Symbols()
    for cat, items in (
        ("enum", schema.enums),
        ("struct", schema.structs),
        ("variant", schema.variants),
        ("union", schema.unions),
        ("mixin", schema.mixins),
        ("element", schema.elements),
    ):
        for node in items:
            symbols.add(cat, node, ctx)

    for enum in schema.enums:
        _validate_enum(enum, ctx)

    for struct in schema.structs:
        _check_dup_fields(struct.fields, ctx)
        for f in struct.fields:
            _resolve_and_validate_field(f, symbols, ctx)

    for mixin in schema.mixins:
        _check_dup_fields(mixin.fields, ctx)
        for f in mixin.fields:
            _resolve_and_validate_field(f, symbols, ctx)

    for variant in schema.variants:
        _validate_variant(variant, symbols, ctx)

    for union in schema.unions:
        _validate_union(union, symbols, ctx)

    for elem in schema.elements:
        for f in elem.fields:
            _resolve_and_validate_field(f, symbols, ctx)
        _flatten_element(elem, symbols, ctx)
        _validate_children(elem, symbols, ctx)


def _validate_enum(enum: EnumDef, ctx: _SourceCtx) -> None:
    seen: dict[str, EnumMember] = {}
    for m in enum.members:
        if m.name in seen:
            raise ctx.error(
                f"duplicate enum member {m.name!r} in enum {enum.name!r}",
                m.line,
                m.col,
            )
        seen[m.name] = m


def _check_dup_fields(fields: list[Field], ctx: _SourceCtx) -> None:
    seen: dict[str, Field] = {}
    for f in fields:
        if f.name in seen:
            raise ctx.error(f"duplicate field {f.name!r}", f.line, f.col)
        seen[f.name] = f


def _resolve_type(t: TypeRef, symbols: _Symbols, ctx: _SourceCtx) -> None:
    if t.kind == "prim":
        return
    if t.kind == "ref":
        entry = symbols.get(t.target)
        if entry is None:
            raise ctx.error(
                f"ref<{t.target}> references unknown element {t.target!r}",
                t.line,
                t.col,
            )
        # A ref may target an element (one namespace) or a union (the union of
        # its members' namespaces, e.g. ref<TendonAny> = spatial or fixed).
        if entry[0] not in ("element", "union"):
            raise ctx.error(
                f"ref<{t.target}> target {t.target!r} is a {entry[0]}, not an element",
                t.line,
                t.col,
            )
        return
    if t.kind == "variant":
        entry = symbols.get(t.target)
        if entry is None:
            raise ctx.error(f"unknown variant {t.target!r}", t.line, t.col)
        if entry[0] != "variant":
            raise ctx.error(
                f"{t.target!r} is a {entry[0]}, not a variant", t.line, t.col
            )
        return
    # named: enum or struct
    entry = symbols.get(t.name)
    if entry is None:
        raise ctx.error(f"unknown type {t.name!r}", t.line, t.col)
    cat = entry[0]
    if cat == "element":
        raise ctx.error(
            f"{t.name!r} is an element; use ref<{t.name}> to reference it",
            t.line,
            t.col,
        )
    if cat == "variant":
        raise ctx.error(
            f"{t.name!r} is a variant; write 'variant {t.name}'", t.line, t.col
        )
    if cat == "union":
        raise ctx.error(
            f"{t.name!r} is a union and cannot be used as a field type; use it in "
            f"a child list ('children NAME : {t.name} *') or a reference "
            f"('ref<{t.name}>')",
            t.line,
            t.col,
        )
    if cat == "mixin":
        raise ctx.error(
            f"{t.name!r} is a mixin and cannot be used as a field type",
            t.line,
            t.col,
        )
    if t.arity is not None and cat != "enum":
        raise ctx.error(
            f"unbounded arity '[]' is only valid on an enum reference "
            f"(a space-separated keyword set), not the {cat} {t.name!r}",
            t.line,
            t.col,
        )
    t.category = cat  # 'enum' | 'struct'


def _resolve_and_validate_field(f: Field, symbols: _Symbols, ctx: _SourceCtx) -> None:
    _resolve_type(f.type, symbols, ctx)
    _validate_default(f, symbols, ctx)


def _validate_default(f: Field, symbols: _Symbols, ctx: _SourceCtx) -> None:
    d = f.default
    if d is None:
        return
    t = f.type
    if d.kind == "array":
        if not (t.kind == "prim" and t.arity is not None):
            raise ctx.error(
                f"brace-array default is only valid for an array-typed field, "
                f"but {f.name!r} is not",
                d.line,
                d.col,
            )
        n = len(d.values or [])
        a = t.arity
        if a["kind"] == "fixed" and n != a["size"]:
            raise ctx.error(
                f"default for {f.name!r} has {n} values but the field arity is "
                f"[{a['size']}]",
                d.line,
                d.col,
            )
        if a["kind"] == "range" and not (a["min"] <= n <= a["max"]):
            raise ctx.error(
                f"default for {f.name!r} has {n} values, outside the field arity "
                f"[{a['min']}..{a['max']}]",
                d.line,
                d.col,
            )
        return
    if d.kind == "enum":
        if t.kind != "named" or t.category != "enum":
            raise ctx.error(
                f"bare-identifier default {d.member!r} is only valid on an "
                f"enum-typed field, but {f.name!r} is not enum-typed",
                d.line,
                d.col,
            )
        if t.arity is not None:
            raise ctx.error(
                f"bare-identifier default {d.member!r} is not valid on the "
                f"keyword-set field {f.name!r}; a list-typed enum has no scalar "
                "default",
                d.line,
                d.col,
            )
        enum = symbols.get(t.name)[1]
        if d.member not in {m.name for m in enum.members}:
            raise ctx.error(
                f"{d.member!r} is not a member of enum {t.name!r}", d.line, d.col
            )
        return
    # scalar
    if t.kind == "prim" and t.arity is not None:
        raise ctx.error(
            f"scalar default is not valid for the array-typed field {f.name!r}; "
            "use a brace array",
            d.line,
            d.col,
        )


def _validate_variant(variant: VariantDef, symbols: _Symbols, ctx: _SourceCtx) -> None:
    seen: dict[str, VariantArm] = {}
    for arm in variant.arms:
        if arm.tag in seen:
            raise ctx.error(
                f"duplicate arm {arm.tag!r} in variant {variant.name!r}",
                arm.line,
                arm.col,
            )
        seen[arm.tag] = arm
        t = arm.type
        if t.kind in ("ref", "variant"):
            raise ctx.error(
                "variant arms must be a struct or primitive type, "
                f"not a {t.kind}",
                t.line,
                t.col,
            )
        _resolve_type(t, symbols, ctx)


def _validate_union(union: UnionDef, symbols: _Symbols, ctx: _SourceCtx) -> None:
    seen: dict[str, _UnionMember] = {}
    for m in union.members:
        if m.name in seen:
            raise ctx.error(
                f"duplicate member {m.name!r} in union {union.name!r}",
                m.line,
                m.col,
            )
        seen[m.name] = m
        entry = symbols.get(m.name)
        if entry is None:
            raise ctx.error(
                f"union {union.name!r} references unknown element {m.name!r}",
                m.line,
                m.col,
            )
        cat = entry[0]
        if cat == "union":
            raise ctx.error(
                f"union {union.name!r} cannot contain another union "
                f"({m.name!r}); unions may only contain elements",
                m.line,
                m.col,
            )
        if cat != "element":
            raise ctx.error(
                f"union member {m.name!r} is a {cat}, not an element",
                m.line,
                m.col,
            )


def _flatten_element(elem: ElementDef, symbols: _Symbols, ctx: _SourceCtx) -> None:
    """Inject mixin fields (in ``use`` order, then mixin field order) before the
    element's local fields, detecting duplicate field names across the union."""
    flattened: list[Field] = []
    seen: dict[str, Field] = {}
    for use in elem.uses:
        entry = symbols.get(use.name)
        if entry is None:
            raise ctx.error(
                f"'use {use.name}' references undefined mixin {use.name!r}",
                use.line,
                use.col,
            )
        if entry[0] != "mixin":
            raise ctx.error(
                f"'use {use.name}' names a {entry[0]}, not a mixin",
                use.line,
                use.col,
            )
        for src in entry[1].fields:
            if src.name in seen:
                origin = seen[src.name].source_mixin or "this element"
                raise ctx.error(
                    f"mixin {use.name!r} injects duplicate field {src.name!r} "
                    f"(already provided by {origin})",
                    use.line,
                    use.col,
                )
            injected = copy.copy(src)
            injected.source_mixin = use.name
            seen[src.name] = injected
            flattened.append(injected)
    for f in elem.fields:
        if f.name in seen:
            origin = seen[f.name].source_mixin
            where = f"mixin {origin!r}" if origin else "an earlier field"
            raise ctx.error(
                f"duplicate field {f.name!r} (already provided by {where})",
                f.line,
                f.col,
            )
        seen[f.name] = f
        flattened.append(f)
    elem.fields = flattened


def _validate_children(elem: ElementDef, symbols: _Symbols, ctx: _SourceCtx) -> None:
    member_names = {f.name for f in elem.fields}
    seen: set[str] = set()
    for c in elem.children:
        if c.name in seen or c.name in member_names:
            raise ctx.error(
                f"duplicate member {c.name!r} in element {elem.name!r}",
                c.line,
                c.col,
            )
        seen.add(c.name)
        entry = symbols.get(c.element)
        if entry is None:
            raise ctx.error(
                f"child list {c.name!r} references unknown element {c.element!r}",
                c.line,
                c.col,
            )
        if entry[0] == "union":
            # A union child list: a single ordered, heterogeneous list whose
            # items are any member element (source order preserved).
            c.union = c.element
            c.element = None
            continue
        if entry[0] != "element":
            raise ctx.error(
                f"child list {c.name!r} target {c.element!r} is a {entry[0]}, "
                "not an element",
                c.line,
                c.col,
            )


# --------------------------------------------------------------------------- #
# Public API                                                                    #
# --------------------------------------------------------------------------- #
def parse_spec(path_or_text, *, filename: Optional[str] = None) -> Schema:
    """Parse a ProtoSpec ``.spec`` schema into a validated :class:`Schema`.

    ``path_or_text`` is either a filesystem path (``str``/``os.PathLike`` to an
    existing ``.spec`` file) or the schema source text itself. Parsing runs the
    tokenizer and recursive-descent parser, then AST validation: duplicate-name
    detection, type-reference resolution, mixin flattening with duplicate-field
    detection, and presence semantics (every non-``required`` field is optional).

    Raises :class:`SchemaError` (with ``filename``/``line``/``col`` and a
    caret-rendered source line) on any lexical, syntactic, or semantic error.
    """
    if isinstance(path_or_text, os.PathLike):
        path = os.fspath(path_or_text)
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        filename = filename or path
    elif (
        isinstance(path_or_text, str)
        and "\n" not in path_or_text
        and path_or_text.endswith(".spec")
        and os.path.exists(path_or_text)
    ):
        with open(path_or_text, "r", encoding="utf-8") as fh:
            text = fh.read()
        filename = filename or path_or_text
    else:
        text = path_or_text
        filename = filename or "<string>"

    ctx = _SourceCtx(filename=filename, lines=text.splitlines())
    tokens, comments = _tokenize(text, ctx)
    schema = _Parser(tokens, comments, ctx).parse()
    _validate(schema, ctx)
    return schema
