# Copyright 2026 General Atomics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Rewriting T-SQL so DuckDB runs it.

Two passes, in this order. Placeholders: pymssql's `%s` / `%(name)s`
become DuckDB's `?` / `$name`. Dialect: sqlglot transpiles T-SQL to
DuckDB. Placeholders go first because `%s` is a parse error for sqlglot.
"""

import re

_NAMED = re.compile(r"%\((\w+)\)[sd]")


def convert_placeholders(sql):
    """Return `(sql, style)`; style is 'qmark', 'named' or None.

    Skips string literals (with '' escapes), [bracketed] and "quoted"
    identifiers, and both comment forms, so a `%` in `LIKE '2024%'`
    survives. `%%` is pyformat's escape for one `%`. Mixing the two
    placeholder styles is refused: DuckDB binds either a sequence or a
    mapping, not both.
    """
    out = []
    style = None
    i, n = 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch == "'":
            j = i + 1
            while j < n:
                if sql[j] == "'":
                    if j + 1 < n and sql[j + 1] == "'":
                        j += 2
                        continue
                    break
                j += 1
            out.append(sql[i:j + 1])
            i = j + 1
        elif ch == "[":
            # `]]` inside a bracketed identifier is an escaped `]`.
            j = i + 1
            while j < n:
                if sql[j] == "]":
                    if j + 1 < n and sql[j + 1] == "]":
                        j += 2
                        continue
                    break
                j += 1
            j = min(j, n - 1)
            out.append(sql[i:j + 1])
            i = j + 1
        elif ch == '"':
            j = sql.find('"', i + 1)
            j = n - 1 if j < 0 else j
            out.append(sql[i:j + 1])
            i = j + 1
        elif sql.startswith("--", i):
            j = sql.find("\n", i)
            j = n if j < 0 else j
            out.append(sql[i:j])
            i = j
        elif sql.startswith("/*", i):
            j = sql.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append(sql[i:j])
            i = j
        elif sql.startswith("%%", i):
            out.append("%")
            i += 2
        elif sql.startswith("%s", i) or sql.startswith("%d", i):
            if style == "named":
                raise ValueError("mixed %s and %(name)s placeholders")
            style = "qmark"
            out.append("?")
            i += 2
        elif ch == "%" and _NAMED.match(sql, i):
            if style == "qmark":
                raise ValueError("mixed %s and %(name)s placeholders")
            m = _NAMED.match(sql, i)
            style = "named"
            out.append("$" + m.group(1))
            i = m.end()
        else:
            out.append(ch)
            i += 1
    return "".join(out), style


def transpile(sql, nocase=False):
    """T-SQL in, DuckDB out: `(sql, note)`.

    On a parse failure the input is returned unchanged and `note` carries
    the parser's message: nothing was altered silently, and if DuckDB
    rejects it too the user sees both opinions. Under `nocase` every LIKE
    becomes ILIKE -- T-SQL's LIKE is case-insensitive and DuckDB's is not,
    whatever the collation.

    A tokenizer or generator failure passes through exactly like a parse
    failure. T-SQL LIKE character classes (`[0-9]`) have no DuckDB
    equivalent and are refused with ValueError rather than silently
    returning wrong rows; a pattern supplied as a parameter cannot be
    inspected and is left alone.
    """
    import sqlglot
    from sqlglot import exp

    try:
        trees = [t for t in sqlglot.parse(sql, read="tsql") if t is not None]
        for tree in trees:
            for like in tree.find_all(exp.Like, exp.ILike):
                pattern = like.expression
                if (isinstance(pattern, exp.Literal) and pattern.is_string
                        and "[" in pattern.this):
                    raise ValueError(
                        "T-SQL LIKE character classes ([...]) have no DuckDB "
                        f"equivalent; pattern {pattern.this!r}. "
                        "Use regexp_matches() instead.")
        if nocase:
            for tree in trees:
                for like in list(tree.find_all(exp.Like)):
                    # Carry every arg: `NOT LIKE` is Like(negate=True) in
                    # sqlglot, and ESCAPE rides along too.
                    like.replace(exp.ILike(**like.args))
        return "; ".join(t.sql(dialect="duckdb") for t in trees), None
    except sqlglot.errors.SqlglotError as exc:
        return sql, f"sqlglot {type(exc).__name__}: " + str(exc).splitlines()[0]


def rewrite(sql, nocase=False):
    """Both passes: `(sql, placeholder_style, note)`."""
    sql, style = convert_placeholders(sql)
    sql, note = transpile(sql, nocase=nocase)
    return sql, style, note
