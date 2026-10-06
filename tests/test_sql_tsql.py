"""Rewriting T-SQL for DuckDB: placeholders first, then the dialect.

Placeholders go first because `%s` is a parse error for sqlglot. Anything
inside a string literal, a [bracketed] identifier or a comment is left
alone -- `LIKE '2024%'` is the common case.
"""

import unittest

from toksearch.sql import _tsql


class TestPlaceholders(unittest.TestCase):
    def test_qmark(self):
        sql, style = _tsql.convert_placeholders(
            "SELECT shot FROM shots WHERE shot = %s AND run = %s")
        self.assertEqual(sql, "SELECT shot FROM shots WHERE shot = ? AND run = ?")
        self.assertEqual(style, "qmark")

    def test_named(self):
        sql, style = _tsql.convert_placeholders(
            "SELECT shot FROM shots WHERE shot = %(shot)s")
        self.assertEqual(sql, "SELECT shot FROM shots WHERE shot = $shot")
        self.assertEqual(style, "named")

    def test_no_placeholders(self):
        sql, style = _tsql.convert_placeholders("SELECT 1")
        self.assertEqual((sql, style), ("SELECT 1", None))

    def test_percent_inside_a_string_literal_is_kept(self):
        sql, style = _tsql.convert_placeholders(
            "SELECT shot FROM shots WHERE run LIKE '2024%' AND shot = %s")
        self.assertEqual(
            sql, "SELECT shot FROM shots WHERE run LIKE '2024%' AND shot = ?")
        self.assertEqual(style, "qmark")

    def test_escaped_quote_inside_a_literal(self):
        sql, _ = _tsql.convert_placeholders(
            "SELECT 1 WHERE brief = 'it''s %s' AND shot = %s")
        self.assertEqual(sql, "SELECT 1 WHERE brief = 'it''s %s' AND shot = ?")

    def test_bracketed_identifier_is_kept(self):
        sql, _ = _tsql.convert_placeholders("SELECT [100%s] FROM t WHERE a = %s")
        self.assertEqual(sql, "SELECT [100%s] FROM t WHERE a = ?")

    def test_double_quoted_identifier_is_kept(self):
        sql, _ = _tsql.convert_placeholders('SELECT "100%s" FROM t WHERE a = %s')
        self.assertEqual(sql, 'SELECT "100%s" FROM t WHERE a = ?')

    def test_comments_are_kept(self):
        sql, _ = _tsql.convert_placeholders(
            "SELECT 1 -- %s here\n/* and %(x)s here */ WHERE a = %s")
        self.assertEqual(
            sql, "SELECT 1 -- %s here\n/* and %(x)s here */ WHERE a = ?")

    def test_doubled_percent_is_one_percent(self):
        sql, style = _tsql.convert_placeholders("SELECT 100 %% 7")
        self.assertEqual((sql, style), ("SELECT 100 % 7", None))

    def test_bare_modulo_is_kept(self):
        sql, style = _tsql.convert_placeholders("SELECT shot % 10 FROM shots")
        self.assertEqual((sql, style), ("SELECT shot % 10 FROM shots", None))

    def test_percent_d_is_a_qmark(self):
        sql, style = _tsql.convert_placeholders("WHERE shot = %d")
        self.assertEqual((sql, style), ("WHERE shot = ?", "qmark"))

    def test_named_percent_d(self):
        sql, style = _tsql.convert_placeholders("WHERE shot = %(shot)d")
        self.assertEqual((sql, style), ("WHERE shot = $shot", "named"))

    def test_mixed_percent_d_and_named_refused(self):
        with self.assertRaises(ValueError):
            _tsql.convert_placeholders("WHERE a = %d AND b = %(x)s")

    def test_escaped_bracket_in_identifier(self):
        sql, style = _tsql.convert_placeholders(
            "SELECT [a]]'b] FROM t WHERE x = %s")
        self.assertEqual(
            (sql, style), ("SELECT [a]]'b] FROM t WHERE x = ?", "qmark"))

    def test_mixed_styles_are_refused(self):
        with self.assertRaises(ValueError):
            _tsql.convert_placeholders("SELECT 1 WHERE a = %s AND b = %(b)s")


try:
    import sqlglot  # noqa: F401
    HAVE_SQLGLOT = True
except ImportError:
    HAVE_SQLGLOT = False


@unittest.skipUnless(HAVE_SQLGLOT, "sqlglot not installed (conda-forge: sqlglot)")
class TestTranspile(unittest.TestCase):
    def test_top_becomes_limit(self):
        sql, note = _tsql.transpile("SELECT TOP 50 shot FROM shots ORDER BY shot DESC")
        self.assertEqual(sql, "SELECT shot AS shot FROM shots ORDER BY shot DESC LIMIT 50")
        self.assertIsNone(note)

    def test_isnull_getdate_datediff(self):
        sql, _ = _tsql.transpile(
            "SELECT ISNULL(betanmax, 0) b, DATEDIFF(day, entered, GETDATE()) age FROM summaries")
        self.assertIn("COALESCE(betanmax, 0)", sql)
        self.assertIn("DATE_DIFF('DAY'", sql)
        self.assertIn("CURRENT_TIMESTAMP", sql)

    def test_brackets_and_dbo(self):
        sql, _ = _tsql.transpile("SELECT [shot], [time] FROM dbo.disruption_warning")
        self.assertEqual(
            sql, 'SELECT "shot" AS "shot", "time" AS "time" FROM dbo.disruption_warning')

    def test_placeholders_survive(self):
        sql, _ = _tsql.transpile("SELECT shot FROM shots WHERE shot = ? AND run = $run")
        self.assertEqual(sql, "SELECT shot AS shot FROM shots WHERE shot = ? AND run = $run")

    def test_like_is_case_insensitive_only_under_nocase(self):
        src = "SELECT shot FROM shots WHERE brief LIKE 'ELM%'"
        self.assertIn("LIKE 'ELM%'", _tsql.transpile(src)[0])
        self.assertIn("ILIKE 'ELM%'", _tsql.transpile(src, nocase=True)[0])

    def test_not_like_under_nocase(self):
        sql, _ = _tsql.transpile("SELECT 1 WHERE a NOT LIKE 'x%'", nocase=True)
        self.assertIn("a NOT ILIKE 'x%'", sql)

    def test_parse_failure_passes_the_sql_through_unchanged(self):
        src = "SELECT FROM WHERE GARBAGE ((("
        sql, note = _tsql.transpile(src)
        self.assertEqual(sql, src)
        self.assertIn("ParseError", note)

    def test_tokenizer_failure_passes_through(self):
        src = "select 'abc"
        sql, note = _tsql.transpile(src)
        self.assertEqual(sql, src)
        self.assertIn("TokenError", note)

    def test_like_character_class_is_refused(self):
        for nocase in (False, True):
            with self.assertRaises(ValueError) as cm:
                _tsql.transpile("SELECT 1 WHERE a LIKE '[0-9]%'", nocase=nocase)
            self.assertIn("regexp_matches", str(cm.exception))

    def test_like_without_class_or_with_parameter_is_fine(self):
        _tsql.transpile("SELECT 1 WHERE a LIKE 'x%'")
        _tsql.transpile("SELECT 1 WHERE a LIKE ?")

    def test_two_statements(self):
        sql, _ = _tsql.transpile("SELECT TOP 1 a FROM t; SELECT TOP 2 b FROM u")
        self.assertEqual(sql, "SELECT a AS a FROM t LIMIT 1; SELECT b AS b FROM u LIMIT 2")


@unittest.skipUnless(HAVE_SQLGLOT, "sqlglot not installed (conda-forge: sqlglot)")
class TestResultColumnSpelling(unittest.TestCase):
    """SQL Server names a result column by the query's spelling; DuckDB by
    the stored one. A bare column projection is aliased to its written
    spelling so `df["shot"]` works on both."""

    def t(self, src):
        sql, note = _tsql.transpile(src)
        self.assertIsNone(note)
        return sql

    def test_bare_column(self):
        self.assertEqual(self.t("SELECT shot FROM shots"), "SELECT shot AS shot FROM shots")

    def test_qualified_columns(self):
        self.assertEqual(self.t("SELECT s.shot, s.entered FROM shots s"),
                         "SELECT s.shot AS shot, s.entered AS entered FROM shots AS s")

    def test_written_case_is_kept(self):
        self.assertEqual(self.t("SELECT Shot FROM shots"), "SELECT Shot AS Shot FROM shots")

    def test_star_unchanged(self):
        self.assertEqual(self.t("SELECT * FROM shots"), "SELECT * FROM shots")
        self.assertEqual(self.t("SELECT s.* FROM shots s"), "SELECT s.* FROM shots AS s")

    def test_aliasing_survives_top(self):
        self.assertEqual(self.t("SELECT TOP 5 shot FROM shots"),
                         "SELECT shot AS shot FROM shots LIMIT 5")

    def test_existing_alias_unchanged(self):
        self.assertEqual(self.t("SELECT shot AS s FROM shots"), "SELECT shot AS s FROM shots")

    def test_expression_unchanged(self):
        self.assertEqual(self.t("SELECT count(*) FROM shots"), "SELECT COUNT(*) FROM shots")
        self.assertEqual(self.t("SELECT shot + 1 FROM shots"), "SELECT shot + 1 FROM shots")

    def test_outer_select_of_a_cte(self):
        sql = self.t("WITH c AS (SELECT shot FROM shots) SELECT shot FROM c")
        self.assertTrue(sql.endswith("SELECT shot AS shot FROM c"), sql)

    def test_subquery_body_unchanged(self):
        sql = self.t("SELECT shot FROM (SELECT shot FROM shots) AS x")
        self.assertEqual(sql, "SELECT shot AS shot FROM (SELECT shot FROM shots) AS x")

    def test_union_branches(self):
        sql = self.t("SELECT shot FROM a UNION SELECT shot FROM b")
        self.assertEqual(sql, "SELECT shot AS shot FROM a UNION SELECT shot AS shot FROM b")


class TestRewrite(unittest.TestCase):
    @unittest.skipUnless(HAVE_SQLGLOT, "sqlglot not installed (conda-forge: sqlglot)")
    def test_both_passes_in_order(self):
        sql, style, note = _tsql.rewrite(
            "SELECT TOP 5 shot FROM shots WHERE run LIKE %s", nocase=True)
        self.assertEqual(sql, "SELECT shot AS shot FROM shots WHERE run ILIKE ? LIMIT 5")
        self.assertEqual(style, "qmark")
        self.assertIsNone(note)


if __name__ == "__main__":
    unittest.main()
