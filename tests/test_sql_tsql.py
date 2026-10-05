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
        self.assertEqual(sql, "SELECT shot FROM shots ORDER BY shot DESC LIMIT 50")
        self.assertIsNone(note)

    def test_isnull_getdate_datediff(self):
        sql, _ = _tsql.transpile(
            "SELECT ISNULL(betanmax, 0) b, DATEDIFF(day, entered, GETDATE()) age FROM summaries")
        self.assertIn("COALESCE(betanmax, 0)", sql)
        self.assertIn("DATE_DIFF('DAY'", sql)
        self.assertIn("CURRENT_TIMESTAMP", sql)

    def test_brackets_and_dbo(self):
        sql, _ = _tsql.transpile("SELECT [shot], [time] FROM dbo.disruption_warning")
        self.assertEqual(sql, 'SELECT "shot", "time" FROM dbo.disruption_warning')

    def test_placeholders_survive(self):
        sql, _ = _tsql.transpile("SELECT shot FROM shots WHERE shot = ? AND run = $run")
        self.assertEqual(sql, "SELECT shot FROM shots WHERE shot = ? AND run = $run")

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

    def test_two_statements(self):
        sql, _ = _tsql.transpile("SELECT TOP 1 a FROM t; SELECT TOP 2 b FROM u")
        self.assertEqual(sql, "SELECT a FROM t LIMIT 1; SELECT b FROM u LIMIT 2")


class TestRewrite(unittest.TestCase):
    @unittest.skipUnless(HAVE_SQLGLOT, "sqlglot not installed (conda-forge: sqlglot)")
    def test_both_passes_in_order(self):
        sql, style, note = _tsql.rewrite(
            "SELECT TOP 5 shot FROM shots WHERE run LIKE %s", nocase=True)
        self.assertEqual(sql, "SELECT shot FROM shots WHERE run ILIKE ? LIMIT 5")
        self.assertEqual(style, "qmark")
        self.assertIsNone(note)


if __name__ == "__main__":
    unittest.main()
