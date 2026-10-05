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


if __name__ == "__main__":
    unittest.main()
