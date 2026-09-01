import unittest

from toksearch.provenance.hashing import canonical_json, sha256_of, callable_spec


def _example_func(x):
    return x + 1


class TestCanonicalJson(unittest.TestCase):
    def test_key_order_does_not_matter(self):
        a = {"b": 1, "a": 2}
        b = {"a": 2, "b": 1}
        self.assertEqual(canonical_json(a), canonical_json(b))

    def test_no_incidental_whitespace(self):
        self.assertEqual(canonical_json({"a": 1}), '{"a":1}')

    def test_non_serializable_falls_back_to_repr_string(self):
        out = canonical_json({"a": object()})
        self.assertIn("object object at", out)

    def test_nested_dicts_are_also_sorted(self):
        a = {"outer": {"z": 1, "y": 2}}
        b = {"outer": {"y": 2, "z": 1}}
        self.assertEqual(canonical_json(a), canonical_json(b))


class TestSha256Of(unittest.TestCase):
    def test_is_stable_across_calls(self):
        self.assertEqual(sha256_of({"a": 1}), sha256_of({"a": 1}))

    def test_differs_for_different_values(self):
        self.assertNotEqual(sha256_of({"a": 1}), sha256_of({"a": 2}))

    def test_is_a_64_char_hex_digest(self):
        digest = sha256_of({"a": 1})
        self.assertEqual(len(digest), 64)
        int(digest, 16)


class TestCallableSpec(unittest.TestCase):
    def test_none_returns_none(self):
        self.assertIsNone(callable_spec(None))

    def test_captures_name_and_module(self):
        spec = callable_spec(_example_func)
        self.assertEqual(spec["name"], "_example_func")
        self.assertEqual(spec["module"], __name__)

    def test_captures_source_hash(self):
        spec = callable_spec(_example_func)
        self.assertEqual(len(spec["source_sha256"]), 64)

    def test_source_hash_is_none_when_source_unavailable(self):
        spec = callable_spec(eval("lambda x: x"))
        self.assertIn("source_sha256", spec)

    def test_builtin_does_not_raise(self):
        spec = callable_spec(len)
        self.assertEqual(spec["name"], "len")
