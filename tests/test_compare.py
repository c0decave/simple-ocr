"""Tests for the result comparison used by the benchmark page."""
import unittest

from app import compare

INVOICE = "Rechnung Nr. 2026-0417\nMüller & Söhne GmbH\nDatum: 13.10.2026\nSumme brutto: 68,85 €"


class Normalize(unittest.TestCase):
    def test_whitespace_and_line_breaks_do_not_matter(self):
        self.assertEqual(compare.normalize("a  b\n\n c \t"), "a b c")

    def test_markdown_table_scaffolding_is_ignored(self):
        table = "Pos | Preis\n----|------\n1 | 2,00"
        self.assertEqual(compare.normalize(table), compare.normalize("Pos Preis\n1 2,00"))

    def test_unicode_forms_are_unified(self):
        self.assertEqual(compare.normalize("Müller"), compare.normalize("Müller"))

    def test_case_is_kept(self):
        self.assertNotEqual(compare.normalize("ABC"), compare.normalize("abc"))


class Similarity(unittest.TestCase):
    def test_same_text_is_identical(self):
        self.assertEqual(compare.similarity(INVOICE, INVOICE), 1.0)
        self.assertEqual(compare.classify(1.0), "identical")

    def test_reformatting_only_is_identical(self):
        reflowed = INVOICE.replace("\n", "\n\n").replace(" ", "  ")
        self.assertEqual(compare.similarity(INVOICE, reflowed), 1.0)

    def test_one_wrong_word_is_similar(self):
        wrong = INVOICE.replace("Müller", "Miller")
        ratio = compare.similarity(INVOICE, wrong)
        self.assertLess(ratio, 1.0)
        self.assertEqual(compare.classify(ratio), "similar")

    def test_unrelated_text_is_different(self):
        ratio = compare.similarity(INVOICE, "A cat sits on a red sofa next to a window with curtains.")
        self.assertEqual(compare.classify(ratio), "different")

    def test_empty_against_text_is_zero_and_both_empty_is_identical(self):
        self.assertEqual(compare.similarity("", INVOICE), 0.0)
        self.assertEqual(compare.similarity("  ", "\n"), 1.0)

    def test_similarity_is_symmetric(self):
        a, b = INVOICE, INVOICE.replace("Datum", "Date").replace("Summe", "Total")
        self.assertAlmostEqual(compare.similarity(a, b), compare.similarity(b, a), places=9)

    def test_very_long_input_is_capped_not_slow(self):
        import time
        big = ("word%d " * 1) * 0 + " ".join("w%d" % i for i in range(60000))
        started = time.monotonic()
        ratio = compare.similarity(big, big + " extra")
        self.assertLess(time.monotonic() - started, 5)
        self.assertGreater(ratio, 0.9)


class Grouping(unittest.TestCase):
    def test_models_are_grouped_by_similarity_to_the_group_representative(self):
        texts = {
            "m1": INVOICE,
            "m2": INVOICE,                                   # identical to m1
            "m3": INVOICE.replace("Müller", "Miller"),       # similar to m1
            "m4": "A cat sits on a sofa and looks out of the window.",   # different
            "m5": "A cat sits on a sofa and looks out of the window!",   # similar to m4
        }
        analysis = compare.analyze(texts)
        groups = [g["models"] for g in analysis["groups"]]
        self.assertEqual(groups, [["m1", "m2", "m3"], ["m4", "m5"]])
        self.assertEqual([g["label"] for g in analysis["groups"]], ["A", "B"])
        self.assertEqual(analysis["groups"][0]["identical_to_representative"], ["m2"])
        self.assertEqual(analysis["matrix"]["models"], ["m1", "m2", "m3", "m4", "m5"])

    def test_matrix_is_symmetric_with_ones_on_the_diagonal(self):
        analysis = compare.analyze({"a": INVOICE, "b": INVOICE + " extra", "c": "something else entirely"})
        values = analysis["matrix"]["values"]
        for i in range(3):
            self.assertEqual(values[i][i], 1.0)
            for j in range(3):
                self.assertEqual(values[i][j], values[j][i])

    def test_pairs_carry_a_class(self):
        analysis = compare.analyze({"a": INVOICE, "b": INVOICE})
        self.assertEqual(analysis["pairs"], [{"a": "a", "b": "b", "similarity": 1.0, "class": "identical"}])

    def test_single_model_and_no_models(self):
        self.assertEqual(compare.analyze({"only": "x"})["groups"][0]["models"], ["only"])
        self.assertEqual(compare.analyze({})["groups"], [])

    def test_threshold_is_part_of_the_answer(self):
        self.assertEqual(compare.analyze({})["similar_threshold"], compare.SIMILAR_THRESHOLD)


def leaves(node):
    return [node["model"]] if "model" in node else [m for child in node["children"] for m in leaves(child)]


class TreeAndAgreement(unittest.TestCase):
    """Average-linkage tree over 1 - similarity, for the sideways dendrogram, and per-model agreement."""

    TEXTS = {
        "m1": INVOICE,
        "m4": "A cat sits on a sofa and looks out of the window.",
        "m2": INVOICE,
        "m5": "A cat sits on a sofa and looks out of the window!",
        "m3": INVOICE.replace("Müller", "Miller"),
    }

    def test_closest_models_merge_first_and_far_groups_last(self):
        tree = compare.analyze(self.TEXTS)["tree"]
        self.assertEqual(sorted(leaves(tree)), ["m1", "m2", "m3", "m4", "m5"])
        invoice, cat = tree["children"]
        if "m4" in leaves(invoice):
            invoice, cat = cat, invoice
        self.assertEqual(sorted(leaves(invoice)), ["m1", "m2", "m3"])
        self.assertEqual(sorted(leaves(cat)), ["m4", "m5"])
        identical = [c for c in invoice["children"] if "children" in c][0]   # m1 + m2 merge at distance 0
        self.assertEqual((sorted(leaves(identical)), identical["height"]), (["m1", "m2"], 0.0))
        self.assertGreater(tree["height"], invoice["height"])
        self.assertGreater(tree["height"], cat["height"])

    def test_heights_never_decrease_towards_the_root(self):
        def check(node):
            for child in node.get("children", []):
                self.assertLessEqual(child.get("height", 0.0), node["height"])
                check(child)
        check(compare.analyze(self.TEXTS)["tree"])

    def test_tree_is_deterministic_for_ties(self):
        texts = {"a": "x", "b": "x", "c": "x"}
        self.assertEqual(compare.analyze(texts)["tree"], compare.analyze(dict(texts))["tree"])
        self.assertEqual(leaves(compare.analyze(texts)["tree"]), ["a", "b", "c"])

    def test_agreement_is_the_mean_similarity_to_all_others(self):
        analysis = compare.analyze({"a": INVOICE, "b": INVOICE, "c": "something else entirely"})
        values = analysis["matrix"]["values"]
        self.assertAlmostEqual(analysis["agreement"]["a"], round((values[0][1] + values[0][2]) / 2, 4))
        self.assertGreater(analysis["agreement"]["a"], analysis["agreement"]["c"])

    def test_single_model_and_no_models(self):
        one = compare.analyze({"only": "x"})
        self.assertEqual(one["tree"], {"model": "only"})
        self.assertEqual(one["agreement"], {"only": None})
        none = compare.analyze({})
        self.assertEqual((none["tree"], none["agreement"]), (None, {}))


class DiffOps(unittest.TestCase):
    def test_equal_texts_have_only_equal_ops(self):
        ops = compare.diff("a\nb", "a\nb")
        self.assertEqual([op["tag"] for op in ops], ["equal"])

    def test_changed_word_is_marked_inside_the_line(self):
        ops = compare.diff("Müller & Söhne GmbH", "Miller & Söhne GmbH")
        self.assertEqual([op["tag"] for op in ops], ["replace"])
        a_line, b_line = ops[0]["a"][0], ops[0]["b"][0]
        self.assertIn(["Müller", True], a_line)
        self.assertIn(["Miller", True], b_line)
        self.assertIn(["Söhne", False], a_line)

    def test_inserted_and_deleted_lines(self):
        ops = compare.diff("one\ntwo\nthree", "one\nthree\nfour")
        self.assertEqual([op["tag"] for op in ops], ["equal", "delete", "equal", "insert"])
        self.assertEqual(ops[1]["a"], [[["two", True]]])
        self.assertEqual(ops[3]["b"], [[["four", True]]])

    def test_different_line_counts_in_a_replace_block_stay_plain(self):
        ops = compare.diff("x\ny", "z")
        self.assertEqual([op["tag"] for op in ops], ["replace"])
        self.assertEqual(ops[0]["a"], [[["x", True]], [["y", True]]])
        self.assertEqual(ops[0]["b"], [[["z", True]]])

    def test_empty_inputs(self):
        self.assertEqual(compare.diff("", ""), [])


if __name__ == "__main__":
    unittest.main()


class Bounds(unittest.TestCase):
    def test_similarity_does_not_depend_on_the_order(self):
        a, b = "a a c", "a b c a b c a"
        self.assertEqual(compare.similarity(a, b), compare.similarity(b, a))

    def test_a_huge_single_line_diff_stays_fast(self):
        import time
        words = ["the"] * 8000
        changed = list(words)
        changed[4000] = "THE"
        started = time.monotonic()
        ops = compare.diff(" ".join(words), " ".join(changed))
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertEqual(ops[0]["tag"], "replace")
        self.assertTrue(ops[0]["a"][0][0][1], "a line too long for a word diff is marked changed as a whole")

    def test_a_diff_with_very_many_lines_stays_fast_and_says_it_is_cut(self):
        import time
        a = "\n".join("line %d" % (i % 7) for i in range(20000))
        b = a.replace("line 3", "line 9", 1)
        started = time.monotonic()
        ops = compare.diff(a, b)
        self.assertLess(time.monotonic() - started, 2.0)
        self.assertEqual(ops[-1]["tag"], "truncated")
        self.assertEqual(sum(len(op["a"]) for op in ops if op["tag"] != "truncated"), compare.MAX_TOKENS)

    def test_short_texts_are_diffed_word_by_word_as_before(self):
        ops = compare.diff("Rechnung Nr. 1", "Rechnung Nr. 2")
        self.assertEqual(ops, [{"tag": "replace", "a": [[["Rechnung", False], [" ", False], ["Nr.", False], [" ", False], ["1", True]]],
                                "b": [[["Rechnung", False], [" ", False], ["Nr.", False], [" ", False], ["2", True]]]}])
