import unittest
from src.lab1_ir import (
    InvertedIndex, compress_dictionary_block_front, gap_decode, gap_encode,
    intersect, intersect_with_skips, vb_decode, vb_encode,
)


class CoreTests(unittest.TestCase):
    def test_vb_gap_roundtrip(self):
        ids = [3, 5, 130, 131, 9999]
        self.assertEqual(ids, gap_decode(vb_decode(vb_encode(gap_encode(ids)))))

    def test_skip_intersection_matches_plain(self):
        a, b = [1, 3, 5, 8, 13, 21, 34], [2, 3, 8, 20, 21, 55]
        plain, _ = intersect(a, b)
        skipped, _ = intersect_with_skips(a, b, 2, 2)
        self.assertEqual(plain, skipped)

    def test_phrase_positions(self):
        idx = InvertedIndex(); idx.add(1, ["中国", "科学", "技术", "大学"]); idx.add(2, ["中国", "大学", "科学"])
        idx.finalize()
        self.assertEqual([1], idx.phrase(["科学", "技术", "大学"]))

    def test_dictionary_compression_deterministic(self):
        self.assertEqual(compress_dictionary_block_front(["apple", "apply"]), compress_dictionary_block_front(["apply", "apple"]))


if __name__ == "__main__":
    unittest.main()
