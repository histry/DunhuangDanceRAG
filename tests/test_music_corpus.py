from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from training.music_corpus import discover_training_audio


class MusicCorpusTests(unittest.TestCase):
    def test_content_duplicates_are_removed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "train"
            root.mkdir()
            (root / "a.wav").write_bytes(b"same-audio")
            (root / "b.wav").write_bytes(b"same-audio")
            (root / "c.wav").write_bytes(b"different-audio")
            paths = discover_training_audio([root])
            self.assertEqual([path.name for path in paths], ["a.wav", "c.wav"])

    def test_nested_test_music_is_rejected_from_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "train"
            test_root = root / "test_music_bank"
            test_root.mkdir(parents=True)
            (root / "safe.wav").write_bytes(b"train")
            (test_root / "held_out.wav").write_bytes(b"test")
            paths = discover_training_audio([root])
            self.assertEqual([path.name for path in paths], ["safe.wav"])


if __name__ == "__main__":
    unittest.main()
