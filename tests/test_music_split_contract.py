#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from data_pipeline.split_music_corpus import split_corpus


class MusicSplitContractTests(unittest.TestCase):
    def test_duplicate_audio_hash_is_removed_before_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files = [root / f"song_{index}.wav" for index in range(5)]
            for path in files:
                path.write_bytes(path.name.encode("utf-8"))
            digest = {
                str(files[0]): "a" * 64,
                str(files[1]): "a" * 64,
                str(files[2]): "b" * 64,
                str(files[3]): "c" * 64,
                str(files[4]): "d" * 64,
            }
            with patch(
                "data_pipeline.split_music_corpus.discover_training_audio",
                return_value=[str(path) for path in files],
            ), patch(
                "data_pipeline.split_music_corpus.audio_sha256",
                side_effect=lambda path: digest[str(Path(path))],
            ):
                report = split_corpus([root], root / "split", seed=7)
            self.assertEqual(report["num_songs"], 4)
            self.assertEqual(report["duplicate_audio_files_removed"], 1)
            self.assertTrue(report["song_disjoint"])
            self.assertGreater(report["splits"]["test"]["num_songs"], 0)


if __name__ == "__main__":
    unittest.main()
