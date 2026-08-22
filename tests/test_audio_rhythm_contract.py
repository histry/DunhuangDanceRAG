import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scheduling import audio_features
from scheduling import music_phrase_segmentation


def fake_librosa(*, tempo_bpm: float = 0.0, beat_frames=(), beat_error=None):
    frames = 8
    signal = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)

    def beat_track(**_kwargs):
        if beat_error is not None:
            raise beat_error
        return np.asarray([tempo_bpm], dtype=np.float32), np.asarray(
            beat_frames, dtype=int
        )

    return SimpleNamespace(
        load=lambda *_args, **_kwargs: (signal, 16000),
        feature=SimpleNamespace(
            rms=lambda **_kwargs: np.ones((1, frames), dtype=np.float32),
            spectral_centroid=lambda **_kwargs: np.linspace(
                100.0, 1000.0, frames, dtype=np.float32
            ).reshape(1, -1),
            chroma_stft=lambda **_kwargs: np.tile(
                np.linspace(0.0, 1.0, frames, dtype=np.float32), (12, 1)
            ),
        ),
        onset=SimpleNamespace(
            onset_strength=lambda **_kwargs: np.linspace(
                0.0, 1.0, frames, dtype=np.float32
            )
        ),
        beat=SimpleNamespace(beat_track=beat_track),
    )


class AudioRhythmContractTests(unittest.TestCase):
    def test_required_librosa_backend_never_enters_wave_fallback(self):
        broken = SimpleNamespace(
            __version__="test",
            load=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                RuntimeError("synthetic librosa failure")
            ),
        )
        with patch.dict(sys.modules, {"librosa": broken}):
            with patch.object(audio_features, "_fallback_features") as fallback:
                with self.assertRaisesRegex(
                    audio_features.AudioFeatureBackendError,
                    "wave_fallback is forbidden",
                ):
                    audio_features.extract_audio_features(
                        "synthetic.wav",
                        num_frames=8,
                        require_librosa=True,
                    )
                fallback.assert_not_called()

    def test_beat_exception_is_logged_and_recorded_in_metadata(self):
        fake = fake_librosa(beat_error=ValueError("synthetic beat failure"))
        with patch.dict(sys.modules, {"librosa": fake}):
            with self.assertLogs(audio_features.LOGGER, level="ERROR") as captured:
                features, meta = audio_features.extract_audio_features(
                    "synthetic.wav", num_frames=8
                )

        self.assertTrue(any("Beat extraction failed" in line for line in captured.output))
        self.assertEqual(features.shape, (8, 12))
        self.assertEqual(meta["backend"], "librosa")
        self.assertEqual(meta["backend_version"], "unknown")
        self.assertFalse(meta["beat_tracking"]["ok"])
        self.assertEqual(meta["beat_tracking"]["error_type"], "ValueError")
        self.assertEqual(
            meta["beat_tracking"]["error_message"], "synthetic beat failure"
        )
        self.assertFalse(meta["rhythm_contract"]["ok"])

    def test_strict_mode_raises_with_original_beat_exception(self):
        fake = fake_librosa(beat_error=ValueError("synthetic beat failure"))
        with patch.dict(sys.modules, {"librosa": fake}):
            with self.assertRaisesRegex(
                audio_features.RhythmFeatureError,
                "ValueError: synthetic beat failure",
            ):
                audio_features.extract_audio_features(
                    "synthetic.wav",
                    num_frames=8,
                    require_rhythm=True,
                )

    def test_strict_mode_allows_zero_tempo_when_beats_exist(self):
        fake = fake_librosa(tempo_bpm=0.0, beat_frames=(1, 4))
        with patch.dict(sys.modules, {"librosa": fake}):
            features, meta = audio_features.extract_audio_features(
                "synthetic.wav",
                num_frames=8,
                require_rhythm=True,
            )

        self.assertGreater(int(np.count_nonzero(features[:, 2])), 0)
        self.assertTrue(meta["rhythm_contract"]["ok"])

    def test_strict_mode_allows_tempo_when_beats_are_zero(self):
        fake = fake_librosa(tempo_bpm=120.0, beat_frames=())
        with patch.dict(sys.modules, {"librosa": fake}):
            _, meta = audio_features.extract_audio_features(
                "synthetic.wav",
                num_frames=8,
                require_rhythm=True,
            )

        self.assertTrue(meta["rhythm_contract"]["beat_zero"])
        self.assertFalse(meta["rhythm_contract"]["tempo_zero"])
        self.assertTrue(meta["rhythm_contract"]["ok"])

    def test_strict_mode_rejects_zero_rhythm_from_existing_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            audio_path = cache / "cached_song.wav"
            audio_path.write_bytes(b"formal-cache-content")
            digest = music_phrase_segmentation.audio_content_sha256(audio_path)
            cache_path = (
                cache
                / f"cached_song_{digest[:16]}_whole_song_fps2_2.npy"
            )
            np.save(cache_path, np.zeros((2, 12), dtype=np.float32))
            cache_path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "cache_schema": "music_12d_content_addressed_cache_v2",
                        "audio": str(audio_path),
                        "audio_sha256": digest,
                        "extractor": {
                            "backend": "librosa",
                            "backend_version": "test",
                            "tempo_bpm": 0.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with patch.object(
                music_phrase_segmentation,
                "audio_duration_seconds",
                return_value=1.0,
            ):
                with self.assertRaisesRegex(
                    audio_features.RhythmFeatureError,
                    "beat pulse and tempo are both zero",
                ):
                    music_phrase_segmentation.whole_song_features(
                        audio_path,
                        fps=2.0,
                        cache_dir=cache,
                        require_rhythm=True,
                    )

    def test_whole_song_cache_is_content_addressed_for_same_stem(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a" / "song.wav"
            second = root / "b" / "song.wav"
            first.parent.mkdir()
            second.parent.mkdir()
            first.write_bytes(b"first-audio-content")
            second.write_bytes(b"second-audio-content")
            cache = root / "cache"

            def synthetic_features(path, num_frames, **_kwargs):
                value = 0.25 if Path(path) == first else 0.75
                return (
                    np.full((num_frames, 12), value, dtype=np.float32),
                    {
                        "backend": "librosa",
                        "backend_version": "test",
                        "tempo_bpm": 120.0,
                    },
                )

            with patch.object(
                music_phrase_segmentation,
                "audio_duration_seconds",
                return_value=1.0,
            ), patch.object(
                music_phrase_segmentation,
                "extract_audio_features",
                side_effect=synthetic_features,
            ):
                first_features, first_meta = (
                    music_phrase_segmentation.whole_song_features(
                        first, fps=2.0, cache_dir=cache
                    )
                )
                second_features, second_meta = (
                    music_phrase_segmentation.whole_song_features(
                        second, fps=2.0, cache_dir=cache
                    )
                )

            self.assertNotEqual(
                first_meta["audio_sha256"], second_meta["audio_sha256"]
            )
            self.assertFalse(np.array_equal(first_features, second_features))
            self.assertEqual(len(list(cache.glob("*.npy"))), 2)


if __name__ == "__main__":
    unittest.main()
