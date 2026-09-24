import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import numpy as np
import soundfile as sf

from trainer.audio_output import save_generated_audio


class TestGeneratedAudioOutput(unittest.TestCase):
    def test_keeps_wav_when_mp3_encoder_is_unavailable(self):
        encoder = Mock()
        encoder.export.side_effect = FileNotFoundError("ffmpeg missing")
        pydub = ModuleType("pydub")
        pydub.AudioSegment = Mock(from_wav=Mock(return_value=encoder))
        audio = np.zeros(32, dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"pydub": pydub}):
            mp3_path = Path(directory) / "answer.mp3"
            saved_path, error = save_generated_audio(audio, mp3_path)

            self.assertEqual(saved_path, str(mp3_path.with_suffix(".wav")))
            self.assertIsInstance(error, FileNotFoundError)
            self.assertTrue(Path(saved_path).is_file())
            self.assertFalse(mp3_path.exists())
            decoded, sample_rate = sf.read(saved_path)
            self.assertEqual(sample_rate, 24000)
            self.assertEqual(len(decoded), len(audio))

    def test_removes_temporary_wav_after_successful_mp3_export(self):
        encoder = Mock()
        encoder.export.side_effect = lambda path, **_: Path(path).with_suffix(".mp3").write_bytes(b"mp3")
        pydub = ModuleType("pydub")
        pydub.AudioSegment = Mock(from_wav=Mock(return_value=encoder))

        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"pydub": pydub}):
            mp3_path = Path(directory) / "answer.mp3"
            saved_path, error = save_generated_audio(np.zeros(8, dtype=np.float32), mp3_path)

            self.assertEqual(saved_path, str(mp3_path))
            self.assertIsNone(error)
            self.assertEqual(mp3_path.read_bytes(), b"mp3")
            self.assertFalse(mp3_path.with_suffix(".wav").exists())

    def test_wav_output_needs_no_mp3_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            wav_path = Path(directory) / "answer.wav"
            saved_path, error = save_generated_audio(np.zeros(4, dtype=np.float32), wav_path)

            self.assertEqual(saved_path, str(wav_path))
            self.assertIsNone(error)
            self.assertTrue(wav_path.is_file())

    def test_rejects_unsupported_output_extension(self):
        with self.assertRaisesRegex(ValueError, "must use .wav or .mp3"):
            save_generated_audio(np.zeros(4, dtype=np.float32), "answer.ogg")


if __name__ == "__main__":
    unittest.main()
