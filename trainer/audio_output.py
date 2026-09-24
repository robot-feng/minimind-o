"""Save decoded model audio with a WAV fallback when MP3 encoding is unavailable."""

import os
import warnings

import soundfile as sf


def save_generated_audio(audio, output_path, sample_rate=24000):
    output_path = os.fspath(output_path)
    output_format = os.path.splitext(output_path)[1].lower().lstrip(".")
    if output_format == "wav":
        sf.write(output_path, audio, sample_rate)
        return output_path, None
    if output_format != "mp3":
        raise ValueError("generated audio output must use .wav or .mp3")

    wav_path = os.path.splitext(output_path)[0] + ".wav"
    sf.write(wav_path, audio, sample_rate)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Couldn't find ffmpeg or avconv.*", category=RuntimeWarning
            )
            from pydub import AudioSegment
        AudioSegment.from_wav(wav_path).export(output_path, format="mp3", bitrate="64k")
    except Exception as error:
        return wav_path, error
    os.remove(wav_path)
    return output_path, None
