

import json
from faster_whisper import WhisperModel




def transcribe_audio(audio_path):

    print("Loading Whisper model...")

    model = WhisperModel(
        "base",
        device="cpu",
        compute_type="int8"
    )

    print("Starting transcription...")

    segments, info = model.transcribe(
        audio_path,
        beam_size=5
    )

    print(f"Detected language: {info.language}")

    print("\n--- TRANSCRIPT ---")

    transcript = []

    for segment in segments:

        print(
            f"[{segment.start:.2f}s -> "
            f"{segment.end:.2f}s] "
            f"{segment.text}"
        )

        transcript.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    print("\n--- END ---")
    with open("transcript.json", "w", encoding="utf-8") as file:
        json.dump(transcript, file, indent=4)

    return transcript


if __name__ == "__main__":

    audio_file = "sample.mp3"

    transcribe_audio(audio_file)