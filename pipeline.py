import json
from faster_whisper import WhisperModel
from ollama import chat


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

    transcript = []

    for segment in segments:

        transcript.append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })

    print(f"Detected language: {info.language}")

    return transcript


def create_meeting_text(transcript):

    meeting_text = ""

    for segment in transcript:

        meeting_text += (
            f"[{segment['start']:.2f}s - "
            f"{segment['end']:.2f}s] "
            f"{segment['text']}\n"
        )

    return meeting_text


def analyze_meeting(meeting_text):

    prompt = f"""
You are an AI meeting assistant.

Read the meeting transcript below and extract the REAL information.

Return ONLY valid JSON in exactly this format:

{{
    "summary": "Actual 2-3 sentence summary of the meeting",
    "action_items": [
        {{
            "task": "Actual task mentioned",
            "responsible": null,
            "deadline": null
        }}
    ],
    "decisions": [
        "Actual decision made"
    ]
}}

Rules:
- Use information from the actual transcript.
- Do not write "short meeting summary".
- Do not invent information.
- If the responsible person is not mentioned, use null.
- If a deadline is not mentioned, use null.
- Return only JSON.
- Do not use markdown.
- Do not use ```json.
- Do not add explanations outside the JSON.

Meeting Transcript:

{meeting_text}
"""

    response = chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def save_transcript(transcript):

    with open("transcript.json", "w", encoding="utf-8") as file:

        json.dump(
            transcript,
            file,
            indent=4
        )


if __name__ == "__main__":

    audio_file = "sample.mp3"

    # Step 1: Audio → Transcript
    transcript = transcribe_audio(audio_file)

    # Save transcript
    save_transcript(transcript)

    # Step 2: Transcript → Text
    meeting_text = create_meeting_text(transcript)

    # Step 3: Text → Llama
    result = analyze_meeting(meeting_text)

    print("\n--- AI MEETING ANALYSIS ---")

    try:

        data = json.loads(result)

        print(json.dumps(data, indent=4))

    except json.JSONDecodeError:

        print("ERROR: Llama did not return valid JSON.")
        print(result)