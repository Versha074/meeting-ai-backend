import json
from ollama import chat


def load_transcript():

    with open("transcript.json", "r", encoding="utf-8") as file:
        transcript = json.load(file)

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

Read the meeting transcript below and extract the REAL information from it.

Return ONLY valid JSON.

The JSON must have exactly these fields:

{{
    "summary": "Write a real 2-3 sentence summary based on the transcript.",
    "action_items": [
        {{
            "task": "Actual task mentioned in the meeting",
            "responsible": null,
            "deadline": null
        }}
    ],
    "decisions": [
        "Actual decision made during the meeting"
    ]
}}

Rules:
- Write the summary based on the actual transcript.
- Do NOT write "short meeting summary".
- Extract only tasks actually mentioned or clearly agreed upon.
- If the responsible person is not mentioned, use null.
- If a deadline is not mentioned, use null.
- Extract actual decisions from the meeting.
- Return ONLY JSON.
- Do not use markdown.
- Do not use ```json.
- Do not add any explanation outside the JSON.

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


if __name__ == "__main__":

    transcript = load_transcript()

    meeting_text = create_meeting_text(transcript)

    result = analyze_meeting(meeting_text)

    print("\n--- AI MEETING ANALYSIS ---")

    try:

        data = json.loads(result)

        print(json.dumps(data, indent=4))

    except json.JSONDecodeError:

        print("ERROR: Llama did not return valid JSON.")
        print(result)