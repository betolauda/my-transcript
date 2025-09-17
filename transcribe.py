#!/usr/bin/env python3
import sys
import whisper
import datetime
import json
import os

if len(sys.argv) != 2:
    print("Usage: python transcribe.py <audio_file>")
    sys.exit(1)

audio_file = sys.argv[1]
model = whisper.load_model("base")
result = model.transcribe(audio_file, verbose=True, language="Spanish")
print(result["text"])

# Create outputs directory if it doesn't exist
os.makedirs("outputs", exist_ok=True)

# Save transcription as text file with timestamp and audio filename
#timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
timestamp = datetime.datetime.now().strftime("%Y%m%d")
base_name = os.path.splitext(os.path.basename(audio_file))[0]
txt_filename = f"outputs/{base_name}_{timestamp}.txt"
with open(txt_filename, "w") as f:
    f.write(result["text"])
print(f"Transcription saved to: {txt_filename}")

# Save as JSONL with segment data
jsonl_filename = f"outputs/{base_name}_{timestamp}.jsonl"
with open(jsonl_filename, "w") as f:
    for segment in result["segments"]:
        segment_data = {
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        }
        f.write(json.dumps(segment_data) + "\n")
print(f"Segments saved to: {jsonl_filename}")