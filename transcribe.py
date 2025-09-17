#!/usr/bin/env python3
import sys
import whisper
import datetime
import json
import os

# Configuration constants
WHISPER_MODEL = "base"
OUTPUT_DIR = "outputs"
LANGUAGE = "Spanish"


def load_whisper_model(model_name):
    """Load Whisper model with error handling."""
    try:
        model = whisper.load_model(model_name)
        print(f"Loaded Whisper model: {model_name}")
        return model
    except Exception as e:
        print(f"Error loading Whisper model '{model_name}': {e}")
        sys.exit(1)


def transcribe_audio(model, audio_file, language):
    """Transcribe audio file and return result."""
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found")
        sys.exit(1)

    try:
        print(f"Transcribing audio file: {audio_file}")
        result = model.transcribe(audio_file, verbose=True, language=language)
        return result
    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)


def save_transcription_files(result, base_filename, timestamp):
    """Save transcription in both text and JSONL formats."""
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save text file
    txt_filename = f"{OUTPUT_DIR}/{base_filename}_{timestamp}.txt"
    try:
        with open(txt_filename, "w") as f:
            f.write(result["text"])
        print(f"Transcription saved to: {txt_filename}")
    except Exception as e:
        print(f"Error saving text file: {e}")

    # Save JSONL file with segments
    jsonl_filename = f"{OUTPUT_DIR}/{base_filename}_{timestamp}.jsonl"
    try:
        with open(jsonl_filename, "w") as f:
            for segment in result["segments"]:
                segment_data = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"]
                }
                f.write(json.dumps(segment_data) + "\n")
        print(f"Segments saved to: {jsonl_filename}")
    except Exception as e:
        print(f"Error saving JSONL file: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python transcribe.py <audio_file>")
        sys.exit(1)

    audio_file = sys.argv[1]

    # Load Whisper model
    model = load_whisper_model(WHISPER_MODEL)

    # Transcribe audio
    result = transcribe_audio(model, audio_file, LANGUAGE)
    print(result["text"])

    # Prepare output filenames
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    base_filename = os.path.splitext(os.path.basename(audio_file))[0]

    # Save transcription files
    save_transcription_files(result, base_filename, timestamp)