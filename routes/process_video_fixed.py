# -*- coding: utf-8 -*-
"""Updated process_video.py for dynamic file paths.

This version uses a process_video_file(file_path) function which builds the analysis
and compliance report file names based on the uploaded file's name.
"""

import os
import json
import base64
import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
import whisper
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import io
import re

# Load environment variables and API key

load_dotenv()
#os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


##############################
# Video Analysis Functions
##############################

def extract_audio_from_video(file_path):
    """Extract audio from a video file and return the temporary audio file path."""
    audio_path = "temp_audio.wav"
    video = VideoFileClip(file_path)
    video.audio.write_audiofile(audio_path)
    return audio_path

def transcribe_audio_with_timestamps(audio_path, model_name="large"):
    """Transcribe audio with timestamps using Whisper."""
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, verbose=True)
    return [{"start": seg["start"], "end": seg["end"], "text": seg["text"]} for seg in result["segments"]]

def compress_resize_encode_image(file_path, max_size=(1024, 1024), quality=85):
    """Compress, resize, and encode an image as a base64 string."""
    try:
        with Image.open(file_path) as img:
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.thumbnail(max_size)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", optimize=True, quality=quality)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error processing image {file_path}: {str(e)}")
        return None

def analyze_image(file_path, transcript_text=""):
    """Analyzes the image using OpenAI API and returns the output."""
    try:
        transcript_text = transcript_text or "No transcription available for this frame."
        base64_image = compress_resize_encode_image(file_path)
        if not base64_image:
            print(f"Skipping frame {file_path} due to encoding error.")
            return None

        response = client.chat_completions.create(
            model="gpt-4",
            messages=[
                {"role": "user", "content": f"Analyze this image: {base64_image}. Transcription: {transcript_text}"}
            ],
        )
        return response.choices[0].message["content"].strip()
    except Exception as e:
        print(f"Error analyzing image {file_path}: {str(e)}")
        return None

def is_frame_blurry(frame, threshold=100):
    """Determines if a frame is blurry based on the variance of the Laplacian."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    return variance < threshold

def match_transcription_to_frame(transcriptions, timestamp):
    """Match the appropriate transcription text for a given timestamp."""
    for transcript in transcriptions:
        if transcript["start"] <= timestamp <= transcript["end"]:
            return transcript["text"]
    return "No transcription available for this frame."

def extract_frames_at_intervals(video_path, interval, transcriptions, blur_threshold=100):
    """
    Extracts frames from the video at regular intervals, matches transcription,
    and excludes blurry frames.
    Returns a list of analyzed frames.
    """
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = frame_count / fps

    results = []
    frame_dir = "frames"
    os.makedirs(frame_dir, exist_ok=True)

    for timestamp in range(0, int(total_duration), interval):
        frame_index = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        if not success:
            continue

        if is_frame_blurry(frame, threshold=blur_threshold):
            print(f"Skipping blurry frame at index {frame_index} (timestamp: {timestamp:.2f}s)")
            continue

        frame_path = os.path.join(frame_dir, f"frame_{frame_index}.jpg")
        cv2.imwrite(frame_path, frame)

        transcription_text = match_transcription_to_frame(transcriptions, timestamp)
        analysis = analyze_image(frame_path, transcription_text)
        if analysis:
            results.append({
                "frame_index": frame_index,
                "timestamp": timestamp,
                "transcription": transcription_text,
                "analysis": analysis,
            })

    cap.release()
    return results

def combine_frame_analyses(frame_analyses):
    """
    Combines individual frame analyses into a comprehensive video narrative.
    Aggregates details from each frame and queries an LLM to generate a flowing story.
    """
    combined_details = "The following details are gathered from various points in the video:\n"
    for frame in frame_analyses:
        combined_details += f"\nAt {frame['timestamp']:.2f} seconds:\n"
        if frame.get("transcription"):
            combined_details += f"Transcription: {frame['transcription']}\n"
        combined_details += f"Analysis: {frame['analysis']}\n"

    prompt = (
        "Based on the following frame details, generate a coherent narrative that tells a "
        "comprehensive story of the video. Integrate the transcriptions and the analysis points "
        "to describe what is happening throughout the video in a clear, engaging, and flowing manner:\n\n"
        f"{combined_details}"
    )

    try:
        response = client.chat_completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1600,
            temperature=0.7,
        )
        story = response.choices[0].message["content"].strip()
    except Exception as e:
        print(f"Error generating narrative from frame analyses: {str(e)}")
        story = combined_details

    return story

def process_video(video_path, interval=1, blur_threshold=100, transcription_model="large"):
    """Processes the video: transcribes audio, analyzes frames, and generates a summary."""
    print(f"Processing video: {video_path}")

    audio_path = extract_audio_from_video(video_path)
    transcriptions = transcribe_audio_with_timestamps(audio_path, transcription_model)
    frame_analyses = extract_frames_at_intervals(video_path, interval, transcriptions, blur_threshold)
    video_summary = combine_frame_analyses(frame_analyses)

    results = {
        "video_file": os.path.basename(video_path),
        "frame_analyses": frame_analyses,
        "video_summary": video_summary,
    }

    # Build the JSON file name based on the video file's base name
    base = os.path.splitext(os.path.basename(video_path))[0]
    output_file = f"{base}.json"
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=4, ensure_ascii=False)

    print(f"Analysis completed. Results saved to {output_file}")

    if os.path.exists(audio_path):
        os.remove(audio_path)

##############################
# Entry Point for Video Processing
##############################

def process_video_file(file_path):
    """
    Processes a video file for analysis and compliance.
    - file_path: Path to the uploaded video (from main.py).
    Returns a dictionary with the analysis result (if needed) and pending clarifications.
    """
    # Generate JSON filenames based on the uploaded video's basename.
    base = os.path.splitext(os.path.basename(file_path))[0]
    video_description_json = f"{base}.json"

    # Process the video and generate the analysis JSON.
    process_video(file_path, interval=1, blur_threshold=100, transcription_model="large")

    return {
        "analysis_result": f"Analysis saved to {video_description_json}",
    }