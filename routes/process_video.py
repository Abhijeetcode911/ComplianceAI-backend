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
from backend.models import model_image,model_text,model_assistant

# Load environment variables and API key
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Determine the path to the JSON file (assuming it's in the same directory)
current_dir = os.path.dirname(__file__)
rules_path = os.path.join(current_dir, 'rules.json')

# Load the JSON file into a dictionary
with open(rules_path, 'r') as file:
    rules = json.load(file)

# Option 1: Create individual variables for each rule
Junglee_Internal_Guidlines = rules.get("Junglee_Internal_Guidlines")
rule_trademark = rules.get("rule_trademark")
rule_truthful_honest = rules.get("rule_truthful_honest")
rule_non_offensive = rules.get("rule_non_offensive")
rule_against_harmful = rules.get("rule_against_harmful")
rule_fair_competition = rules.get("rule_fair_competition")
rule_asci_guidelines = rules.get("rule_asci_guidelines")
rule_influencer_advertising = rules.get("rule_influencer_advertising")
rule_misleading_advertisment = rules.get("rule_misleading_advertisment")
##############################
# Video Analysis Functions
##############################

def extract_audio_from_video(video_path):
    """Extract audio from a video file and return the temporary audio file path."""
    audio_path = "temp_audio.wav"
    video = VideoFileClip(video_path)
    video.audio.write_audiofile(audio_path)
    return audio_path

def transcribe_audio_with_timestamps(audio_path, model_name="base"):
    """Transcribe audio with timestamps using Whisper."""
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path, verbose=True)
    return [{"start": seg["start"], "end": seg["end"], "text": seg["text"]} for seg in result["segments"]]

def compress_resize_encode_image(image_path, max_size=(1024, 1024), quality=85):
    """Compress, resize, and encode an image as a base64 string."""
    try:
        with Image.open(image_path) as img:
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.thumbnail(max_size)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", optimize=True, quality=quality)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None

def analyze_image(image_path, transcript_text=""):
    """Analyzes the image using OpenAI API and returns the output."""
    try:
        transcript_text = transcript_text or "No transcription available for this frame."
        base64_image = compress_resize_encode_image(image_path)
        if not base64_image:
            print(f"Skipping frame {image_path} due to encoding error.")
            return None
        prompt = f"Analyze this image and describe its content comprehensively. The frame transcription is: {transcript_text}"
        response = model_image(prompt, "", base64_image)
        return response.replace("\n", "").replace("**", "").replace('"', "")
    except Exception as e:
        print(f"Error analyzing image {image_path}: {str(e)}")
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
        story = model_text(prompt,"")
        return story.strip().replace("\n", "").replace("**", "").replace('"', "")
    except Exception as e:
        print(f"Error generating narrative from frame analyses: {str(e)}")
        story = combined_details

    return story

def process_video(file_path, interval=1, blur_threshold=100, transcription_model="base"):
    """Processes the video: transcribes audio, analyzes frames, and generates a summary."""
    print(f"Processing video: {file_path}")

    audio_path = extract_audio_from_video(file_path)
    transcriptions = transcribe_audio_with_timestamps(audio_path, transcription_model)
    frame_analyses = extract_frames_at_intervals(file_path, interval, transcriptions, blur_threshold)
    analysis = combine_frame_analyses(frame_analyses)

    results = {
        "file_name": os.path.basename(file_path),
        "frame_analyses": frame_analyses,
        "analysis" : analysis
    }

    # Build the JSON file name based on the video file's base name
    base = os.path.splitext(os.path.basename(file_path))[0]
    output_file = f"{base}_analysis.json"
    with open(output_file, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=4, ensure_ascii=False)

    print(f"Analysis completed. Results saved to {output_file}")

    if os.path.exists(audio_path):
        os.remove(audio_path)

    return analysis



##############################
# Compliance Checking Functions for Video
##############################
def generate_compliance_prompt(file_name, analysis, custom_rule):
    return f"""
        Ad  file name: {file_name}
        Summary: {analysis}
        Rule: {custom_rule}

        Please provide up to two clarifying questions only if there are genuine ambiguities or missing details strictly related to any potential grey areas. 
        If there are no such ambiguities, respond with "No questions."

        (No compliance evaluation or detailed assessment is needed—only questions, if any.)
        """


global_compliance_thread_id = None

def get_clarifications_video(file_name, analysis, rule):
    global global_compliance_thread_id

    prompt = generate_compliance_prompt(file_name, analysis, rule)
    print(f"\nProcessing compliance rule for video:\n{prompt}\n")
    
    # Reuse existing thread if available; otherwise, create a new thread
    if global_compliance_thread_id is None:
        thread = client.beta.threads.create()
        global_compliance_thread_id = thread.id
    else:
        # Create a dummy object with the existing thread ID for consistency
        thread = type('Thread', (), {})()
        thread.id = global_compliance_thread_id
    thread_id, clarifying_response = model_assistant(prompt, thread)
    prompt_2 = f"Convert the text to markdown format, text:{clarifying_response}"
    clarifying_response= model_text(prompt_2, "").strip()
    return thread.id, clarifying_response

def process_compliance_assistant_video(input_path, output_path, rules):
    """
    Processes compliance for the video by:
      1. Loading the video analysis JSON.
      2. For each rule, starting a thread to generate clarifying questions.
      3. Saving and returning the pending clarifications.
    Returns a dictionary of pending clarifications.
    """
    global global_compliance_thread_id
    try:
        with open(input_path, "r") as file:
            data = json.load(file)

        file_name = data.get("file_name")
        analysis = data.get("analysis")

        if not file_name or not analysis:
            print("Missing 'file_name' or 'analysis' in input JSON.")
            return {}

        # Hardcoded display names for eight fixed rules
        fixed_display_names = [
            "Junglee_Internal_Guidlines",
            "Trademark Guidelines",
            "Truthfulness & Honesty",
            "Non-offensiveness",
            "Avoidance of Harmful Content",
            "Fair Competition",
            "ASCI Guidelines",
            "Influencer Advertising"
        ]

        pending_clarifications = {}
        for idx, rule in enumerate(rules, start=1):
            thread_id, clarifying_response = get_clarifications_video(file_name, analysis, rule)
            if not thread_id:
                print(f"Skipping Rule {idx} due to error in getting clarifications.")
                continue

            # Assign a display name based on the fixed list (fallback to "Rule {idx}" if out of range)
            display_name = fixed_display_names[idx - 1] if idx <= len(fixed_display_names) else f"Rule {idx}"

            pending_clarifications[f"Rule {idx}"] = {
                "thread_id": thread_id,
                "rule": rule,
                "displayName": display_name,
                "clarifying_questions": clarifying_response,
                "file_name": file_name
            }

        with open(output_path, "w") as output_file:
            json.dump(pending_clarifications, output_file, indent=4)
        print(f"Compliance clarifications pending. Details saved to '{output_path}'.")
        global_compliance_thread_id = None
        return pending_clarifications

    except json.JSONDecodeError as json_error:
        print(f"Error reading JSON file: {json_error}")
        return {}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}


def process_json_file(input_path, output_path, rules):
    """
    Processes the video analysis JSON to generate pending compliance clarifications.
    Returns the pending clarifications for the frontend.
    """
    try:
        with open(input_path, "r") as file:
            data = json.load(file)
        file_name = data.get("file_name")
        anslysis = data.get("analysis")
        if not file_name or not anslysis:
            print("Missing 'file_name' or 'analysis' in input JSON.")
            return
        pending_clarifications = process_compliance_assistant_video(input_path, output_path, rules)
        return pending_clarifications
    except Exception as e:
        print(f"An error occurred: {e}")

##############################
# New Entry Point for Video Processing
##############################

def process_video_file(file_path):
    """
    Processes a video file for analysis and compliance.
    - file_path: Path to the uploaded video (from main.py).
    Returns a dictionary with the analysis result (if needed) and pending clarifications.
    """
    # Generate JSON filenames based on the uploaded video's basename.
    base = os.path.splitext(os.path.basename(file_path))[0]
    video_description_json = f"{base}_analyis.json"
    video_compliance_report = f"{base}_questions.json"
    
    # Process the video and generate the analysis JSON.
    analysis_result = process_video(file_path, interval=1, blur_threshold=100, transcription_model="base")
    
    # Process the analysis JSON to generate pending compliance clarifications.
    pending_clarifications = process_json_file(video_description_json, video_compliance_report, compliance_rules)
    
    return {"analysis_result": analysis_result, "pending_clarifications": pending_clarifications, "file_name": file_path}

##############################
# Compliance Rules
##############################



# Combine all the individual rules into one list
compliance_rules = [
    Junglee_Internal_Guidlines,
    # rule_trademark,
    # rule_truthful_honest,
    # rule_non_offensive,
    # rule_against_harmful,
    # rule_fair_competition,
    # rule_asci_guidelines,
    # rule_influencer_advertising
]

