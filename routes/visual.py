import os
import json
import base64
import io
import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
import whisper
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import re
from models import model_text,model_image

# Load environment variables and API key
load_dotenv()
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

##############################
# Video and Image Analysis Functions
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

# Define a prompt for visual analysis.
visual_rules = (
    "You are a compliance expert analyzing an advertisement for online gaming for real money winnings. "
    "Please evaluate the advertisement image according to the following ASCI guidelines:\n\n"
    "ASCI GUIDELINES FOR ADVERTISING OF ONLINE GAMING FOR REAL MONEY WINNINGS:\n"
    "Guideline 2 - Every such gaming advertisement must carry the following disclaimer:\n"
    "a. Print/static: This game may be habit-forming or financially risky. Play responsibly.\n"
    "   i. The disclaimer should occupy no less than 20% of the advertisement space.\n"
    "   ii. It must specifically meet disclaimer guidelines 4(i), 4(ii), 4(iv), and 4(viii) of the ASCI code.\n\n"
    "ASCI GUIDELINES FOR DISCLAIMERS MADE IN SUPPORTING, LIMITING OR EXPLAINING CLAIMS MADE IN ADVERTISEMENTS:\n"
    "4) Legibility of Disclaimers:\n"
    "I. A disclaimer shall be in the same language as the advertisement's claims. In bilingual advertisements, "
    "the disclaimer should be in the dominant language.\n"
    "II. The font should be either the same as the claim or a sans serif for better readability, and not in italics.\n"
    "IV. The disclaimer's direction should match the majority of the copy, following the natural reading direction. "
    "Exceptions may exist for very small packages.\n"
    "VIII. The visual presentation must ensure that the disclaimer text contrasts with the background for clear legibility.\n\n"
    "Based on these guidelines, analyze the image and describe any compliance issues."
)

def analyze_image(file_path, transcript_text=""):
    """
    Analyzes the image using the OpenAI API along with a provided transcription (if any)
    and returns the compliance analysis output.
    """
    try:
        transcript_text = transcript_text or "No transcription available for this frame."
        base64_image = compress_resize_encode_image(file_path)
        if not base64_image:
            print(f"Skipping frame {file_path} due to encoding error.")
            return None
        prompt_text = (
            f"Given the following visual compliance rules:\n\n{visual_rules}\n\n"
            f"And the advertisement transcript (if any):\n\n{transcript_text}\n\n"
            "Identify any violations, missing disclaimers, or misleading claims strictly related to these rules. "
            "Output only the non-compliant items in a numbered list, with no additional explanation. "
            "If everything is compliant, output 'No non-compliant items found.'"
          )


        response = model_image(prompt_text,"",base64_image)
        return response.replace("\n", "").replace("**", "").replace('"', "")
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
    Extract frames from the video at regular intervals, match transcription,
    and exclude blurry frames.
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
        analysis_result = analyze_image(frame_path, transcription_text)
        if analysis_result:
            results.append({
                "frame_index": frame_index,
                "timestamp": timestamp,
                "transcription": transcription_text,
                "analysis": analysis_result,
            })

    cap.release()
    return results

def combine_frame_analyses(frame_analyses):
    """
    Combines individual frame analyses into a comprehensive narrative.
    Aggregates details from each frame and queries an LLM to generate a flowing story.
    """
    combined_details = "The following details are gathered from various points in the video:\n"
    for frame in frame_analyses:
        combined_details += f"\nAt {frame['timestamp']:.2f} seconds:\n"
        if frame.get("transcription"):
            combined_details += f"Transcription: {frame['transcription']}\n"
        combined_details += f"Analysis: {frame['analysis']}\n"

    prompt_combined = (
        "Based on the following frame details, generate a coherent narrative that tells a "
        "comprehensive story of the video. Integrate the transcriptions and the analysis points "
        "to describe what is happening throughout the video in a clear, engaging, and flowing manner:\n\n"
        f"{combined_details}"
    )
    try:
        response = model_text(prompt_combined,"")
        # Clean up and return response text.
        return response.replace("\n", "").replace("**", "").replace('"', "")
    except Exception as e:
        print(f"Error generating narrative from frame analyses: {str(e)}")
        return combined_details

def combine_image_analysis(analysis, transcription=""):
    """
    Combines the analysis for a single image by wrapping it as a frame analysis
    and calling the combine_frame_analyses function to generate a narrative.
    """
    frame_analyses = [{
        "timestamp": 0.0,
        "transcription": transcription or "No transcription available for this image.",
        "analysis": analysis,
    }]
    narrative = combine_frame_analyses(frame_analyses)
    return narrative


def visual_law(file_path):
    """
    Processes the given file path according to its type:
      - For image files (.png, .jpeg, .jpg): Analyzes the image using analyze_image() and generates a narrative.
      - For video files (.mp4): Extracts audio and transcribes it, then extracts frames at intervals,
        analyzes each frame, and combines the results into a comprehensive narrative.
    Returns a consolidated compliance analysis string.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext in ['.png', '.jpeg', '.jpg']:
        # For image files, analyze the image and generate a narrative.
        image_analysis_result = analyze_image(file_path)
        print(f"*****************{image_analysis_result}")
        return image_analysis_result
    elif ext == '.mp4':
        # For video files, extract audio and get transcriptions.
        audio_path = extract_audio_from_video(file_path)
        transcriptions = transcribe_audio_with_timestamps(audio_path)
        os.remove(audio_path)  # Clean up the temporary audio file
        
        # Extract and analyze frames at intervals (e.g., every 5 seconds)
        frames_analysis = extract_frames_at_intervals(file_path, interval=5, transcriptions=transcriptions)
        if not frames_analysis:
            return "No frames analyzed."
        # Combine individual frame analyses into a comprehensive narrative.
        combined_narrative = combine_frame_analyses(frames_analysis)
        return combined_narrative
    else:
        return "Unsupported file type for visual law analysis."

