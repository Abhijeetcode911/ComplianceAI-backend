import os
import json
import base64
import io
import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from dotenv import load_dotenv
from openai import OpenAI, AsyncOpenAI
from PIL import Image
import re
import asyncio
import uuid
import logging
import requests
from models import model_text, model_image  # Synchronous functions from your models
from supabase_config import supabase  # Import Supabase client

# Load environment variables and initialize clients
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_async(coro):
    """
    Creates a new event loop to run a coroutine and then closes it.
    This avoids issues with reusing or closing the default event loop.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
# --- Helper functions ---

def compress_resize_encode_image_from_bytes(image_bytes, max_size=(1024, 1024), quality=85) -> str:
    """
    Opens an image from bytes, resizes and compresses it, and returns a base64-encoded JPEG.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode == "RGBA":
                img = img.convert("RGB")
            img.thumbnail(max_size)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", optimize=True, quality=quality)
            return base64.b64encode(buffer.getvalue()).decode("utf-8")
    except Exception as e:
        logging.error(f"Error processing image bytes: {e}")
        return None

def is_frame_blurry(frame, threshold=100):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray_frame, cv2.CV_64F).var() < threshold

def download_from_supabase(bucket: str, file_key: str) -> bytes:
    """
    Downloads file bytes from a given Supabase bucket using its file key.
    """
    try:
        file_bytes = supabase.storage.from_(bucket).download(file_key)
        return file_bytes
    except Exception as e:
        logging.error(f"Error downloading {file_key} from {bucket}: {e}")
        return None

def get_public_url(bucket: str, file_key: str) -> str:
    """
    Returns the public URL for a file stored in a given Supabase bucket.
    """
    try:
        return supabase.storage.from_(bucket).get_public_url(file_key)
    except Exception as e:
        logging.error(f"Error getting public URL for {file_key} in {bucket}: {e}")
        return None

def handle_file_upload(bucket: str, file_key: str, file_data: bytes) -> None:
    """
    Uploads file_data (bytes) to the specified bucket with the given file_key.
    """
    try:
        supabase.storage.from_(bucket).upload(file_key, file_data, {"upsert": "true"})
        logging.info(f"Uploaded file to {bucket}/{file_key}")
    except Exception as e:
        logging.error(f"Error uploading file to {bucket}/{file_key}: {e}")

# --- Audio Processing (Video) ---
def extract_audio_from_video_url(video_url: str) -> bytes:
    """
    Uses MoviePy to extract audio from a video given its public URL.
    Returns the audio data as WAV bytes (in memory).
    """
    try:
        # MoviePy accepts a URL if the video is public.
        video = VideoFileClip(video_url)
        # Write audio to a BytesIO buffer using wave.
        sample_rate = int(video.audio.fps)
        audio_array = video.audio.to_soundarray(fps=sample_rate)
        audio_array_int16 = (audio_array * 32767).astype(np.int16)
        channels = audio_array_int16.shape[1] if audio_array_int16.ndim > 1 else 1
        buffer = io.BytesIO()
        import wave  # local import here
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(sample_rate)
            wf.writeframes(audio_array_int16.tobytes())
        video.close()
        return buffer.getvalue()
    except Exception as e:
        logging.error(f"Error extracting audio from video URL: {e}")
        return None

def check_audio_disclaimer_from_bytes(wav_bytes: bytes) -> str:
    """
    Uses the OpenAI API to check the audio disclaimer from WAV bytes.
    """
    try:
        encoded_string = base64.b64encode(wav_bytes).decode("utf-8")
        completion = client.chat.completions.create(
            model="gpt-4o-audio-preview",
            modalities=["text", "audio"],
            audio={"voice": "alloy", "format": "wav"},
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text",
                         "text": ("Does this audio contain the disclaimer: 'This game may be habit-forming or financially risky. Play responsibly.'? "
                                  "It should be presented for at least 4 seconds at a normal speaking pace.")},
                        {"type": "input_audio",
                         "input_audio": {"data": encoded_string, "format": "wav"}}
                    ]
                },
            ]
        )
        transcript = completion.choices[0].message.audio.transcript
        logging.info(f"Audio disclaimer transcription: {transcript}")
        return transcript
    except Exception as e:
        logging.error(f"Error checking audio disclaimer: {e}")
        return "Error checking audio disclaimer."

# --- Asynchronous API calls ---

async def model_assistant_acreate(prompt: str) -> str:
    try:
        response = await aclient.chat.completions.create(
            model="o3-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in model_assistant_acreate: {e}")
        return None

async def model_image1(prompt: str, instruction: str, base64_image: str) -> str:
    prompt_full = instruction + prompt
    response = await aclient.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt_full},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]}]
    )
    return response.choices[0].message.content

# --- Asynchronous image analysis functions ---

async def analyze_image_async_from_bytes(image_bytes: bytes) -> str:
    base64_image = compress_resize_encode_image_from_bytes(image_bytes)
    if not base64_image:
        logging.error("Skipping image due to encoding error.")
        return None
    prompt_text = (
        "Ad image analysis:\n"
        "Extract and list only the key observable visual elements (text, disclaimers, layout, graphics) as they appear in the ad, "
        "without interpretation. Follow these guidelines:\n"
        "1. Text and Disclaimers: exact wording, placement and size.\n"
        "2. Font, color, and formatting details.\n"
        "3. Layout and positioning.\n"
        "4. Visual elements such as graphics and animations.\n"
        "Then, perform a compliance evaluation using the provided ASCI guidelines."
    )
    response = await model_image1(prompt_text, "", base64_image)
    return response.replace("\n", "").replace("**", "").replace('"', "")

async def analyze_image_from_supabase(file_key: str) -> str:
    """
    Downloads image bytes from the Supabase 'uploads' bucket and analyzes them.
    """
    image_bytes = download_from_supabase("uploads", file_key)
    if not image_bytes:
        return "Image not found."
    return await analyze_image_async_from_bytes(image_bytes)

# --- Asynchronous frame extraction without local files ---

async def extract_frames_at_intervals_merged_async(video_key: str, interval: float, blur_threshold: int = 100) -> list:
    """
    Extracts frames from a video stored in the Supabase 'uploads' bucket.
    Uses the public URL of the video to create a VideoFileClip.
    For each eligible frame (or merged pair), the frame is processed in memory and optionally uploaded to the 'frames' bucket.
    Returns a list of dictionaries with frame details.
    """
    # Get public URL for the video from Supabase
    video_url = get_public_url("uploads", video_key)
    if not video_url:
        logging.error("Video URL not available.")
        return []
    
    cap = cv2.VideoCapture(video_url)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = frame_count / fps
    results = []
    tasks = []
    
    # Use a unique folder for this session in the 'frames' bucket.
    folder_uuid = str(uuid.uuid4())
    
    timestamp = max(0, total_duration - 6)  # analyze last 6 seconds
    while timestamp < total_duration:
        frame_index1 = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index1)
        success1, frame1 = cap.read()
        if not success1:
            timestamp += interval
            continue
        if is_frame_blurry(frame1, threshold=blur_threshold):
            logging.info(f"Skipping blurry frame at index {frame_index1} (timestamp: {timestamp:.2f}s)")
            timestamp += interval
            continue
        
        next_timestamp = timestamp + interval
        frame_index2 = int(next_timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index2)
        success2, frame2 = cap.read()
        if success2 and not is_frame_blurry(frame2, threshold=blur_threshold):
            # Merge frames horizontally
            merged_frame = np.hstack((frame1, frame2))
            merged_frame_id = str(uuid.uuid4())
            filename = f"merged_frame_{merged_frame_id}.jpg"
            storage_path = f"{folder_uuid}/{filename}"
            ret, buffer = cv2.imencode('.jpg', merged_frame)
            if not ret:
                logging.error(f"Failed to encode merged frame at timestamp {timestamp}")
                timestamp += 2 * interval
                continue
            image_bytes = buffer.tobytes()
            # Optionally upload the merged frame to the 'frames' bucket
            try:
                await asyncio.to_thread(handle_file_upload, "frames", storage_path, image_bytes)
            except Exception as e:
                logging.error(f"Upload failed for merged frame: {e}")
            public_url = get_public_url("frames", storage_path)
            task = asyncio.create_task(analyze_image_async_from_bytes(image_bytes))
            tasks.append((merged_frame_id, timestamp, public_url, task))
            timestamp += 2 * interval
        else:
            # Process single frame
            frame_id = str(uuid.uuid4())
            filename = f"frame_{frame_id}.jpg"
            storage_path = f"{folder_uuid}/{filename}"
            ret, buffer = cv2.imencode('.jpg', frame1)
            if not ret:
                logging.error(f"Failed to encode frame at timestamp {timestamp}")
                timestamp += interval
                continue
            image_bytes = buffer.tobytes()
            try:
                await asyncio.to_thread(handle_file_upload, "frames", storage_path, image_bytes)
            except Exception as e:
                logging.error(f"Upload failed for frame: {e}")
            public_url = get_public_url("frames", storage_path)
            task = asyncio.create_task(analyze_image_async_from_bytes(image_bytes))
            tasks.append((frame_id, timestamp, public_url, task))
            timestamp += interval
    
    cap.release()
    for frame_id, ts, public_url, task in tasks:
        analysis = await task
        results.append({"frame_id": frame_id, "timestamp": ts, "analysis": analysis, "frame_url": public_url})
    return results

# --- Synchronous narrative generation (wrapped asynchronously) ---

def combine_frame_analyses(frame_analyses, final_transcription=""):
    combined_details = "The following frame analysis details are provided:\n"
    for frame in frame_analyses:
        combined_details += f"\nAt {frame['timestamp']:.2f} seconds:\n"
        combined_details += f"Analysis: {frame['analysis']}\n"
    prompt = (
        "Based on the following frame details, generate a coherent narrative that describes the video for compliance evaluation. "
        "Your narrative should objectively capture all observable elements, including subjects, actions, layout, text overlays, disclaimers, and promotional messages.\n\n"
        "Final Transcription:\n" + final_transcription + "\n\n" +
        "Batch Analysis Details:\n" + combined_details
    )
    try:
        return model_text(prompt, "").strip().replace("\n", "").replace("**", "").replace('"', "")
    except Exception as e:
        logging.error(f"Error generating narrative: {e}")
        return combined_details

async def combine_frame_analyses_async(frame_analyses, final_transcription=""):
    return await asyncio.to_thread(combine_frame_analyses, frame_analyses, final_transcription)

# --- Main async function for visual law analysis ---

async def visual_law(file_key: str) -> str:
    """
    Performs visual law analysis for a file stored in Supabase.
    For images, it downloads from the 'uploads' bucket and analyzes them.
    For videos, it obtains the public URL from the 'uploads' bucket and processes audio and frames.
    """
    ext = os.path.splitext(file_key)[1].lower()
    if ext in ['.png', '.jpeg', '.jpg']:
        return await analyze_image_from_supabase(file_key)
    elif ext == '.mp4':
        video_url = get_public_url("uploads", file_key)
        if not video_url:
            return "Video URL not available."
        # Extract audio from the video via its public URL (in memory)
        wav_bytes = extract_audio_from_video_url(video_url)
        audio_disclaimer_result = check_audio_disclaimer_from_bytes(wav_bytes) if wav_bytes else "Audio extraction failed."
        # Extract frames from video without local storage
        frames_analysis = await extract_frames_at_intervals_merged_async(file_key, interval=1, blur_threshold=100)
        if not frames_analysis:
            return "No frames analyzed."
        combined_narrative = await combine_frame_analyses_async(frames_analysis, audio_disclaimer_result)
        return f"Visual law analysis: {combined_narrative}"
    else:
        return "Unsupported file type for visual law analysis."

