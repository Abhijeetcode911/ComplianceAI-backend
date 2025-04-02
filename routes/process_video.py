import os
import json
import base64
import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import io
import uuid
import logging
from supabase_config import supabase  # Ensure Supabase is correctly configured
import requests
from models import model_image, model_text, model_assistant
from openai import AsyncOpenAI
import wave
import asyncio

# Load environment variables and API key
load_dotenv()
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

prompt_instructions = (
    "Describe the image by listing all observable elements such as subjects, actions, layout, text, and graphics. "
    "Include any visible or audible details without adding interpretation or narrative. "
    "Simply extract and list what is directly seen or heard. "
    "Refer any person as an actor/influencer, without trying to identify him."
)

# Load rules from Supabase storage
def load_rules_from_supabase() -> dict:
    """
    Fetches the rules.json file from Supabase storage and loads it into a dictionary.

    Returns:
        dict: A dictionary containing the rules.
    """
    bucket_name = "rules"  # Supabase bucket name
    supabase_path = "rules.json"  # Path in Supabase storage

    try:
        # Fetch the file from Supabase storage
        url = supabase.storage.from_(bucket_name).get_public_url(supabase_path)
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)
        
        rules = response.json()
        print("Rules fetched successfully from Supabase.")
        return rules
    except Exception as e:
        print(f"Error fetching rules from Supabase: {e}")
        return {}

# Load rules
rules = load_rules_from_supabase()

Junglee_Internal_Guideline = rules.get("Junglee_Internal_Guideline")
Junglee_internal_gaming_IP_laws = rules.get("Junglee_internal_gaming_IP_laws")
ASCII_code_chapter_1 = rules.get("ASCII_code_chapter_1")
ASCII_code_chapter_2 = rules.get("ASCII_code_chapter_2")
ASCII_code_chapter_3 = rules.get("ASCII_code_chapter_3")
ASCII_code_chapter_4 = rules.get("ASCII_code_chapter_4")
ASCII_RealMoney_Guideline = rules.get("ASCII_RealMoney_Guideline")
Guidelines_for_Influencer_Advertising_in_Digital_Media = rules.get("Guidelines_for_Influencer_Advertising_in_Digital_Media")
CCPA_Guideline = rules.get("CCPA_Guideline")

compliance_rules = [
    Junglee_Internal_Guideline,
    Junglee_internal_gaming_IP_laws,
    ASCII_code_chapter_1,
    ASCII_code_chapter_2,
    ASCII_code_chapter_3,
    ASCII_code_chapter_4,
    Guidelines_for_Influencer_Advertising_in_Digital_Media,
    ASCII_RealMoney_Guideline,
    CCPA_Guideline
]

fixed_display_names = [
    "Junglee Internal Guidelines",
    "Junglee_internal_gaming_IP_laws",
    "ASCII_code_chapter_1",
    "ASCII_code_chapter_2",
    "ASCII_code_chapter_3",
    "ASCII_code_chapter_4",
    "Guidelines_for_Influencer_Advertising_in_Digital_Media",
    "ASCII_RealMoney_Guideline",
    "CCPA_Guideline"
]
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

async def model_image1(prompt, instruction, base64_image):
    """
    Asynchronously calls the OpenAI API to analyze an image.
    """
    prompt_full = instruction + prompt
    response = await aclient.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt_full}",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content

async def model_assistant_acreate(prompt):
    try:
        response = await aclient.chat.completions.create(
            model="o3-mini",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        content = response.choices[0].message.content
        return content
    except Exception as e:
        print(f"Error in model_assistant_acreate: {e}")
        return None

### 1. Audio Extraction (In-Memory)
def get_public_url(bucket: str, file_key: str) -> str:
    """
    Returns the public URL for a file stored in a given Supabase bucket.
    """
    try:
        return supabase.storage.from_(bucket).get_public_url(file_key)
    except Exception as e:
        logging.error(f"Error getting public URL for {file_key} in {bucket}: {e}")
        return None

### Updated Audio Extraction (Using Public URL)
def extract_audio_from_video(file_key: str):
    """
    Extracts audio from a video file stored in the Supabase 'uploads' bucket.
    This function obtains the public URL of the video, opens it with MoviePy,
    extracts the audio, and uploads the audio to the Supabase 'audios' bucket.
    Returns the storage key for the uploaded audio, or None if an error occurs.
    """
    # Get the public URL from Supabase
    video_url = get_public_url("uploads", file_key)
    if not video_url:
        logging.error("Video URL not available from Supabase.")
        return None

    logging.info(f"Extracting audio from video at URL: {video_url}")
    try:
        video = VideoFileClip(video_url)
    except Exception as e:
        logging.error(f"Error opening video file from URL: {e}")
        return None

    # Check if the video has audio
    if not video.audio:
        logging.error("Video has no audio track.")
        video.close()
        return None

    try:
        audio_clip = video.audio
        sample_rate = int(audio_clip.fps)
        logging.info(f"Audio sample rate: {sample_rate} Hz")
        audio_array = audio_clip.to_soundarray(fps=sample_rate)
        if audio_array.size == 0:
            logging.error("Audio array is empty.")
            video.close()
            return None
    except Exception as e:
        logging.error(f"Error extracting audio array: {e}")
        video.close()
        return None

    try:
        audio_array_int16 = (audio_array * 32767).astype(np.int16)
        channels = audio_array_int16.shape[1] if audio_array_int16.ndim > 1 else 1
        buffer = io.BytesIO()
        import wave  # local import for wave module
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(2)  # 16-bit audio => 2 bytes per sample.
            wf.setframerate(sample_rate)
            wf.writeframes(audio_array_int16.tobytes())
        wav_bytes = buffer.getvalue()
    except Exception as e:
        logging.error(f"Error converting audio to WAV bytes: {e}")
        video.close()
        return None

    video.close()

    # Generate a unique filename for the audio file and include a folder if needed
    unique_audio_filename = f"{uuid.uuid4()}.wav"
    storage_key = unique_audio_filename  # or f"audios/{unique_audio_filename}" if needed

    try:
        supabase.storage.from_("audios").upload(storage_key, wav_bytes, {"upsert": "true"})
        logging.info(f"Audio uploaded to Supabase at {storage_key}")
    except Exception as e:
        logging.error(f"Error uploading audio to Supabase: {e}")
        return None

    return storage_key




### 2. Audio Transcription
def transcribe_audio_with_timestamps(audio_key):
    """
    Transcribe audio using the GPT-4o-audio-preview model.
    The audio file is downloaded from the Supabase 'audios' bucket using the provided audio_key.
    """
    audio_data = supabase.storage.from_("audios").download(audio_key)
    if not audio_data:
        logging.error("Audio file not found in Supabase.")
        return None
    encoded_string = base64.b64encode(audio_data).decode('utf-8')
    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "transcribe the audio and if any music has been used?"},
                    {"type": "input_audio", "input_audio": {"data": encoded_string, "format": "wav"}}
                ]
            }
        ]
    )
    transcript = completion.choices[0].message.audio.transcript
    logging.info(f"Transcription: {transcript}")
    return transcript


### 3. Image Processing from Bytes (In-Memory)
def compress_resize_encode_image_from_bytes(image_bytes, max_size=(1024, 1024), quality=85):
    """
    Open image from bytes, resize and compress it, and return a base64-encoded JPEG.
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

### 4. Asynchronous Image Analysis from Bytes
async def analyze_image_async_from_bytes(image_bytes, transcript_text=""):
    """
    Asynchronously analyze the image using the OpenAI API.
    Operates on image bytes (in memory) rather than a local file.
    """
    try:
        transcript_text = transcript_text or "No transcription available for this frame."
        base64_image = compress_resize_encode_image_from_bytes(image_bytes)
        if not base64_image:
            logging.error("Failed to compress/encode image from bytes.")
            return None
        prompt = prompt_instructions
        response = await model_image1(prompt, "", base64_image)
        return response.replace("\n", "").replace("**", "").replace('"', "")
    except Exception as e:
        logging.error(f"Error analyzing image from bytes: {e}")
        return None
    
def is_frame_blurry(frame, threshold=100):
    """Determines if a frame is blurry based on the variance of the Laplacian."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
    return variance < threshold

### 5. Frame Extraction and Analysis (Avoiding Local Files)
async def extract_frames_at_intervals_merged(video_path, interval, blur_threshold=100):
    """
    Extracts frames from the video at regular intervals.
    For each eligible frame (or merged pair), the frame is encoded to JPEG,
    uploaded to the Supabase "frames" bucket under a UUID-based path, and analyzed directly from its bytes.
    Returns a list of dictionaries with frame details including the public URL.
    """
    video_url = get_public_url("uploads", video_path)
    if not video_url:
        logging.error("Video URL not available.")
        return []
    cap = cv2.VideoCapture(video_url)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        logging.error("FPS is 0, cannot process video.")
        return []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = frame_count / fps

    results = []
    tasks = []
    folder_uuid = str(uuid.uuid4())  # Unique folder for this extraction session
    timestamp = 0

    while timestamp < total_duration:
        # Read first frame at the current timestamp.
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

        # Attempt to read a second frame.
        next_timestamp = timestamp + interval
        frame_index2 = int(next_timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index2)
        success2, frame2 = cap.read()

        if success2 and not is_frame_blurry(frame2, threshold=blur_threshold):
            # Merge frames horizontally.
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
            try:
                await asyncio.to_thread(supabase.storage.from_("frames").upload, storage_path, image_bytes, {"upsert": "true"})
            except Exception as e:
                logging.error(f"Failed to upload merged frame: {e}")
                timestamp += 2 * interval
                continue
            public_url = supabase.storage.from_("frames").get_public_url(storage_path)
            task = asyncio.create_task(analyze_image_async_from_bytes(image_bytes, ""))
            tasks.append((merged_frame_id, timestamp, public_url, task))
            timestamp += 2 * interval
        else:
            # Process single frame.
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
                await asyncio.to_thread(supabase.storage.from_("frames").upload, storage_path, image_bytes, {"upsert": "true"})
            except Exception as e:
                logging.error(f"Failed to upload frame: {e}")
                timestamp += interval
                continue
            public_url = supabase.storage.from_("frames").get_public_url(storage_path)
            task = asyncio.create_task(analyze_image_async_from_bytes(image_bytes, ""))
            tasks.append((frame_id, timestamp, public_url, task))
            timestamp += interval

    cap.release()
    for frame_id, ts, public_url, task in tasks:
        analysis = await task
        results.append({
            "frame_id": frame_id,
            "timestamp": ts,
            "analysis": analysis,
            "frame_url": public_url
        })
    return results


### 6. Combine Frame Analyses into Narrative
def combine_frame_analyses(frame_analyses, final_transcription=""):
    combined_details = "The following frame analysis details are provided:\n"
    for frame in frame_analyses:
        combined_details += f"\nAt {frame['timestamp']:.2f} seconds:\n"
        combined_details += f"Analysis: {frame['analysis']}\n"
    prompt = (
      "Based on the following frame details, generate a coherent narrative that describes the video for compliance evaluation. "
      "Your narrative should objectively capture all observable elements, including subjects, actions, layout, text overlays, disclaimers, and promotional messages. "
      "Ensure that you address the following aspects:\n\n"
      "Content Depiction: Detail all visible and audible elements, specifying subjects, actions, and interactions (including age, gender, attire, etc.).\n"
      "Visual Elements: Describe the layout, background, setting, text overlays, graphics, and animations, including details like fonts, colors, and the placement of disclaimers or promotional messages.\n"
      "Messaging and Claims: Summarize any explicit or implicit messaging, claims, promises, or calls-to-action present in the content.\n"
      "Promotions and Offers: Note any promotions, bonuses, or offers, including their phrasing and presentation.\n"
      "Disclaimers: Document the presence, placement, and formatting of any disclaimers, ensuring they are clear and compliant.\n"
      "Behavior and Practices: Describe any observed behaviors, especially those involving risks or sensitive subjects.\n"
      "Cultural and Social Indicators: Note any elements related to race, religion, gender, or other societal factors in a neutral, factual manner.\n\n"
      "Final Transcription:\n" + final_transcription + "\n\n" +
      "Batch Analysis Details:\n" + combined_details
    )
    try:
        story = model_text(prompt, "")
        return story.strip().replace("\n", "").replace("**", "").replace('"', "")
    except Exception as e:
        logging.error(f"Error generating narrative: {e}")
        return combined_details


### 7. Process Video (No Local Cleanup Required)
async def process_video(file_path, interval=1, blur_threshold=100, transcription_model="base"):
    """
    Processes the video: extracts audio, analyzes frames concurrently, and generates a summary.
    Before processing, attempts to retrieve an existing analysis JSON from the Supabase "results" bucket.
    """
    logging.info(f"Processing video: {file_path}")
    file_name = os.path.basename(file_path).split('.')[0]
    storage_key = f"{file_name}_analysis.json"
    
    audio_storage_key = extract_audio_from_video(file_path)
    transcriptions = transcribe_audio_with_timestamps(audio_storage_key)
    frame_analyses = await extract_frames_at_intervals_merged(file_path, interval, blur_threshold)
    analysis = combine_frame_analyses(frame_analyses, final_transcription=transcriptions)
    results = {
        "file_name": os.path.basename(file_path),
        "frame_analyses": frame_analyses,
        "analysis": analysis
    }
    json_data = json.dumps(results, indent=4, ensure_ascii=False).encode("utf-8")
    try:
        await asyncio.to_thread(supabase.storage.from_("analysis").upload, storage_key, json_data, {"upsert": "true"})
        logging.info(f"Uploaded analysis to Supabase with key: {storage_key}")
    except Exception as e:
        logging.error(f"Error uploading analysis to Supabase: {e}")
    
    return analysis, storage_key

### 8. Compliance Rule Prompts and Processing
def generate_extraction_prompt(file_name, analysis, custom_rule):
    prompt = f"""
        Ad image file name: {file_name}
        Summary: {analysis}
        Rule: {custom_rule}

        Strictly identify and extract only the guidelines from the custom rule that are definitively potentially violated as indicated by the analysis summary.
        For each identified guideline, return the complete text exactly as it appears, including the guideline heading, numbering, and all associated text—capturing every sentence or detail that follows, regardless of the format.
        Do not modify, paraphrase, or add any extra commentary.
        If no guideline is identified as potentially violated, respond with "No potential violations."
"""
    return prompt

def generate_clarification_prompt(file_name, analysis, extracted_violations):
    prompt = (
        f"Ad image file name: {file_name}\n"
        f"Summary: {analysis}\n"
        f"Extracted potential violations: {extracted_violations}\n\n"
        "If no potential violations are present, strictly respond with 'No questions.'\n"
        "Based solely on the above extracted potential violations, generate only the clarifying questions that are absolutely necessary to resolve any ambiguities regarding compliance or non-compliance. "
        "Do not include any additional commentary, evaluation, or extraneous questions. "
        "Your response must adhere strictly to the information provided above."
    )
    return prompt

async def get_extracted_violations(file_name, analysis, rule):
    prompt_text = generate_extraction_prompt(file_name, analysis, rule)
    logging.info(f"Processing Extraction for image:\n{prompt_text}\n")
    extraction_response = await model_assistant_acreate(prompt_text)
    return extraction_response

async def get_clarifications_video(file_name, analysis, rule, extracted_violations):
    prompt_text = generate_clarification_prompt(file_name, analysis, extracted_violations)
    logging.info(f"Processing Clarifications for image:\n{prompt_text}\n")
    clarifying_response = await model_assistant_acreate(prompt_text)
    return clarifying_response

async def process_rule(idx, file_name, analysis, rule):
    extracted_violations = await get_extracted_violations(file_name, analysis, rule)
    clarifying_response = await get_clarifications_video(file_name, analysis, rule, extracted_violations)
    display_name = fixed_display_names[idx - 1] if idx <= len(fixed_display_names) else f"Rule {idx}"
    logging.info(f"Completed processing rule {idx}")
    return (f"Rule {idx}", {
        "rule": extracted_violations,
        "clarifying_questions": clarifying_response,
        "displayName": display_name,
        "file_name": file_name
    })


### 9. Compliance Assistant Processing (Using Supabase for I/O)
async def process_compliance_assistant_video(input_path, output_path, rules):
    """
    Asynchronously processes each compliance rule.
    Downloads the input JSON from the Supabase "analysis" bucket and uploads output JSON to the "questions" bucket.
    """
    try:
        json_data = await asyncio.to_thread(supabase.storage.from_("analysis").download, input_path)
        data = json.loads(json_data.decode("utf-8"))
        file_name = data.get("file_name")
        analysis = data.get("analysis")
        if not file_name or not analysis:
            logging.error("Missing 'file_name' or 'analysis' in input JSON.")
            return {}
        tasks = [asyncio.create_task(process_rule(idx, file_name, analysis, rule))
                 for idx, rule in enumerate(rules, start=1)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        pending_clarifications = {}
        for result in results:
            if result is not None:
                key, value = result
                pending_clarifications[key] = value
        output_json = json.dumps(pending_clarifications, indent=4).encode("utf-8")
        await asyncio.to_thread(supabase.storage.from_("questions").upload, output_path, output_json, {"upsert": "true"})
        logging.info(f"Compliance clarifications saved to Supabase at '{output_path}'.")
        return pending_clarifications
    except Exception as e:
        logging.error(f"Error processing compliance assistant: {e}")
        return {}

def process_json_file(input_path, output_path, rules):
    """
    Processes the video analysis JSON to generate pending compliance clarifications.
    Downloads input JSON from Supabase "analysis bucket and uploads output JSON to Supabase "questions" bucket.
    """
    try:
        json_data = supabase.storage.from_("analysis").download(input_path)
        data = json.loads(json_data.decode("utf-8"))
        file_name = data.get("file_name")
        analysis = data.get("analysis")
        if not file_name or not analysis:
            logging.error("Missing 'file_name' or 'analysis' in input JSON.")
            return
        pending_clarifications = asyncio.run(process_compliance_assistant_video(input_path, output_path, rules))
        return pending_clarifications
    except Exception as e:
        logging.error(f"An error occurred: {e}")

def process_video_file(file_path):
    """
    Processes a video file for analysis and compliance.
    Calls process_video() to store analysis JSON in Supabase "results" bucket, then generates pending clarifications
    stored in the Supabase "questions" bucket.
    Returns a dictionary with the analysis result, pending clarifications, and the original file name.
    """
    try:
        file_name = os.path.basename(file_path).split('.')[0]
        video_compliance_report = f"{file_name}_questions.json"
        logging.info(f"Processing video: {file_path}")
        analysis_result, video_description_storage_key = asyncio.run(
            process_video(file_path, interval=1, blur_threshold=50, transcription_model="base")
        )
        pending_clarifications = process_json_file(video_description_storage_key, video_compliance_report, compliance_rules)
        return {"analysis_result": analysis_result, "pending_clarifications": pending_clarifications, "file_name": file_path}
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return {"error": str(e)}
