from PIL import Image
import json
import base64
from dotenv import load_dotenv
from openai import OpenAI
import io
import uuid
import os
import requests
from openai import AsyncOpenAI
import asyncio
from models import model_image, model_assistant
from supabase_config import supabase  # Import Supabase client

# Load environment variables
load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load rules from Supabase storage
def load_rules_from_supabase() -> dict:
    """
    Fetches the rules.json file from Supabase storage and loads it into a dictionary.
    """
    bucket_name = "rules"  # Supabase bucket name
    supabase_path = "rules.json"  # Path in Supabase storage

    try:
        url = supabase.storage.from_(bucket_name).get_public_url(supabase_path)
        response = requests.get(url)
        response.raise_for_status()  # Raise error for bad responses
        rules = response.json()
        print("Rules fetched successfully from Supabase.")
        return rules
    except Exception as e:
        print(f"Error fetching rules from Supabase: {e}")
        return {}

rules = load_rules_from_supabase()

# Option 1: Create individual variables for each rule using new keys
Junglee_Internal_Guideline = rules.get("Junglee_Internal_Guideline")
Junglee_internal_gaming_IP_laws = rules.get("Junglee_internal_gaming_IP_laws")
ASCII_code_chapter_1 = rules.get("ASCII_code_chapter_1")
ASCII_code_chapter_2 = rules.get("ASCII_code_chapter_2")
ASCII_code_chapter_3 = rules.get("ASCII_code_chapter_3")
ASCII_code_chapter_4 = rules.get("ASCII_code_chapter_4")
ASCII_RealMoney_Guideline = rules.get("ASCII_RealMoney_Guideline")
Guidelines_for_Influencer_Advertising_in_Digital_Media = rules.get("Guidelines_for_Influencer_Advertising_in_Digital_Media")
CCPA_Guideline = rules.get("CCPA_Guideline")  # Optional if present

fixed_display_names = [
    "Junglee Internal Guidelines",
    "Junglee_internal_gaming_IP_laws",
    "ASCII_code_chapter_1 ",
    "ASCII_code_chapter_2 ",
    "ASCII_code_chapter_3 ",
    "ASCII_code_chapter_4 ",
    "Guidelines_for_Influencer_Advertising_in_Digital_Media",
    "ASCII_RealMoney_Guideline",
    "CCPA_Guideline"
]

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

prompt = (
    "Describe the image by listing all observable elements such as subjects, actions, layout, text, and graphics. "
    "Include any visible or audible details without adding interpretation or narrative. "
    "Simply extract and list what is directly seen or heard. "
    "Refer any person as an actor/influencer, without trying to identify him."
)

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
        print(f"Error processing image bytes: {e}")
        return None

def analyze_image(file_key: str) -> str:
    """
    Downloads an image from the Supabase 'uploads' bucket using the file key,
    then analyzes it using the OpenAI API.
    """
    try:
        # Download the image bytes from Supabase 'uploads' bucket
        image_data = supabase.storage.from_("uploads").download(file_key)
        if not image_data:
            print(f"Image not found in Supabase for key: {file_key}")
            return "Image not found"
    except Exception as e:
        print(f"Error downloading image from Supabase: {e}")
        return "Error downloading image"

    base64_image = compress_resize_encode_image_from_bytes(image_data)
    response = model_image(prompt, "", base64_image)
    return response.replace("\n", "").replace("**", "").replace('"', "")

def handle_file_upload(bucket_name: str, file_path: str, file_data: bytes) -> None:
    """
    Handles file upload to Supabase storage.
    """
    try:
        supabase.storage.from_(bucket_name).upload(file_path, file_data)
        print(f"Uploaded new file: {file_path}")
    except Exception as e:
        print(f"Error uploading file: {e}")

async def process_compliance_assistant(input_path: str, output_path: str, rules: list) -> dict:
    """
    Processes each compliance rule by getting clarifying questions.
    Downloads the input JSON from the Supabase "analysis" bucket via its public URL,
    and writes the output JSON locally.
    """
    try:
        bucket_name = "analysis"
        supabase_path = os.path.basename(input_path)
        url = supabase.storage.from_(bucket_name).get_public_url(supabase_path)
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        file_name = data.get("file_name")
        analysis = data.get("analysis")
        if not file_name or not analysis:
            print("Missing 'file_name' or 'analysis' in input JSON.")
            return {}

        tasks = [
            asyncio.create_task(process_rule(idx, file_name, analysis, rule))
            for idx, rule in enumerate(rules, start=1)
        ]
        results = await asyncio.gather(*tasks)

        pending_clarifications = {}
        for result in results:
            if result is not None:
                key, value = result
                pending_clarifications[key] = value

        handle_file_upload("questions", output_path, json.dumps(pending_clarifications).encode('utf-8'))
        print(f"\nCompliance clarifications pending. Details saved to '{output_path}'.")
        return pending_clarifications

    except Exception as e:
        print(f"Error processing compliance assistant: {str(e)}")
        return {}

def generate_extraction_prompt(file_name, analysis, custom_rule):
    return f"""
Ad image file name: {file_name}
Summary: {analysis}
Rule: {custom_rule}

Strictly identify and extract only the guidelines from the custom rule that are definitively potentially violated as indicated by the analysis summary.
Return only the guidelines using their exact original wording and numbering without any modifications.
If no guideline is identified as potentially violated, respond with "No potential violations."
"""

def generate_clarification_prompt(file_name, analysis, custom_rule, extracted_violations):
    return (
        f"Ad image file name: {file_name}\n"
        f"Summary: {analysis}\n"
        f"Extracted potential violations: {extracted_violations}\n\n"
        "If no potential violations are present, strictly respond with 'No questions.'\n"
        "Based solely on the above extracted potential violations, generate only the clarifying questions that are absolutely necessary to resolve any ambiguities regarding compliance or non-compliance. "
        "Do not include any additional commentary."
    )

async def model_assistant_acreate(prompt_text):
    try:
        response = await client.chat.completions.create(
            model="o3-mini",
            messages=[{"role": "user", "content": prompt_text}]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in model_assistant_acreate: {e}")
        return None

async def get_extracted_violations_image(file_name, analysis, rule):
    prompt_text = generate_extraction_prompt(file_name, analysis, rule)
    print(f"\nProcessing Extraction for image:\n{prompt_text}\n")
    extraction_response = await model_assistant_acreate(prompt_text)
    return extraction_response

async def get_clarifications_image(file_name, analysis, rule, extracted_violations):
    prompt_text = generate_clarification_prompt(file_name, analysis, rule, extracted_violations)
    print(f"\nProcessing Clarifications for image:\n{prompt_text}\n")
    clarifying_response = await model_assistant_acreate(prompt_text)
    return clarifying_response

async def process_rule(idx, file_name, analysis, rule):
    extracted_violations = await get_extracted_violations_image(file_name, analysis, rule)
    clarifying_response = await get_clarifications_image(file_name, analysis, rule, extracted_violations)
    display_name = fixed_display_names[idx - 1] if idx <= len(fixed_display_names) else f"Rule {idx}"
    print(f"Completed processing rule {idx}")
    return (f"Rule {idx}", {
        "rule": extracted_violations,
        "clarifying_questions": clarifying_response,
        "displayName": display_name,
        "file_name": file_name
    })

def process_image_file(file_key: str) -> dict:
    """
    Processes an image by downloading it from the Supabase 'uploads' bucket using the given file key,
    analyzing the image, saving the analysis JSON to the Supabase 'analysis' bucket,
    and initiating compliance clarifications.
    Returns a dictionary containing the analysis result and pending clarifications.
    """
    try:
        # Generate a UUID for the results file
        base = os.path.splitext(file_key)[0]
        result_filename = f"{base}_analysis.json"
        
        print(f"Analyzing image from Supabase key: {file_key}")
        analysis_result = analyze_image(file_key)

        # Save analysis result to Supabase "results" bucket
        result_data = {"file_name": file_key, "analysis": analysis_result}
        handle_file_upload("analysis", result_filename, json.dumps(result_data).encode('utf-8'))
        print(f"Analysis completed. Results saved to Supabase: {result_filename}")

        # Process compliance clarifications
        pending_clarifications = asyncio.run(process_compliance_assistant(result_filename, f"{base}_questions.json", compliance_rules))
        return {"analysis_result": analysis_result, "pending_clarifications": pending_clarifications, "file_name": file_key}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}
