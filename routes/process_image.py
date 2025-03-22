from PIL import Image
import os
import json
import base64
from dotenv import load_dotenv
from openai import OpenAI
import io
import re
from backend.models import model_image , model_assistant
# Path for saving the final compliance report


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


# Prompt to send to the API for initial analysis
prompt = (
    "Analyze the provided video or image and create a detailed description focusing on the following aspects to assist in compliance verification:\n\n"
    "Content Depiction: Describe all visible and audible elements in the media, including the primary subject(s), their actions, and interactions. Specify the age range, gender, attire, and other identifying details of the individuals depicted.\n"
    "Visual Elements: Describe the layout, background, setting, text overlays, graphics, or animations present. Include information about fonts, colors, and placement of any written disclaimers or promotional messages.\n"
    "Messaging and Claims: Extract and describe any explicit or implicit messaging related to the advertised product, service, or game. Include claims, promises, or calls-to-action conveyed in the content.\n"
    "Promotions and Offers: Note details of any promotions, bonuses, or offers mentioned, including their phrasing and visual/audio presentation.\n"
    "Disclaimers: Document the presence, placement, formatting, and duration of any disclaimers, both in visual and audio forms. Note their compliance with formatting rules, including size, clarity, and legibility.\n"
    "Behavior and Practices: Describe any behaviors or practices depicted, particularly those involving risks, minors, or potentially hazardous actions.\n"
    "Cultural and Social Indicators: Note any elements that may relate to race, religion, gender, or other societal factors, ensuring a neutral and factual account of how these are represented.\n"
    "Do not perform any legal or compliance analysis. The output should only describe the content objectively and comprehensively for subsequent evaluation."
)

def compress_resize_encode_image(image_path, max_size=(1024, 1024), quality=85):
    with Image.open(image_path) as img:
        if img.mode == "RGBA":
            img = img.convert("RGB")  # Handle transparency by converting to RGB
        img.thumbnail(max_size)
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", optimize=True, quality=quality)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

def analyze_image(image_path):
    base64_image = compress_resize_encode_image(image_path)
    response = model_image(prompt,"",base64_image)
    return response.replace("\n", "").replace("**", "").replace('"', "")

def generate_compliance_prompt(file_name, analysis, custom_rule):
    return f"""
        Ad image file name: {file_name}
        Summary: {analysis}
        Rule: {custom_rule}

        Please provide clarifying questions only if there are genuine ambiguities or missing details strictly related to any potential grey areas 
        If there are no such ambiguities, respond with "No questions."

        (No compliance evaluation or detailed assessment is needed—only questions, if any.)
"""


# Global variable to store the single thread ID for image compliance clarifications
global_compliance_thread_id_image = None

def get_clarifications_image(file_name, analysis, rule):
   
    global global_compliance_thread_id_image 
    prompt_text = generate_compliance_prompt(file_name, analysis, rule)
    print(f"\nProcessing Rule for image:\nInitial Prompt:\n{prompt_text}\n")
    
    # Reuse an existing thread if available; otherwise, create a new thread
    if global_compliance_thread_id_image is None:
        thread = client.beta.threads.create()
        global_compliance_thread_id_image = thread.id
    else:
        # Create a dummy object with the existing thread ID for consistency
        thread = type('Thread', (), {})()
        thread.id = global_compliance_thread_id_image
    thread_id, clarifying_response = model_assistant(prompt_text, thread)
    return thread_id, clarifying_response


def process_compliance_assistant(input_path, output_path, rules):
    """
    Processes each compliance rule by starting a thread to get clarifying questions.
    Instead of using interactive input, it builds and returns a dictionary of pending clarifications.
    Uses a hardcoded list of eight display names for the fixed rules.
    """
    global global_compliance_thread_id_image
    try:
        with open(input_path, "r") as file:
            data = json.load(file)
       
        file_name = data.get("file_name")
        analysis = data.get("analysis")

        if not file_name or not analysis:
            print("Missing 'file_name' or 'analysis' in input JSON.")
            return {}

        # Hardcoded display names for eight fixed rules.
        fixed_display_names = [
            "Junglee Internal Guidlines",
            "Trademark Guidelines",
            "Truthfulness & Honesty",
            "Non-offensiveness",
            "Avoidance of Harmful Content",
            "Fair Competition",
            "ASCI Guidelines",
            "Influencer Advertising",
            "Misleading_advertisment"
        ]

        pending_clarifications = {}
        for idx, rule in enumerate(rules, start=1):
            thread_id, clarifying_response = get_clarifications_image(file_name, analysis, rule)
            if not thread_id:
                print(f"Skipping Rule {idx} due to error in getting clarifications.")
                continue

            # Use the fixed display name for each rule (if available)
            display_name = fixed_display_names[idx - 1] if idx <= len(fixed_display_names) else f"Rule {idx}"

            pending_clarifications[f"Rule {idx}"] = {
                "thread_id": thread_id,
                "rule": rule,
                "displayName": display_name,
                "clarifying_questions": clarifying_response,
                "file_name": file_name
            }
        print(global_compliance_thread_id_image)
        global_compliance_thread_id_image = None
        print(global_compliance_thread_id_image)
        with open(output_path, "w") as output_file:
            json.dump(pending_clarifications, output_file, indent=4)
        print(f"\nCompliance clarifications pending. Details saved to '{output_path}'.")
        return pending_clarifications

    except Exception as e:
        print(f"Error processing compliance assistant: {str(e)}")
        return {}


def process_image_file(file_path):
    """
    Processes the image:
      1. Analyzes the image and saves the analysis result.
      2. Initiates the compliance assistant for all rules and returns the pending clarifications.
    """
    try:
        
        results_dir = "results"
        questions_dir = "questions"
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(questions_dir, exist_ok=True)
        file_name = os.path.basename(file_path).split('.')[0]
        
        image_description_path = os.path.join(results_dir, f"{file_name}_analysis.json")
        image_compliance_report = os.path.join(questions_dir , f"{file_name}_questions.json")
        print(f"Analyzing image: {file_path}")
        analysis_result = analyze_image(file_path)
        with open(image_description_path, "w") as json_file:
            json.dump({"file_name": file_path, "analysis": analysis_result}, json_file, indent=4)
        print(f"Analysis completed. Results saved to {image_description_path}")
        pending_clarifications = process_compliance_assistant(image_description_path, image_compliance_report, compliance_rules)
        return {"analysis_result": analysis_result, "pending_clarifications": pending_clarifications, "file_name": file_path}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}


# Combine all the individual rules into one list
compliance_rules = [
    Junglee_Internal_Guidlines,
    rule_trademark,
    rule_truthful_honest,
    rule_non_offensive,
    rule_against_harmful,
    rule_fair_competition,
    rule_asci_guidelines,
    rule_influencer_advertising,
    rule_misleading_advertisment
]

