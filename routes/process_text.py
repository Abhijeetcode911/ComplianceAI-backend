import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from backend.models import model_assistant

load_dotenv()
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
output_path = "output.json"


# Determine the path to the JSON file (assuming it's in the same directory)
current_dir = os.path.dirname(__file__)
rules_path = os.path.join(current_dir, 'rules.json')

# Load the JSON file into a dictionary
with open(rules_path, 'r') as file:
    rules = json.load(file)

# Option 1: Create individual variables for each rule
Junglee_Internal_Guidlines= rules.get("Junglee_Internal_Guidlines")
rule_trademark = rules.get("rule_trademark")
rule_truthful_honest = rules.get("rule_truthful_honest")
rule_non_offensive = rules.get("rule_non_offensive")
rule_against_harmful = rules.get("rule_against_harmful")
rule_fair_competition = rules.get("rule_fair_competition")
rule_asci_guidelines = rules.get("rule_asci_guidelines")
rule_influencer_advertising = rules.get("rule_influencer_advertising")
rule_misleading_advertisment = rules.get("rule_misleading_advertisment")

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

def generate_compliance_prompt(file_name, analysis, custom_rule):
    return f"""
        Ad  file name: {file_name}
        Summary: {analysis}
        Rule: {custom_rule}

        Please provide up to two clarifying questions only if there are genuine ambiguities or missing details strictly related to any potential grey areas. 
        If there are no such ambiguities, respond with "No questions."

        (No compliance evaluation or detailed assessment is needed—only questions, if any.)
"""


# Global variable to store the single thread ID for image compliance clarifications
global_compliance_thread_id_text = None

def get_clarifications_text(file_name, analysis, rule):
    global global_compliance_thread_id_text

    prompt_text = generate_compliance_prompt(file_name, analysis, rule)
    print(f"\nProcessing Rule for text:\nInitial Prompt:\n{prompt_text}\n")
    
    # Reuse an existing thread if available; otherwise, create a new thread
    if global_compliance_thread_id_text is None:
        thread = client.beta.threads.create()
        global_compliance_thread_id_image = thread.id
    else:
        # Create a dummy object with the existing thread ID for consistency
        thread = type('Thread', (), {})()
        thread.id = global_compliance_thread_id_text
    thread_id, clarifying_response = model_assistant(prompt_text, thread)
    return thread_id, clarifying_response


    
def process_compliance_assistant(input_path, output_path, rules):
    """
    Processes each compliance rule by starting a thread to get clarifying questions.
    Instead of using interactive input, it builds and returns a dictionary of pending clarifications.
    Uses a hardcoded list of eight display names for the fixed rules.
    """
    global global_compliance_thread_id_text
    try:
        with open(input_path, "r") as file:
            data = json.load(file)

        file_name = data.get("image_file")
        analysis = data.get("analysis")

        if not file_name or not analysis:
            print("Missing 'image_file' or 'analysis' in input JSON.")
            return {}

        # Hardcoded display names for eight fixed rules.
        fixed_display_names = [
            "Junglee_Internal_Guidlines",
            "Trademark Guidelines",
            "Truthfulness & Honesty",
            "Non-offensiveness",
            "Avoidance of Harmful Content",
            "Fair Competition",
            "ASCI Guidelines",
            "Influencer Advertising",
            "misleading_advertisment"
        ]

        pending_clarifications = {}
        for idx, rule in enumerate(rules, start=1):
            thread_id, clarifying_response = get_clarifications_text(file_name, analysis, rule)
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
                "image_file": file_name
            }
        with open(output_path, "w") as output_file:
            json.dump(pending_clarifications, output_file, indent=4)
        print(f"\nCompliance clarifications pending. Details saved to '{output_path}'.")
        global_compliance_thread_id_text = None
        return pending_clarifications

    except Exception as e:
        print(f"Error processing compliance assistant: {str(e)}")
        return {}
    

def process_text_file(file_path):
    """
    Processes a text file:
      1. Reads text content from the provided .txt file.
      2. Processes the text content using the compliance assistant to generate pending clarifications.
      3. Returns a dictionary with the analysis result (the text content) and pending clarifications.
    """
    try:
        # Ensure the file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Read the text file content
        with open(file_path, "r", encoding="utf-8") as file:
            text_content = file.read()
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        pending_clarifications = process_compliance_assistant(file_name,text_content, compliance_rules)
        
        # In this example, we use the text content as the analysis result.
        analysis_result = text_content
        
        return {"analysis_result": analysis_result, "pending_clarifications": pending_clarifications, "file_name": file_path}
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"error": str(e)}

