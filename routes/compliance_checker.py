import json
import os
import requests
from datetime import datetime
import uuid
from models import model_assistant, model_text
from supabase_config import supabase  # Import Supabase client
import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()
aclient = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def handle_file_upload(bucket_name: str, file_path: str, file_data: bytes) -> None:
    """
    Handles file upload to Supabase storage, renaming existing files if necessary.

    Args:
        bucket_name (str): The name of the Supabase bucket.
        file_path (str): The path/name of the file to upload.
        file_data (bytes): The file data to upload.
    """
    try:
        # Check if a file with the same name already exists
        supabase.storage.from_(bucket_name).download(file_path)
        # Rename the existing file by appending a timestamp or UUID
        base_name, ext = os.path.splitext(file_path)
        new_file_path = f"{base_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:6]}{ext}"
        supabase.storage.from_(bucket_name).move(file_path, new_file_path)
        print(f"Renamed existing file to: {new_file_path}")
    except Exception:
        print(f"No existing file found: {file_path}")

    # Upload the new file
    supabase.storage.from_(bucket_name).upload(file_path, file_data)
    print(f"Uploaded new file: {file_path}")
    
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
    
async def perform_compliance_check(clarifications, answers, analysis):
    """
    For each rule in clarifications, create a prompt and send a request to the GPT-4 API concurrently.
    Returns a list of individual compliance responses.
    """
    tasks = []
    for rule_key, clarData in clarifications.items():
        rule = clarData.get("rule", "")
        rule_prompt = (
            f"You are a legal and compliance expert evaluating an advertisement for possible violations of the following rule:\n\n"
            f"\"{rule}\"\n\n"
            f"User's Response:\n"
            f"{answers}\n\n"
            f"Ad Analysis:\n"
            f"{analysis}\n\n"
            f"Your task:\n"
            f"- Analyze the user's response and the ad analysis in the context of the rule above.\n"
            f"- Only report a violation if you are 100% certain that the content clearly violates the rule. However, if there is any uncertainty regarding compliance with any legal requirement—including but not limited to intellectual property, advertising standards, consumer protection, or any other law—you must assume non-compliance.\n"
            f"- Do not consider the advertisement fully compliant if there is ambiguity regarding its adherence to any legal guidelines or licensing requirements.\n\n"
            f"### Output Format:\n\n"
            f"If a violation is found, respond with:\n\n"
            f"**Code Violated:**\n"
            f"[Clearly state the name of the rule, guideline, or policy. If available, include the exact wording of the rule.]\n\n"
            f"**How the Code Was Violated:**\n"
            f"1.1. Description of the Violation:\n"
            f"    - [Describe what is shown, claimed, or implied in the advertisement that violates the rule.]\n\n"
            f"1.2. Conflict with the Guideline:\n"
            f"    - [Explain how the content contradicts the rule and what the potential consequence or risk is.]\n\n"
            f"---\n\n"
            f"If no violation is found, respond exactly like this:\n\n"
            f"**Code Violated:** None\n"
            f"**How the Code Was Violated:** No violation detected.\n"
            f"The advertisement is fully compliant with the relevant codes, laws, and guidelines based on the current analysis.\n\n"
            f"---\n\n"
        )
        # Create a task for each rule prompt
        tasks.append(model_assistant_acreate(rule_prompt))
    
    # Await all tasks concurrently and return their results
    compliance_results = await asyncio.gather(*tasks)
    return compliance_results

def generate_final_report(compliance_results: list) -> str:
    """
    Combines all individual compliance results into one final report by sending a consolidated prompt to GPT-4.
    """
    compliance_summary = "\n\n".join(compliance_results)
    final_prompt = f"""
        You are a senior compliance expert. Below are individual compliance analyses of an advertisement against various rules and guidelines.

        Each entry includes:
        - **Code Violated:** The name or label of the applicable rule or guideline (may or may not follow chapter/section format)
        - **How the Code Was Violated:** A structured explanation of how the advertisement breaches that rule

        Your task is to generate a **final consolidated compliance report**.

        Please follow these instructions:
        1. ✅ **Include only violations** – Skip any entries that say "Code Violated: None".
        2. ♻️ **Merge duplicate violations** – If the same "Code Violated" appears multiple times:
            - Combine all their reasoning under the same heading.
            - If the explanations differ, list all of them clearly as subpoints (1, 2, 3, etc.) under "How the Code Was Violated".
        3. ✍️ **Do not rewrite or reword significantly** – Preserve the original analysis wording.
        4. 🧱 **Structured Output** – Clearly separate each violation using Markdown formatting. Each should include:
            - The **Code Violated** header
            - The **How the Code Was Violated** section with subpoints (1, 2, etc.)
        5. 📄 **Markdown Format** – Use clear headings, lists, and spacing for readability.

        Here is the input to consolidate:
        {compliance_summary}
    """

    # final_compliance = model_text(final_prompt, "")
    thread = type('Thread', (), {})()
    thread.id = None
    thread2, final_compliance = model_assistant(final_prompt,thread,"asst_Ye4cucsgqvY6Cq7QY52ZQ97e")
    final_compliance = final_compliance
    return final_compliance.strip() if final_compliance else ""

def read_answers(file_name: str) -> str:
    """
    Fetches the JSON file from Supabase storage and returns the value of the 'answer' key.

    Args:
        file_name (str): The UUID-based base name used when creating the file (without '_answers.json').

    Returns:
        str: The text stored in the 'answer' key, or None if not found.
    """
    bucket_name = "answers"
    name = os.path.splitext(os.path.basename(file_name))[0]
    supabase_path = f"{name}_answers.json"

    try:
        url = supabase.storage.from_(bucket_name).get_public_url(supabase_path)
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        print(f"File {supabase_path} fetched successfully from Supabase.")
        return data.get("answer")
    except Exception as e:
        print(f"Error fetching file {supabase_path} from Supabase: {e}")
        return None


