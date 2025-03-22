import json
from backend.models import model_text,model_assistant
import os

def perform_compliance_check(clarifications, answers, analysis):
    """
    For each rule in clarifications, create a prompt and send a request to the GPT-4 API.
    Returns a list of individual compliance responses.
    """
    compliance_results = []
    for rule_key, clarData in clarifications.items():
        rule = clarData.get("rule", "")
        #answer = answers.get(rule_key, "No answer provided.")
        
        rule_prompt = f"""
        You are a compliance expert.
        You must only list a violation if you are 100% certain that the user's response and/or the advertisement
        definitely breaks the Code for the rule: "{rule}".
        If you are not fully sure that there is a definite violation, do not list any violation or code reference.
        
        User's Response:
        {answers}

        analysis of ad:
        {analysis}
        if viloations has happened then follow the below structure: 
        **Determine the Code Violated:**
        - Identify the exact chapter and section from the Code/guidelines that is being violated.
        - Explain in detail how this rule is being violated.

        **Output Format:**
        - Code Violated: [Headings/subheading/chapter/sub chapter/pointers]
        - How the Code Was Violated: [Detailed Explanation]
        if no violation has happened:
        **NO violation**
        
        """
        instructions = "You are a compliance expert providing structured analysis."
        compliance_results.append(model_text(rule_prompt, instructions).strip())
        
    return compliance_results

def generate_final_report(compliance_results):
    """
    Combines all individual compliance results into one final report by sending a consolidated prompt to GPT-4.
    """
    compliance_summary = "\n\n".join(compliance_results)
    final_prompt = f"""
            You are an experienced compliance expert. Below are individual compliance analyses for various rules. Each analysis includes:
            - **Code Violated:** [Chapter and Section of the ASCI Code or ]
            - **Explanation:** Detailed information on how the code was violated
             or 
             ** NO violation**
            Your task is to merge these individual analyses into one consolidated final compliance report. Please follow these guidelines:
            1. **Merge Without Significant Rewording:** Combine the analyses while preserving the original wording as much as possible.
            2. **Structured Output:** The final report should list each unique "Code Violated" entry along with its corresponding explanation.
            3. **Markdown Format:** Ensure the final output is formatted in Markdown.
            4. Ignore if it's **No Violation**
            Here is the consolidated individual compliance analysis:
            {compliance_summary}
        """

    # final_compliance = model_text(final_prompt, "")
    thread = type('Thread', (), {})()
    thread.id = None
    thread2, final_compliance = model_assistant(final_prompt,thread,"asst_RPTL9D6vrDYTgwnUFVwE5CRc")
    final_compliance = final_compliance.strip()
    return final_compliance

def read_answers(file_name):
    """
    Opens the JSON file in the 'answers' folder and returns the value of the 'answer' key.
    
    :param file_name: The base name used when creating the file (without '_answers.json')
    :return: The text stored in the 'answer' key, or None if not found
    """
    answers_folder = "answers"
    json_filename = os.path.join(answers_folder, f"{file_name}_answers.json")
    
    if not os.path.exists(json_filename):
        print(f"File {json_filename} does not exist.")
        return None

    with open(json_filename, "r", encoding="utf-8") as f:
        data = json.load(f)
        print(f"File {json_filename} does exist and read")
        return data.get("answer")