import os
from dotenv import load_dotenv
from openai import OpenAI
import logging

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def model_image(prompt, instruction, base64_image):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # Updated to vision-capable model
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
                    ],
                }
            ],
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in model_image: {str(e)}")
        return None

def model_assistant(prompt_text, thread, assit_id = "asst_ypetLxffTCSLJ4bZkF1XjChv"):
    try:
        if thread.id is None:
            thread = client.beta.threads.create()
        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=prompt_text
        )
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=assit_id
        )
        if run.status == 'completed':
            messages = client.beta.threads.messages.list(thread_id=thread.id)
            clarifying_response = messages.data[0].content[0].text.value
            return thread.id, clarifying_response
        else:
            logging.warning(f"Run status: {run.status}")
            return None, None
    except Exception as e:
        logging.error(f"Error in model_assistant: {str(e)}")
        return None, None

def model_text(prompt_text, instructions):
    try:
        final_response = client.chat.completions.create(
            model="o3-mini",  # Updated to valid model name
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt_text}
            ]
        )
        return final_response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error in model_text: {str(e)}")
        return None
