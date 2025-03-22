import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def model_image(prompt, insturction, base64_image):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"{prompt}",
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

def model_assistant(prompt_text, thread, assit_id = "asst_BHWO2gzHgASMC88S5x9RzVSJ"):
        print(prompt_text)
        print(assit_id)
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
            print(f"Run status: {run}")
            return None, None

def model_text(prompt_text, instructions):
    final_response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": prompt_text}
            ],
        max_tokens=1000,
        temperature=0.5
    )
    
    final_compliance = final_response.choices[0].message.content
    return final_compliance
