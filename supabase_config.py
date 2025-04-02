import uuid
from supabase import create_client, Client

# Initialize Supabase client
supabase_url = "https://emeaiacgnqsamuztsscr.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImVtZWFpYWNnbnFzYW11enRzc2NyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDIwMjQ0NzMsImV4cCI6MjA1NzYwMDQ3M30.-tqu5XT0fGBlIQORxaf3iM455Jyrnl2ghBmb9ZuBVFI"
supabase: Client = create_client(supabase_url, supabase_key)

def upload_file(file_data, bucket_name="answers", file_path="answers/image_answers.json"):
    try:
        # Generate a unique filename if needed
        unique_filename = f"image_answers_{uuid.uuid4()}.json"
        # Upload the file with upsert option to overwrite if it exists
        supabase.storage.from_(bucket_name).upload(file_path, file_data, {"upsert": True})
        return {"message": "File uploaded successfully"}
    except Exception as e:
        if "Duplicate" in str(e):
            return {"error": "Resource already exists. Use a unique name or overwrite the existing resource."}
        else:
            return {"error": f"An error occurred: {str(e)}"}


file_key = f"frames/{uuid.uuid4()}.jpg"
try:
    supabase.storage.from_("frames").upload(file_key, b"test data", {"upsert": "true"})
    print("Upload succeeded:", file_key)
except Exception as e:
    print("Upload failed:", e)
