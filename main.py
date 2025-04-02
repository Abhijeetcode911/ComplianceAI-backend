from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from routes.process_image import process_image_file  
from routes.process_video import process_video_file
from routes.compliance_checker import perform_compliance_check, generate_final_report, read_answers
import json
import re
from PIL import Image
import PyPDF2
import docx
import pytesseract
from routes.visual import visual_law 
from models import model_text
from datetime import datetime
import uuid
import logging
import asyncio
from supabase_config import supabase
from flask_executor import Executor
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
executor = Executor(app)
tasks = {}
# Allowed file types
ALLOWED_EXTENSIONS = {"txt", "png", "mp4", "jpeg", "jpg"}

@app.after_request
def add_cors_headers(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["POST"])
def upload_file():
    print("1")
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    print("2")
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    print("3")
    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        file_extension = original_filename.rsplit(".", 1)[1].lower()
        # Generate a unique filename using UUID
        unique_filename = f"{uuid.uuid4()}{os.path.splitext(original_filename)[1]}"
        bucket_name = "uploads"
        file_path = f"{unique_filename}"

        # Check if a file with the same name exists in Supabase (unlikely with UUID)
        try:
            supabase.storage.from_(bucket_name).download(file_path)
            base_name, ext = os.path.splitext(file_path)
            new_file_path = f"{base_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}{ext}"
            supabase.storage.from_(bucket_name).move(file_path, new_file_path)
            logging.info(f"Renamed existing file to: {new_file_path}")
        except Exception:
            logging.info(f"No existing file found for: {file_path}")

        # Upload file to Supabase
        try:
            file.seek(0)
            supabase.storage.from_(bucket_name).upload(file_path, file.read())
        except Exception as e:
            logging.error(f"Failed to upload file to Supabase: {str(e)}")
            return jsonify({"error": f"Failed to upload file: {str(e)}"}), 500

        # Process the file based on its type
        try:
            if file_extension in ["png", "jpeg", "jpg"]:
                result = process_image_file(file_path)
            elif file_extension == "mp4":
                result = process_video_file(file_path)
                if "file_name" in result:
                    file_path = result.get("file_name")
                logging.info("Uploaded video")
            else:
                return jsonify({"error": "Unsupported file type"}), 400
        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            return jsonify({"error": f"Error processing file: {str(e)}"}), 500

        logging.info(f"File processed successfully: {file_path}")
        return jsonify({"message": "File processed successfully", "result": result, "file_name": file_path}), 200
    print("4")
    return jsonify({"error": "Invalid file type"}), 400

@app.route('/submit_clarification', methods=['POST'])
def submit_clarification():
    try:
        rule = request.form.get("rule")
        file_name = request.form.get("file_name")

        if not rule or not file_name:
            return jsonify({"error": "Missing rule, or file_name"}), 400

        # Extract content from the uploaded file or from user input text
        extracted_text = ""
        if "user_input" in request.files:
            uploaded_file = request.files["user_input"]
            ext = uploaded_file.filename.split('.')[-1].lower()

            if ext in ['png', 'jpeg', 'jpg']:
                image = Image.open(uploaded_file.stream)
                extracted_text = pytesseract.image_to_string(image)
            elif ext == "pdf":
                pdf_reader = PyPDF2.PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                extracted_text = text
            elif ext == "docx":
                doc = docx.Document(uploaded_file)
                text = ""
                for para in doc.paragraphs:
                    text += para.text + "\n"
                extracted_text = text
            elif ext == "txt":
                extracted_text = uploaded_file.read().decode("utf-8")
            else:
                return jsonify({"error": "Unsupported file type"}), 400
        else:
            extracted_text = request.form.get("user_input")
            if not extracted_text:
                return jsonify({"error": "No user input provided"}), 400

        bucket_name = "answers"
        base_file_name, _ = os.path.splitext(os.path.basename(file_name))
        json_filename = f"{base_file_name}_answers.json"

        try:
            existing_data = supabase.storage.from_(bucket_name).download(json_filename)
            data = json.loads(existing_data.decode("utf-8"))
            if not isinstance(data, list):
                data = [data]
        except Exception:
            data = []

        new_answer = {
            "rule": rule,
            "extracted_text": extracted_text.strip(),
            "timestamp": datetime.now().isoformat()
        }
        data.append(new_answer)

        try:
            supabase.storage.from_(bucket_name).upload(
                json_filename,
                json.dumps(data, indent=2).encode('utf-8'),
                {"upsert": "true"}
            )
        except Exception as e:
            logging.error(f"Error saving data to Supabase: {str(e)}")
            return jsonify({"error": f"Error saving data: {str(e)}"}), 500

        logging.info(f"Clarification submitted successfully for file: {file_name}")
        return jsonify({
            "final_response": f"Properties extracted: {extracted_text}",
            "extracted_text": extracted_text
        }), 200

    except Exception as e:
        logging.error(f"Error in submit_clarification: {str(e)}")
        return jsonify({"error": str(e)}), 500

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
    
def process_final_compliance(clarifications, file_path):
    """
    This function contains your long-running final compliance analysis.
    It mirrors the logic from your Celery task but is executed in a background thread.
    """
    try:
        base_filename = os.path.basename(file_path)
        name, ext = os.path.splitext(base_filename)
        bucket_uploads = "uploads"
        bucket_answers = "answers"
        bucket_analysis = "analysis"
        bucket_results = "results"

        # Retrieve answers from Supabase 'answers' bucket
        json_filename = f"{name}_answers.json"
        try:
            existing_data = supabase.storage.from_(bucket_answers).download(json_filename)
            answers = json.loads(existing_data.decode("utf-8"))
            if not answers:
                return {"final_compliance": "", "status": "No answers found for the provided file"}
        except Exception as e:
            logging.error(f"Error fetching answers: {str(e)}")
            return {"final_compliance": "", "status": f"Failed to read answers: {str(e)}"}

        # Retrieve the uploaded file from Supabase 'uploads' bucket
        input_path = f"{name}{ext}"
        try:
            file_data = supabase.storage.from_(bucket_uploads).download(input_path)
        except Exception as e:
            logging.error(f"File not found in uploads: {str(e)}")
            return {"final_compliance": "", "status": f"File not found: {str(e)}"}

        compliance_results = []
        analysis = ""
        if ext.lower() in ['.png', '.jpeg', '.jpg', '.mp4']:
            # Execute asynchronous visual analysis (this will block the thread until complete)
            visual_result = run_async(visual_law(input_path))
            compliance_results.append(visual_result)
            # Try to retrieve pre-generated analysis if available
            json_path = f"{name}_analysis.json"
            try:
                json_data = supabase.storage.from_(bucket_analysis).download(json_path)
                analysis = json.loads(json_data).get("analysis", "No analysis provided.")
            except Exception:
                analysis = "No analysis provided."
        else:
            analysis = file_data.decode("utf-8")

        if analysis:
            rule_results = run_async(perform_compliance_check(clarifications, answers, analysis))
            compliance_results.extend(rule_results)

        final_compliance = generate_final_report(compliance_results)
        if not final_compliance:
            raise ValueError("final_compliance is empty or None")

        # Save the final compliance report to Supabase
        output_path = f"{name}_compliance_output.txt"
        try:
            supabase.storage.from_(bucket_results).upload(
                output_path,
                final_compliance.encode('utf-8'),
                {"upsert": "true"}
            )
        except Exception as e:
            logging.error(f"Failed to save compliance report: {str(e)}")
            return {"final_compliance": "", "status": f"Failed to save report: {str(e)}"}

        logging.info(f"Final compliance analysis completed for file: {file_path}")
        return {"final_compliance": final_compliance.replace("`", "").replace("markdown", ""), "status": "Completed"}
    except Exception as e:
        logging.error(f"Unhandled error in task: {str(e)}")
        return {"final_compliance": "", "status": str(e)}

@app.route('/final_compliance_analysis', methods=['POST'])
def final_compliance_analysis():
    try:
        data = request.get_json()
        clarifications = data.get("clarifications")
        file_path = data.get("file_name")

        if not clarifications or not file_path:
            return jsonify({"error": "Missing clarifications or file_name in payload"}), 400

        # Create a unique task ID and submit the background task
        task_id = str(uuid.uuid4())
        future = executor.submit(process_final_compliance, clarifications, file_path)
        tasks[task_id] = future
        return jsonify({"task_id": task_id}), 202
    except Exception as e:
        logging.error(f"Unhandled error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/task_status/<task_id>', methods=['GET'])
def task_status(task_id):
    future = tasks.get(task_id)
    if future is None:
        return jsonify({"error": "Task not found"}), 404

    if future.done():
        try:
            result = future.result()
        except Exception as e:
            result = {"error": str(e)}
        return jsonify({"state": "COMPLETED", "result": result})
    else:
        return jsonify({"state": "PENDING", "status": "Processing..."}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)