from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from routes.process_text import process_text_file
from routes.process_image import process_image_file  
from routes.process_video import process_video_file
from routes.compliance_checker import perform_compliance_check, generate_final_report,read_answers
import json
import re
from flask import Flask, request, jsonify
from PIL import Image
import PyPDF2
import docx
import pytesseract
from routes.visual import visual_law 
from models import model_text
from datetime import datetime
import os
import json
from flask import request, jsonify
from PIL import Image
import pytesseract
import PyPDF2
import docx

app = Flask(__name__)
CORS(app)

# Allowed file types
ALLOWED_EXTENSIONS = {"txt","png", "mp4", "jpeg", "jpg"}

# Function to check if file extension is allowed
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["POST"])
def upload_file():
    
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_extension = filename.rsplit(".", 1)[1].lower()

        # Save the file
        file_path = os.path.join("uploads", filename)
        file.save(file_path)

        # Route file to the correct processing function
        if file_extension == "txt":
            result = process_text_file(file_path)
        elif file_extension in ["png", "jpeg", "jpg"]:
            result = process_image_file(file_path)
        elif file_extension == "mp4":
            result = process_video_file(file_path)
            print("uploaded video")
        else:
            return jsonify({"error": "Unsupported file type"}), 400

        # The result now includes the analysis plus pending clarifications (questions + thread IDs)
        print(result)
        return jsonify({"message": "File processed successfully", "result": result}), 200

    return jsonify({"error": "Invalid file type"}), 400
    

# ======================================================================
# these all those can run in local machine but not in the server
# ======================================================================

@app.route('/submit_clarification', methods=['POST'])
def submit_clarification():
    try:
        # Retrieve required fields from the form data
        thread_id = request.form.get("thread_id")
        rule = request.form.get("rule")
        file_name = request.form.get("file_name")
        
        if not thread_id or not rule or not file_name:
            return jsonify({"error": "Missing thread_id, rule, or file_name"}), 400

        # Extract content from the uploaded file or text input
        extracted_text = ""
        if "user_input" in request.files:
            uploaded_file = request.files["user_input"]
            ext = uploaded_file.filename.split('.')[-1].lower()
            
            if ext in ['png', 'jpeg', 'jpg']:
                # Use OCR to extract text from the image
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

        # Ensure the "answers" folder exists
        answers_folder = "answers"
        if not os.path.exists(answers_folder):
            os.makedirs(answers_folder)
        base_file_name, ext = os.path.splitext(os.path.basename(file_name))
        # Define the JSON file name in the answers folder
        json_filename = os.path.join(answers_folder, f"{base_file_name}_answers.json")

        # Load existing data if the file exists, otherwise start with a new dict
        if os.path.exists(json_filename):
            with open(json_filename, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}

        # Append to the "answer" key if it exists; otherwise, create it
        if "answer" in data:
            data["answer"] += " " + extracted_text.strip()
        else:
            data["answer"] = extracted_text.strip()

        # Write the updated data back to the JSON file
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        return jsonify({
            "final_response": f"Properties extracted: {extracted_text}",
            "extracted_text": extracted_text
        }), 200

    except Exception as e:
        print(str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/final_compliance_analysis', methods=['POST'])
def final_compliance_analysis():
    try:
        data = request.get_json()
        clarifications = data.get("clarifications")

        # Extract file_name from the payload; 
        file_path = data.get("file_name")
        base_filename = os.path.basename(file_path)  # returns "image.jpg"
        name, ext = os.path.splitext(base_filename)
        answers = read_answers(name)
        if not clarifications or not answers:
            return jsonify({"error": "Missing clarifications or answers in payload"}), 400
         # Reconstruct input path with the provided extension
        input_path = os.path.join("uploads",  base_filename)
        print(input_path)
        compliance_results = []
        analysis = ""
        ext_lower = ext.lower()
        
        if ext_lower == '.txt':
            # For .txt files, use only the text content as analysis.
            with open(input_path, "r", encoding="utf-8") as file:
                analysis = file.read()
        elif ext_lower in ['.png', '.jpeg', '.jpg', '.mp4']:
            # For image/video files, first run visual_law on the file.
            visual_result = visual_law(input_path)
            print(visual_result)
            compliance_results.append(visual_result)
            # Then load analysis from a corresponding JSON file (same base name, .json extension).
            json_path = os.path.join("results", f"{name}_analysis.json")
            if os.path.exists(json_path):
                with open(json_path, "r", encoding="utf-8") as file:
                    file_data = json.load(file)
                analysis = file_data.get("analysis", "No analysis provided.")
            else:
                analysis = "No analysis provided."
        else:
            # Fallback: read file as text.
            with open(input_path, "r", encoding="utf-8") as file:
                analysis = file.read()

        # Use the analysis (from text or JSON) to run the compliance check.
        if analysis:
            rule_results = perform_compliance_check(clarifications, answers, analysis)
            compliance_results.extend(rule_results)
        # Generate the final compliance report from the combined results.
        final_compliance = generate_final_report(compliance_results)
       
        # Save the final compliance report to a file.
        output_path = "compliance_output.txt"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_compliance)
        response_payload = {
            "message": "Final compliance analysis completed",
            "final_compliance": final_compliance.replace("`","").replace("markdown","")
        }
        return jsonify(response_payload), 200

    except Exception as e:
        print(str(e))
        return str(e), 500  # Return error as plain text if any issue occurs

if __name__ == "__main__":
    os.makedirs("uploads", exist_ok=True)
    app.run(debug=True, host="0.0.0.0" , port=8000)


