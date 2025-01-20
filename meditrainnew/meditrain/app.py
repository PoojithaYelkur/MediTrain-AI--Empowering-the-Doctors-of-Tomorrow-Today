from flask import Flask, request, jsonify, render_template
import json
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def clean_medical_data():
    """Clean medical dataset by removing duplicates based on symptoms and severity"""
    try:
        with open('medical_dataset.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Create a set to track unique combinations
        unique_entries = {}
        cleaned_conditions = []
        
        for condition in data['conditions']:
            # Sort symptoms to ensure consistent comparison
            sorted_symptoms = sorted(condition['symptoms'])
            # Create a unique key based on symptoms and severity
            key = (tuple(sorted_symptoms), condition['severity'])
            
            # Only keep the first occurrence of each unique combination
            if key not in unique_entries:
                unique_entries[key] = True
                cleaned_conditions.append(condition)
        
        # Update the data with cleaned conditions
        data['conditions'] = cleaned_conditions
        
        # Save the cleaned data back to file
        with open('medical_dataset.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)
            
        logger.info(f"Cleaned dataset. Reduced from {len(data.get('conditions', []))} to {len(cleaned_conditions)} conditions")
        return True
    except Exception as e:
        logger.error(f"Error cleaning medical data: {str(e)}")
        return False

# Initialize the medical conversation model
model_name = "GanjinZero/biobart-v2-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Clean the dataset when the app starts
clean_medical_data()

app = Flask(__name__)

def load_medical_data():
    """Load medical conditions from JSON file"""
    try:
        with open('medical_dataset.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Loaded {len(data.get('conditions', []))} medical conditions")
            return data
    except Exception as e:
        logger.error(f"Error loading medical data: {str(e)}")
        return {"conditions": []}

def find_matching_conditions(symptoms):
    """Find matching conditions based on symptoms"""
    symptoms_list = [s.strip().lower() for s in symptoms.replace(',', ' ').split()]
    matches = []
    
    for condition in medical_data.get("conditions", []):
        condition_symptoms = [s.lower() for s in condition["symptoms"]]
        matching_count = 0
        matched_symptoms = set()
        
        # Check each user symptom against condition symptoms
        for user_symptom in symptoms_list:
            for cond_symptom in condition_symptoms:
                if user_symptom in cond_symptom or cond_symptom in user_symptom:
                    matching_count += 1
                    matched_symptoms.add(cond_symptom)
        
        if matching_count > 0:
            # Calculate confidence based on both matching symptoms and total symptoms
            symptom_match_ratio = len(matched_symptoms) / len(condition["symptoms"])
            user_input_ratio = matching_count / len(symptoms_list)
            confidence = ((symptom_match_ratio + user_input_ratio) / 2) * 100
            
            matches.append({
                "name": condition["name"],
                "matching_symptoms": list(matched_symptoms),
                "all_symptoms": condition["symptoms"],
                "severity": condition["severity"],
                "suggestion": condition["suggestion"],
                "duration": condition["duration"],
                "confidence": round(confidence, 1)
            })
    
    return sorted(matches, key=lambda x: x["confidence"], reverse=True)

def format_response(matches):
    """Format the medical response"""
    if not matches:
        return "I couldn't find any specific conditions matching your symptoms. Please provide more details about your symptoms or consult a healthcare provider for proper evaluation."
    
    response = "🏥 Based on your symptoms, here are the potential conditions I've identified:\n\n"
    
    # Take top 3 most likely conditions
    for i, match in enumerate(matches[:3], 1):
        confidence_emoji = "🟢" if match['confidence'] > 75 else "🟡" if match['confidence'] > 50 else "🔴"
        
        response += f"{i}. {confidence_emoji} {match['name']} (Confidence: {match['confidence']}%)\n"
        response += f"   • Matching Symptoms: {', '.join(match['matching_symptoms'])}\n"
        response += f"   • Additional Symptoms to Watch: {', '.join(set(match['all_symptoms']) - set(match['matching_symptoms']))}\n"
        response += f"   • Severity: {match['severity'].upper()}\n"
        response += f"   • Duration: {match['duration']}\n"
        response += f"   • Suggestion: {match['suggestion']}\n\n"
    
    response += "⚠️ IMPORTANT: This is not a medical diagnosis. Please consult a healthcare provider for proper evaluation and treatment."
    
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({"response": "Please describe your symptoms."})
        
        # Find matching conditions
        matches = find_matching_conditions(user_message)
        
        # Format and return response
        response = format_response(matches)
        return jsonify({"response": response})
        
    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        return jsonify({
            "response": "Sorry, there was an error processing your request. Please try again."
        }), 500

# Load medical data at startup
medical_data = load_medical_data()

if __name__ == '__main__':
    app.run(debug=True, port=5000)
