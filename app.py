# ==================================================
# AI DETECTOR SERVICE
# Port: 5005
# Route: POST /api/analyze
# ==================================================

import os
import sys
import nltk
from flask import Flask, request, jsonify
from flask_cors import CORS
import jwt
from datetime import datetime
from collections import defaultdict
from dotenv import load_dotenv
from transformers import pipeline

# Load .env from parent backend directory
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_BASE_DIR, ".env"))

app = Flask(__name__)
CORS(app)

JWT_SECRET = os.getenv("JWT_SECRET")

# --------------------------
# AI usage limit tracking per IP
# --------------------------
ai_trials = defaultdict(lambda: {"count": 0, "date": datetime.today().date()})

# --------------------------
# NLTK resource downloader
# --------------------------
def download_nltk_resources():
    resources = [
        'punkt', 'punkt_tab', 'averaged_perceptron_tagger',
        'averaged_perceptron_tagger_eng', 'wordnet', 'omw-1.4', 'vader_lexicon'
    ]
    for res in resources:
        try:
            nltk.data.find(res)
        except Exception:
            nltk.download(res, quiet=True)

# --------------------------
# Load RoBERTa AI detection model
# --------------------------
MODEL_NAME = "JinalShah2002/distilbert-detector"
print("Loading AI detection model (this may take a moment on first boot)...")
try:
    classifier = pipeline("text-classification", model=MODEL_NAME, truncation=True, max_length=512)
    print("[SUCCESS] AI Detector model loaded!")
except Exception as e:
    print(f"[ERROR] Failed to load AI detection model: {e}")
    classifier = None

# --------------------------
# Rate limit helper
# --------------------------
def check_ai_limit():
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
            return True, None
        except Exception:
            pass

    ip = request.remote_addr
    today = datetime.today().date()
    if ai_trials[ip]["date"] < today:
        ai_trials[ip] = {"count": 0, "date": today}

    if ai_trials[ip]["count"] >= 8:
        return False, jsonify({
            "success": False,
            "error": "FREE_LIMIT_REACHED",
            "message": "You have reached your free daily limit of 8 uses. Please Log In or Sign Up to continue."
        })

    ai_trials[ip]["count"] += 1
    return True, None

# --------------------------
# Core analysis function
# --------------------------
def analyze_text_ai(text):
    if not text or not text.strip() or not classifier:
        return None

    full_result = classifier(text)[0]

    if full_result['label'] in ['Fake', 'ChatGPT']:
        ai_prob = full_result['score'] * 100
    else:
        ai_prob = (1.0 - full_result['score']) * 100

    final_ai_score = max(0, min(100, int(ai_prob)))

    if final_ai_score >= 80:
        mood = "AI Based"
    elif final_ai_score >= 60:
        mood = "AI Based & AI Refined"
    elif final_ai_score >= 40:
        mood = "Human Written & AI Refined"
    else:
        mood = "Human Written"

    download_nltk_resources()
    sentences = nltk.sent_tokenize(text)
    segments = []

    for s in sentences:
        if len(s.split()) < 3:
            s_score = final_ai_score
        else:
            s_result = classifier(s)[0]
            if s_result['label'] in ['Fake', 'ChatGPT']:
                s_score = s_result['score'] * 100
            else:
                s_score = (1.0 - s_result['score']) * 100

        if s_score >= 75:
            sType = "ai"
        elif s_score >= 55:
            sType = "ai-refined"
        elif s_score >= 35:
            sType = "human-ai"
        else:
            sType = "human"

        segments.append({"text": s.strip(), "type": sType})

    return {
        "aiPercentage": final_ai_score,
        "mood": mood,
        "segments": segments
    }

# --------------------------
# Route: POST /api/analyze
# --------------------------
@app.route('/api/analyze', methods=['POST'])
def handle_ai_analyze():
    allowed, err_response = check_ai_limit()
    if not allowed:
        return err_response, 429

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    if not classifier:
        return jsonify({"error": "AI Detector engine is still warming up (loading neural weights). Please try again in a few seconds."}), 503

    text = data['text']
    result = analyze_text_ai(text)

    if not result:
        return jsonify({"error": "Analysis failed. Please check the input text and try again."}), 500

    return jsonify(result)

# --------------------------
# Health check
# --------------------------
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "service": "ai-detector", "model_loaded": classifier is not None})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    app.run(host="0.0.0.0", port=port, debug=True)
