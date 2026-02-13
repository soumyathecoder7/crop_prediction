from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd
import pickle
import requests
import sqlite3
import os
#from dotenv import load_dotenv  # NEW

#load_dotenv()  # NEW: Load environment variables

app = Flask(__name__,template_folder='template1')

# --- NEW: Database Setup ---
def init_db():
    with sqlite3.connect('crop_data.db') as conn:
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS reports 
                     (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                      nitrogen REAL, phosphorus REAL, potassium REAL, 
                      temperature REAL, humidity REAL, ph REAL, rainfall REAL, 
                      prediction TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()

init_db()

# --- 1. Load Models ---
try:
    model = pickle.load(open("model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
except FileNotFoundError:
    class MockModel:
        def predict(self, data): return ["Rice"]
    class MockScaler:
        def transform(self, data): return data
    model = MockModel()
    scaler = MockScaler()

# --- 2. Routes ---
@app.route("/")
def home():
    return render_template("crop.html", prediction_text=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        N = float(request.form['Nitrogen'])
        P = float(request.form['Phosphorus'])
        K = float(request.form['Potassium'])
        temp = float(request.form['temperature'])
        hum = float(request.form['humidity'])
        ph = float(request.form['pH'])
        rain = float(request.form['rainfall'])

        features = [N, P, K, temp, hum, ph, rain]
        
        scaled = scaler.transform(np.array([features]))
        prediction = model.predict(scaled)[0]

        with sqlite3.connect('crop_data.db') as conn:
            c = conn.cursor()
            c.execute("INSERT INTO reports (nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall, prediction) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                      (N, P, K, temp, hum, ph, rain, prediction))
            conn.commit()

        return render_template("crop.html", prediction_text=f"Predicted Crop: {prediction}")
    except Exception as e:
        return render_template("crop.html", prediction_text=f"Error: {str(e)}")

@app.route("/report")
def report():
    try:
        with sqlite3.connect('crop_data.db') as conn:
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute("SELECT * FROM reports ORDER BY timestamp DESC")
            rows = c.fetchall()
        return render_template("report.html", reports=rows)
    except Exception as e:
        return f"Error loading reports: {e}"

@app.route("/india_soil_fertilizer")
def india_soil_fertilizer():
    try:
        df = pd.read_csv("india_soil_fertilizer.csv")
        return jsonify(df.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- 3. Chatbot API ---
@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.get_json()
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    API_URL = "https://api.groq.com/openai/v1/chat/completions"
    MODEL = "llama-3.1-8b-instant"
    API_KEY = os.getenv("GROQ_API_KEY")  # UPDATED (Secure)
    print("API KEY VALUE:", API_KEY)

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful agricultural expert for Indian farmers."},
            {"role": "user", "content": user_message}
        ]
    }

    try:
        response = requests.post(API_URL, json=payload, headers=headers)

        if response.status_code != 200:
            return jsonify({"reply": f"API Error: {response.status_code}"}), 500
        
        result = response.json()
        bot_reply = result['choices'][0]['message']['content']
        return jsonify({"reply": bot_reply})

    except Exception as e:
        print(f"Server Error: {e}")
        return jsonify({"reply": "Sorry, server error."}), 500

if __name__ == "__main__":
    app.run(debug=True)
