from flask import Flask, render_template, request, jsonify
import pickle
import re
from scipy.sparse import hstack, csr_matrix

app = Flask(__name__)

model = pickle.load(open("phishing_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

PHISHING_KEYWORDS = [
    "verify",
    "password",
    "bank",
    "urgent",
    "login",
    "click",
    "account",
    "security",
    "update",
    "limited"
]

def extract_features(email):

    urls = re.findall(r'https?://\S+', email)

    url_count = len(urls)

    keyword_count = sum(
        word.lower() in email.lower()
        for word in PHISHING_KEYWORDS
    )

    return {
        "url_count": url_count,
        "keyword_count": keyword_count
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    email_text = request.form["email"]

    features = extract_features(email_text)


    # TF-IDF vector for text
    vector_text = vectorizer.transform([email_text])

    # Numeric feature vector (url_count, keyword_count)
    num = csr_matrix([[features["url_count"], features["keyword_count"]]])

    # Combine features to match training
    vector = hstack([vector_text, num])

    prediction = model.predict(vector)[0]

    # Some sklearn classifiers require dense input for predict_proba but many accept sparse; try directly
    try:
        probability = model.predict_proba(vector)[0]
        confidence = round(max(probability) * 100, 2)
    except Exception:
        confidence = None

    return jsonify({
        "prediction": prediction,
        "confidence": confidence,
        "urls": features["url_count"],
        "keywords": features["keyword_count"]
    })

if __name__ == "__main__":
    app.run(debug=True)