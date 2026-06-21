import pandas as pd
import pickle
import re
import matplotlib.pyplot as plt

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# --------------------------
# Load Dataset
# --------------------------

df = pd.read_csv("dataset/emails.csv")

df["text"] = df["subject"] + " " + df["body"]

# --------------------------
# Text Cleaning
# --------------------------

def preprocess(text):
    text = str(text).lower()

    text = re.sub(r"http\S+", " URL ", text)

    text = re.sub(r"[^a-zA-Z ]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text

df["text"] = df["text"].apply(preprocess)

# --------------------------
# Feature engineering
# --------------------------

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


def extract_numeric_features(text):
    urls = re.findall(r'https?://\S+|www\.\S+', str(text))
    url_count = len(urls)
    keyword_count = sum(word in str(text).lower() for word in PHISHING_KEYWORDS)
    return url_count, keyword_count


vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

# TF-IDF on text
X_text = vectorizer.fit_transform(df["text"])

# Numeric features
df[["url_count", "keyword_count"]] = df["text"].apply(lambda t: pd.Series(extract_numeric_features(t)))

X_num = csr_matrix(df[["url_count", "keyword_count"]].values)

# Combine sparse TF-IDF + numeric features
X = hstack([X_text, X_num])

y = df["label"]

# --------------------------
# Split Data
# --------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# --------------------------
# Train Model
# --------------------------

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# --------------------------
# Evaluation
# --------------------------

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:\n")

print(classification_report(y_test, predictions))

# --------------------------
# Confusion Matrix
# --------------------------

cm = confusion_matrix(y_test, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=model.classes_
)

disp.plot()

plt.savefig("static/confusion_matrix.png")

plt.close()

# --------------------------
# Save Model
# --------------------------


# Save model and vectorizer
pickle.dump(model, open("phishing_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel Saved Successfully")

print("\nModel Saved Successfully")