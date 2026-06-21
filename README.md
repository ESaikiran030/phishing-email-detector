Phishing Email Detector

Overview
- Simple Flask app and training script that classifies emails as `phishing` or `safe`.

Files
- `train_model.py`: trains the model, produces `phishing_model.pkl`, `vectorizer.pkl`, and saves a confusion matrix at `static/confusion_matrix.png`.
- `app.py`: Flask app exposing `/` and `/predict` endpoints.
- `templates/index.html`, `static/style.css`, `static/script.js`: frontend.
- `dataset/emails.csv`: sample dataset.
- `requirements.txt`: Python dependencies.

Quick start

1. Create a virtualenv and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

2. Train the model:

```bash
python train_model.py
```

3. Run the app:

```bash
python app.py
```

4. Open http://127.0.0.1:5000 in your browser and paste an email to test.

Notes
- Training script now combines TF-IDF text features with numeric URL/keyword counts for better detection.
- If your dataset is larger, consider using a stronger classifier or performing cross-validation.
