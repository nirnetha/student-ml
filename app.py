"""
app.py - Flask web application for AI Study Tracker
"""

from flask import Flask, render_template, request, jsonify
import pickle, os, csv
from datetime import datetime
import pandas as pd

app = Flask(__name__)

# ── Load Model & Artifacts ────────────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("accuracy.txt", "r") as f:
    accuracy = f.read().strip()

with open("subject_encoder.pkl", "rb") as f:
    le_subject = pickle.load(f)

USER_DATA_FILE = "user_data.csv"

if not os.path.exists(USER_DATA_FILE):
    with open(USER_DATA_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "StudyHours", "Attendance", "Marks", "Subject", "Prediction"])


def save_user_data(study_hours, attendance, marks, subject, prediction):
    with open(USER_DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            study_hours, attendance, marks, subject, prediction
        ])


def encode_subject(subject_str):
    known = list(le_subject.classes_)
    if subject_str not in known:
        subject_str = known[0]
    return le_subject.transform([subject_str])[0]


@app.route("/")
def index():
    known_subjects = list(le_subject.classes_)
    return render_template("index.html", accuracy=accuracy, subjects=known_subjects)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        attendance = float(request.form["attendance"])

        # Collect per-subject data: name, marks, study_hours
        subject_names      = request.form.getlist("subject_name[]")
        subject_marks_raw  = request.form.getlist("subject_marks[]")
        subject_hours_raw  = request.form.getlist("subject_hours[]")

        triples = []
        for name, marks_s, hours_s in zip(subject_names, subject_marks_raw, subject_hours_raw):
            name = name.strip()
            if name and marks_s.strip() and hours_s.strip():
                triples.append((name, float(marks_s), float(hours_s)))

        if not triples:
            raise ValueError("Please add at least one subject.")

        avg_marks = sum(m for _, m, _ in triples) / len(triples)
        avg_hours = sum(h for _, _, h in triples) / len(triples)
        primary_subject = triples[0][0]
        subject_enc = encode_subject(primary_subject)

        features   = [[avg_hours, attendance, avg_marks, subject_enc]]
        pred_enc   = model.predict(features)[0]
        prediction = "Pass" if pred_enc == 1 else "Fail"
        proba      = model.predict_proba(features)[0]
        confidence = round(max(proba) * 100, 1)

        # Interpretation message
        if prediction == "Pass":
            if confidence >= 85:
                interpretation = "You are doing excellent! Keep it up 🌟"
            else:
                interpretation = "You are doing well 👍 Stay consistent!"
        else:
            if avg_marks < 50:
                interpretation = "Focus more on weak subjects ⚠️ You can improve!"
            else:
                interpretation = "Almost there! A little more effort will make a difference 💪"

        # Save one row per subject
        for subj_name, subj_marks, subj_hours in triples:
            save_user_data(subj_hours, attendance, subj_marks, subj_name, prediction)

        known_subjects = list(le_subject.classes_)
        return render_template(
            "index.html",
            accuracy=accuracy,
            subjects=known_subjects,
            prediction=prediction,
            confidence=confidence,
            interpretation=interpretation,
            attendance=attendance,
            avg_marks=round(avg_marks, 1),
            avg_hours=round(avg_hours, 1),
            subject_triples=triples,
        )
    except Exception as e:
        return f"<h2>Error: {e}</h2><a href='/'>Go Back</a>", 400


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/chart-data")
def chart_data():
    if not os.path.exists(USER_DATA_FILE):
        return jsonify({"records": []})

    df = pd.read_csv(USER_DATA_FILE)
    if df.empty:
        return jsonify({"records": []})

    records = df.to_dict(orient="records")

    # Subject-wise avg marks, avg hours, pass/fail counts
    subject_stats = {}
    for _, row in df.iterrows():
        subj = row["Subject"]
        pred = row["Prediction"]
        if subj not in subject_stats:
            subject_stats[subj] = {"Pass": 0, "Fail": 0, "marks": [], "hours": []}
        subject_stats[subj][pred] += 1
        subject_stats[subj]["marks"].append(float(row["Marks"]))
        subject_stats[subj]["hours"].append(float(row["StudyHours"]))

    # Compute averages
    for s in subject_stats:
        m = subject_stats[s]["marks"]
        h = subject_stats[s]["hours"]
        subject_stats[s]["avg_marks"] = round(sum(m) / len(m), 1) if m else 0
        subject_stats[s]["avg_hours"] = round(sum(h) / len(h), 1) if h else 0

    pass_count = int((df["Prediction"] == "Pass").sum())
    fail_count = int((df["Prediction"] == "Fail").sum())

    # Pass/fail subject lists for clickable stats
    pass_subjects = sorted(set(df[df["Prediction"] == "Pass"]["Subject"].tolist()))
    fail_subjects = sorted(set(df[df["Prediction"] == "Fail"]["Subject"].tolist()))

    return jsonify({
        "records": records,
        "subject_stats": subject_stats,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "pass_subjects": pass_subjects,
        "fail_subjects": fail_subjects,
    })


if __name__ == "__main__":
    app.run(debug=True)
