"""
model.py - Train ML models and select the best one
Run this file first to generate model.pkl and accuracy.txt
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# ── 1. Load Dataset ──────────────────────────────────────────────────────────
print("📂 Loading dataset...")
df = pd.read_csv("data.csv")
print(f"   Rows: {len(df)}, Columns: {list(df.columns)}")

# ── 2. Encode Categorical Columns ────────────────────────────────────────────
le_subject = LabelEncoder()
le_result  = LabelEncoder()

df["Subject_enc"] = le_subject.fit_transform(df["Subject"])
df["Result_enc"]  = le_result.fit_transform(df["Result"])   # Fail=0, Pass=1

# Save the subject encoder so app.py can reuse it
with open("subject_encoder.pkl", "wb") as f:
    pickle.dump(le_subject, f)

print(f"   Classes: {le_result.classes_}")   # ['Fail', 'Pass']

# ── 3. Features & Target ─────────────────────────────────────────────────────
X = df[["StudyHours", "Attendance", "Marks", "Subject_enc"]]
y = df["Result_enc"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Train Three Models ─────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
print("\n📊 Model Accuracies:")
for name, model in models.items():
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    results[name] = (model, acc)
    print(f"   {name}: {acc*100:.2f}%")

# ── 5. Pick Best Model ────────────────────────────────────────────────────────
best_name, (best_model, best_acc) = max(results.items(), key=lambda x: x[1][1])
print(f"\n🏆 Best Model: {best_name} ({best_acc*100:.2f}%)")

# ── 6. Save Artifacts ─────────────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("accuracy.txt", "w") as f:
    f.write(f"{best_acc*100:.2f}")

print("✅ Saved: model.pkl, accuracy.txt, subject_encoder.pkl")
print("\nAll done! Now run:  python app.py")
