# train_ml.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# -------------------------
# Load Dataset
# -------------------------
# Columns needed:
#   text → claim or article
#   label → 1 = REAL, 0 = FAKE

df = pd.read_csv("dataset_fake_real.csv")

X = df["text"]
y = df["label"]

# -------------------------
# Vectorizer
# -------------------------
vectorizer = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1,2),
    stop_words="english"
)

X_vec = vectorizer.fit_transform(X)

# -------------------------
# Train/Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42
)

# -------------------------
# Logistic Regression Model
# -------------------------
clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)

acc = clf.score(X_test, y_test)
print("Test Accuracy:", acc)

# -------------------------
# Save Model + Vectorizer
# -------------------------
joblib.dump(clf, "models/fake_real_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Model saved successfully!")
