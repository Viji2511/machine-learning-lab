import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["text", "label"]
    )



dir_path = r"C:\Users\DELL\Desktop\SSN\sem 6\ml_lab\EXP 3\sentiment labelled sentences"

files = [
    "imdb_labelled.csv",
    "amazon_cells_labelled.csv",
    "yelp_labelled.csv"
]


data = pd.concat(
    [load_data(os.path.join(dir_path, f)) for f in files],
    ignore_index=True
)

# Features & Labels
X = data["text"]
y = data["label"]


vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000,
    ngram_range=(1, 2)
)

X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_vec,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


log_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)

print("\nLogistic Regression Results")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log):.4f}")
print(classification_report(y_test, y_pred_log))

cm_log = confusion_matrix(y_test, y_pred_log)
ConfusionMatrixDisplay(
    confusion_matrix=cm_log,
    display_labels=["Negative", "Positive"]
).plot(cmap="Blues")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

dt_model = DecisionTreeClassifier(
    max_depth=5,
    random_state=42
)

dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

print("\nDecision Tree Results (max_depth = 5)")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
print(classification_report(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test, y_pred_dt)
ConfusionMatrixDisplay(
    confusion_matrix=cm_dt,
    display_labels=["Negative", "Positive"]
).plot(cmap="Greens")
plt.title("Decision Tree Confusion Matrix (Depth = 5)")
plt.show()
