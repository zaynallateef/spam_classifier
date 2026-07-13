

import re
import string
import pickle

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation/digits/extra whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_data(path: str) -> pd.DataFrame:
    # The original Kaggle "spam.csv" is latin-1 encoded and has 3 stray
    # unnamed columns full of NaNs that we don't need.
    df = pd.read_csv(path, encoding="latin-1")
    df = df.iloc[:, :2]
    df.columns = ["label", "text"]
    df = df.dropna(subset=["text"])
    df["label"] = df["label"].str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])]
    df["text_clean"] = df["text"].apply(clean_text)
    return df


def main():
    print("Loading data...")
    df = load_data("spam.csv")
    print(f"Loaded {len(df)} rows. Class balance:\n{df['label'].value_counts()}")

    X = df["text_clean"]
    y = df["label"].map({"ham": 0, "spam": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    candidates = {}

    print("\nTraining Logistic Regression...")
    log_reg = LogisticRegression(max_iter=1000, class_weight="balanced", C=10)
    log_reg.fit(X_train_vec, y_train)
    candidates["logistic_regression"] = log_reg

    print("Training Linear SVM (calibrated for probability output)...")

    svm_base = LinearSVC(class_weight="balanced", C=1.0)
    svm = CalibratedClassifierCV(svm_base, cv=5)
    svm.fit(X_train_vec, y_train)
    candidates["svm"] = svm

    print("\n=== Model comparison (held-out test set) ===")
    results = {}
    for name, model in candidates.items():
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        results[name] = f1
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1 score:  {f1:.4f}")
        print(classification_report(y_test, preds, target_names=["ham", "spam"]))

    best_name = max(results, key=results.get)
    best_model = candidates[best_name]
    print(f"\n=> Best model by F1 score: {best_name} (F1={results[best_name]:.4f})")

    with open("model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    with open("model_name.txt", "w") as f:
        f.write(best_name)

    print("\nSaved model.pkl, vectorizer.pkl, model_name.txt")


if __name__ == "__main__":
    main()
