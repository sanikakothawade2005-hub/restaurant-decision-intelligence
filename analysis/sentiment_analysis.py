import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# Load Kaggle dataset
df = pd.read_csv("data/food_reviews.csv")

print("Total Reviews:", len(df))
print("\nDataset:")
print(df.head())

# Convert Liked values into sentiment labels
df["sentiment"] = df["Liked"].map({
    1: "positive",
    0: "negative"
})

# Convert review text into numerical features using TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2))

X = vectorizer.fit_transform(df["Review"])
y = df["sentiment"]

print("\nTF-IDF Shape:", X.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# Create and train the model
model = LogisticRegression()
model.fit(X_train, y_train)

print("\nModel trained successfully!")

# Save the trained model and TF-IDF vectorizer
joblib.dump(model, "analysis/sentiment_model.pkl")
joblib.dump(vectorizer, "analysis/tfidf_vectorizer.pkl")

print("Sentiment model saved successfully!")
print("TF-IDF vectorizer saved successfully!")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Predict sentiment of a new review
new_review = ["The food was delicious and the service was excellent"]

new_review_tfidf = vectorizer.transform(new_review)
prediction = model.predict(new_review_tfidf)

print("\nNew Review:")
print(new_review[0])

print("Predicted Sentiment:", prediction[0])