import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("news.csv")

# Input and output
x = data["text"]
y = data["label"]

# Convert text into vectors
vectorizer = TfidfVectorizer(stop_words="english")

x_vector = vectorizer.fit_transform(x)

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x_vector,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression()

model.fit(x_train, y_train)

# Prediction
y_pred = model.predict(x_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Save model
pickle.dump(model, open("fake_news_model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully")
print("Saving model...")
print("Done")
