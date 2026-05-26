import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# Sample dataset
data = {
    "text": [
        "A ghost scares a family in a haunted house",
        "Two college students fall in love",
        "Aliens attack the earth with spaceships",
        "A detective investigates a murder case",
        "A superhero saves the city from villains",
        "A funny story about friends on vacation",
        "A king fights to save his kingdom",
        "A scientist creates a dangerous robot",
        "A family faces paranormal activities",
        "A couple struggles in their relationship",
        "A police officer catches criminals",
        "An astronaut travels through space",
        "A cursed doll kills people",
        "Best friends enjoy a comedy trip",
        "A warrior protects his nation"
    ],
    
    "genre": [
        "Horror",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "Action",
        "Comedy",
        "Adventure",
        "Sci-Fi",
        "Horror",
        "Romance",
        "Action",
        "Sci-Fi",
        "Horror",
        "Comedy",
        "Adventure"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and Labels
X = df["text"]
y = df["genre"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Machine Learning Pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, predictions)

print("========== MOVIE GENRE CLASSIFICATION ==========")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

# Custom Prediction
while True:
    user_input = input("\nEnter movie plot summary (or type 'exit'): ")

    if user_input.lower() == "exit":
        print("Program Ended")
        break

    result = model.predict([user_input])

    print("Predicted Genre:", result[0])