import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from agents.stress_detection_agent import StressDetectionAgent


# Load Dataset
data_frame = pd.read_csv("data/mental_health.csv")

target_column = "Growing_Stress"

# Remove columns not used by the early stress detection agent
columns_to_remove = [
    "Timestamp",
    "treatment",
    "Mental_Health_History",
    "Mood_Swings",
    "Work_Interest",
    "Social_Weakness",
    "mental_health_interview",
    "care_options"
]

data_frame = data_frame.drop(columns_to_remove, axis=1)

# Remove rows where self_employed is missing
data_frame = data_frame.dropna(subset=["self_employed"])



# Split Features and Target
X = data_frame.drop(target_column, axis=1)
y = data_frame[target_column]

# Manual Encoding

X["Gender"] = X["Gender"].map({
    "Female": 0,
    "Male": 1
})

binary_map = {
    "No": 0,
    "Yes": 1
}

X["self_employed"] = X["self_employed"].map(binary_map)
X["family_history"] = X["family_history"].map(binary_map)
X["Coping_Struggles"] = X["Coping_Struggles"].map(binary_map)

X["Days_Indoors"] = X["Days_Indoors"].map({
    "Go out Every day": 0,
    "1-14 days": 1,
    "15-30 days": 2,
    "31-60 days": 3,
    "More than 2 months": 4
})

X["Changes_Habits"] = X["Changes_Habits"].map({
    "No": 0,
    "Maybe": 1,
    "Yes": 2
})


# Encode Target

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Target classes:")
print(label_encoder.classes_)


# Train/Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))


# One-Hot Encode Country and Occupation

one_hot_columns = ["Country", "Occupation"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore"),
            one_hot_columns
        )
    ],
    remainder="passthrough"
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)



# Logistic Regression

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(X_train_encoded, y_train)
logistic_predictions = logistic_model.predict(X_test_encoded)
logistic_accuracy = accuracy_score(y_test, logistic_predictions)



# Random Forest

forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

forest_model.fit(X_train_encoded, y_train)
forest_predictions = forest_model.predict(X_test_encoded)
forest_accuracy = accuracy_score(y_test, forest_predictions)

agent = StressDetectionAgent(
    model=forest_model,
    preprocessor=preprocessor,
    label_encoder=label_encoder
)


# Evaluation

print("\nLogistic Regression Accuracy:", logistic_accuracy)
print("Random Forest Accuracy:", forest_accuracy)

print("\nLogistic Regression Report:")
print(classification_report(
    y_test,
    logistic_predictions,
    target_names=label_encoder.classes_
))

print("\nRandom Forest Report:")
print(classification_report(
    y_test,
    forest_predictions,
    target_names=label_encoder.classes_
))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, forest_predictions))

print("\nDataset Shape:")
print(data_frame.shape)

# Feature Importance

encoded_feature_names = preprocessor.get_feature_names_out()

clean_feature_names = []

for feature in encoded_feature_names:
    feature = feature.replace("onehot__", "")
    feature = feature.replace("remainder__", "")
    feature = feature.replace("_", " ")

    clean_feature_names.append(feature)

importances = forest_model.feature_importances_

feature_importance = sorted(
    zip(clean_feature_names, importances),
    key=lambda x: x[1],
    reverse=True
)

print("\nTop 10 Most Important Features:")
for feature, importance in feature_importance[:10]:
    print(f"{feature}: {importance:.4f}")


# Agent Decision Example


sample_user = X_test.iloc[[0]]

decision = agent.run(sample_user)

print("\nAgent Decision")
print(decision)

# --------------------- Graphs -------------------------

# Target distribution
target_counts = y.value_counts()

plt.figure(figsize=(6, 4))

target_counts.plot(
    kind="bar",
    color="#4CAF50"   # Green
)

plt.title("Growing Stress Classification Distribution")
plt.xlabel("Stress Level")
plt.ylabel("Number of Participants")
plt.tight_layout()
plt.savefig("charts/target_distribution.png", dpi=300)
plt.close()


# Model accuracy comparison
model_names = ["Logistic Regression", "Random Forest"]
accuracies = [logistic_accuracy, forest_accuracy]

plt.figure(figsize=(6,4))

plt.bar(
    model_names,
    accuracies,
    color=["#A5D6A7", "#2E7D32"]
)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0,1)

plt.tight_layout()
plt.savefig("charts/model_accuracy_comparison.png", dpi=300)
plt.close()


# Feature importance
top_features = feature_importance[:10]
features = [item[0] for item in top_features]
scores = [item[1] for item in top_features]

plt.figure(figsize=(8,5))

plt.barh(
    features[::-1],
    scores[::-1],
    color="#43A047"
)

plt.title("Most Important Features for Stress Prediction")
plt.xlabel("Importance")

plt.tight_layout()
plt.savefig("charts/feature_importance.png", dpi=300)
plt.close()

# confusion matrixes
disp = ConfusionMatrixDisplay.from_estimator(
    forest_model,
    X_test_encoded,
    y_test,
    display_labels=label_encoder.classes_,
    cmap="Greens"
)

disp.ax_.set_title("Random Forest Confusion Matrix", fontsize=16)
disp.ax_.tick_params(labelsize=12)

plt.savefig("charts/confusion_matrix.png", dpi=300)
plt.close()

disp = ConfusionMatrixDisplay.from_estimator(
    logistic_model,
    X_test_encoded,
    y_test,
    display_labels=label_encoder.classes_,
    cmap="Greens"
)

disp.ax_.set_title("Logistic Regression Confusion Matrix", fontsize=16)
disp.ax_.tick_params(labelsize=12)

plt.savefig("charts/confusion_matrix_logistic.png", dpi=300)
plt.close()

print("\nCharts saved in charts/ folder.")