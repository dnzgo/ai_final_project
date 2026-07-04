class StressDetectionAgent:

    def __init__(self, model, preprocessor, label_encoder):
        self.model = model
        self.preprocessor = preprocessor
        self.label_encoder = label_encoder

    def perceive(self, user_data):
        return self.preprocessor.transform(user_data)

    def think(self, encoded_data):
        prediction = self.model.predict(encoded_data)
        prediction_label = self.label_encoder.inverse_transform(prediction)[0]

        probabilities = self.model.predict_proba(encoded_data)[0]
        confidence = max(probabilities)

        return prediction_label, confidence

    def act(self, prediction_label, confidence):

        if prediction_label == "No":
            return {
                "prediction": prediction_label,
                "risk_level": "Low",
                "confidence": round(float(confidence), 2),
                "action": "Show general wellness tips."
            }

        elif prediction_label == "Maybe":
            return {
                "prediction": prediction_label,
                "risk_level": "Medium",
                "confidence": round(float(confidence), 2),
                "action": "Recommend stress management exercises and suggest a follow-up assessment."
            }

        else:
            return {
                "prediction": prediction_label,
                "risk_level": "High",
                "confidence": round(float(confidence), 2),
                "action": "Recommend professional mental health support."
            }

    def run(self, user_data):
        encoded_data = self.perceive(user_data)
        prediction_label, confidence = self.think(encoded_data)
        return self.act(prediction_label, confidence)