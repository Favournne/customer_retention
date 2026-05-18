# model_utils.py
class ChurnPredictor:
    def __init__(self, model, scaler, features):
        self.model = model
        self.scaler = scaler
        self.features = features

    def predict_and_recommend(self, input_data):
        # This will be overridden by the loaded model
        pass