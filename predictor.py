import pandas as pd
import numpy as np
import torch
import joblib

from model import FreightRateModel


class FreightPredictor:

    def __init__(
        self,
        model_path="freight_rate_model.pth",
        scaler_path="scaler.pkl",
        device=None
    ):

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Load scaler
        self.scaler = joblib.load(scaler_path)

        # Features used during training (must match exactly)
        self.feature_columns = [
            "pickup_lat",
            "pickup_lon",
            "delivery_lat",
            "delivery_lon",
            "distance",
            "weight",
            "market_index",
            "quote_signal",
            "date_month",
            "date_weekday",
            "equipment_Flatbed",
            "equipment_Reefer",
            "delta_lat",
            "delta_lon",
        ]

        self.numeric_features = [
            "pickup_lat",
            "pickup_lon",
            "delivery_lat",
            "delivery_lon",
            "distance",
            "weight",
            "market_index",
            "quote_signal",
            "delta_lat",
            "delta_lon",
        ]

        # Build model
        self.model = FreightRateModel(len(self.feature_columns))

        self.model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device
            )
        )

        self.model.to(self.device)
        self.model.eval()

    #####################################################################

    def preprocess(self, data: pd.DataFrame):

        data = data.copy()

        #############################################################
        # Missing value imputation (same as training)
        #############################################################

        numeric_cols = [
            "pickup_lat",
            "pickup_lon",
            "delivery_lat",
            "delivery_lon",
            "distance",
            "weight",
            "market_index",
            "quote_signal",
        ]

        for col in numeric_cols:
            data[col] = data[col].fillna(
                data[col].median()
            )

        data["equipment"] = data["equipment"].fillna(
            data["equipment"].mode()[0]
        )

        #############################################################
        # Date features
        #############################################################

        data["date"] = pd.to_datetime(data["date"])

        data["date_month"] = data["date"].dt.month
        data["date_weekday"] = data["date"].dt.weekday

        data.drop(columns=["date"], inplace=True)

        #############################################################
        # Feature Engineering
        #############################################################

        data["delta_lat"] = (
            data["delivery_lat"] - data["pickup_lat"]
        )

        data["delta_lon"] = (
            data["delivery_lon"] - data["pickup_lon"]
        )

        #############################################################
        # One-Hot Encoding
        #############################################################

        data = pd.get_dummies(
            data,
            columns=["equipment"],
            drop_first=True
        )

        #############################################################
        # Add missing dummy columns
        #############################################################

        for col in self.feature_columns:
            if col not in data.columns:
                data[col] = 0

        #############################################################
        # Keep only training columns in correct order
        #############################################################

        data = data[self.feature_columns]

        #############################################################
        # Convert bools to integers
        #############################################################

        for col in data.columns:
            if data[col].dtype == bool:
                data[col] = data[col].astype(int)

        #############################################################
        # Scale numeric features
        #############################################################

        data[self.numeric_features] = self.scaler.transform(
            data[self.numeric_features]
        )

        return data

    #####################################################################

    def predict(self, data: pd.DataFrame):

        X = self.preprocess(data)

        X_tensor = torch.tensor(
            X.values,
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            prediction = self.model(X_tensor)

        prediction = prediction.cpu().numpy().flatten()

        # Prevent unrealistic exponentials
        prediction = np.clip(prediction, 0, 10)

        # Reverse log1p transformation
        prediction = np.expm1(prediction)

        return prediction


#########################################################################

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("input_csv")
    parser.add_argument(
        "--output",
        default="validation-predictions.csv"
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    predictor = FreightPredictor()

    predictions = predictor.predict(df)

    output = pd.DataFrame({
        "load_id": df["load_id"],
        "predicted_rate": predictions
    })

    output.to_csv(args.output, index=False)

    print(f"Saved predictions to {args.output}")