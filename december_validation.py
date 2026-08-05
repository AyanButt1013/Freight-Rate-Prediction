import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

#--------------------------------------------------


import pandas as pd
from predictor import FreightPredictor

df = pd.read_csv("december-chart-inputs.csv")

predictor = FreightPredictor()
predictions = predictor.predict(df)

df["predicted_rate"] = predictions

# Keep only the required columns in the required order
df = df[
    [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
    ]
]

df.to_csv("december-predictions.csv", index=False)

print("December predictions saved.")