import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

#--------------------------------------------------

import pandas as pd
from predictor import FreightPredictor


# Load input CSV
df = pd.read_csv("validation.csv")


# Load trained model
predictor = FreightPredictor()


# Predict
predictions = predictor.predict(df)


# Save results
df["predicted_rate"] = predictions

df.to_csv(
    "predictions.csv",
    index=False
)

print("Prediction completed!")