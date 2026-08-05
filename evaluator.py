import torch
import torch.nn as nn
import numpy as np

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

class FreightEvaluator:
    """
    The module to evalaute a trained regression PyTorch model.
    """

    def __init__(self, model:nn.Sequential, dataloader, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.dataloader = dataloader

        self.criterion = nn.HuberLoss()

    def evaluate (self):
        self.model.eval()

        running_losses = 0.0

        predictions = []
        targets = []

        with torch.no_grad():
            for batch_x, batch_y in self.dataloader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                outputs = self.model(batch_x)
                loss = self.criterion(outputs,batch_y)

                running_losses += loss.item()

                

                predictions.extend(
                    np.expm1(outputs.cpu().numpy().flatten())
                )

                targets.extend(
                    np.expm1(batch_y.cpu().numpy().flatten())
                )

        mse = mean_squared_error(targets, predictions)
        rmse = mse ** 0.5
        mae = mean_absolute_error(targets, predictions)
        r2 = r2_score(targets, predictions)

        avg_loss = running_losses / len(self.dataloader)

        print("\n========== Evaluation ==========")
        print(f"Loss : {avg_loss:.4f}")
        print(f"MSE  : {mse:.4f}")
        print(f"RMSE : {rmse:.4f}")
        print(f"MAE  : {mae:.4f}")
        print(f"R²   : {r2:.4f}")
        print("================================")

        return {
        "loss": avg_loss,
        "MSE" : mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "predictions": predictions,
        "targets": targets
        }




       

