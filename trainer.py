import torch
import torch.nn as nn
import torch.optim as optim


class FreightTrainer:
    """
    A module to train on the dataset
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        learning_rate: float = 1e-3,
        epochs: int = 50,
        device: str = None,
    ):

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs

        self.criterion = nn.HuberLoss()

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )

        # Reduce LR when validation loss plateaus
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )

        # Early stopping
        self.best_val_loss = float("inf")
        self.early_stop_patience = 8
        self.counter = 0

        self.train_losses = []
        self.val_losses = []

    def train(self):
        """
        Train the model
        """

        print(f"[+] Training on: {self.device}")

        for epoch in range(self.epochs):

            ############################
            # TRAINING
            ############################

            self.model.train()
            running_train_loss = 0.0

            for batch_x, batch_y in self.train_loader:

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions = self.model(batch_x)

                loss = self.criterion(predictions, batch_y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_train_loss += loss.item()

            train_loss = running_train_loss / len(self.train_loader)

            ############################
            # VALIDATION
            ############################

            self.model.eval()
            running_val_loss = 0.0

            with torch.no_grad():

                for batch_x, batch_y in self.val_loader:

                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    predictions = self.model(batch_x)

                    loss = self.criterion(predictions, batch_y)

                    running_val_loss += loss.item()

            val_loss = running_val_loss / len(self.val_loader)

            ############################
            # Scheduler
            ############################

            self.scheduler.step(val_loss)

            ############################
            # Save history
            ############################

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch+1}/{self.epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"LR: {current_lr:.6f}"
            )

            ############################
            # Early stopping
            ############################

            if val_loss < self.best_val_loss:

                self.best_val_loss = val_loss
                self.counter = 0

                torch.save(
                    self.model.state_dict(),
                    "best_model.pth"
                )

            else:

                self.counter += 1

                print(
                    f"No improvement ({self.counter}/{self.early_stop_patience})"
                )

                if self.counter >= self.early_stop_patience:

                    print("\nEarly stopping triggered.")
                    break

        # Restore best weights
        self.model.load_state_dict(
            torch.load(
                "best_model.pth",
                map_location=self.device
            )
        )

        print("\nTraining Completed")
        print("Best model restored.")

    def save_model(self, path="freight_rate_model.pth"):
        """
        Save trained model.
        """

        torch.save(self.model.state_dict(), path)

        print(f"Model saved to: {path}")

    def get_training_history(self):
        """
        Return training history.
        """

        return {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses
        }