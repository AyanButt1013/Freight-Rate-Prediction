import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import List, Optional, Tuple
from sklearn.model_selection import train_test_split
import numpy as np
import joblib

class DataPreparer:
    """
    A modular pipeline to load a dataset (.csv file) into a Pandas dataframe,
    drop missing values, preprocess features, and convert to PyTorch Dataloaders
    """

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.df: Optional[pd.DataFrame] = None
        self.X: Optional[pd.DataFrame] = None
        self.Y: Optional[pd.Series] = None
        self.scaler = StandardScaler()

    def load_data(self) -> "DataPreparer":
        """Load the raw dataset into Pandas Dataframe"""

        self.df = pd.read_csv(self.file_path)
        print(f"[+] Loaded dataset from {self.file_path} with shape {self.df.shape}")
        return self

    def clean_na(self, numeric_features:Optional[List[str],]=None, cat_features:Optional[List[str]]=None) -> "DataPreparer":
        """For filling of the missing values."""
        for col in numeric_features:
            self.df[col] = self.df[col].fillna(
            self.df[col].median()
        )

        for col in cat_features:
            self.df[col]=self.df[col].fillna(
            self.df[col].mode()[0]
        )
        
        print(f"[+] Missing values imputed. Dataset shape:{self.df.shape}")
        return self

    def date_process (self, date_features:Optional[List[str]]=None) -> "DataPreparer":
        if date_features:
            for col in date_features:
                self.df[col] = pd.to_datetime(self.df[col])

                self.df["date_month"] = self.df[col].dt.month
                self.df["date_weekday"] = self.df[col].dt.weekday

                self.df.drop(columns=[col], inplace=True)

            print("[+] Date features created.")

        return self

    def separate_features_target(self, Target_column: str, Drop_columns: Optional[List[str]]=None) -> "DataPreparer":
        """
        Separate features (X) and target (y), and drop unneeded columns.
        """

        if self.df is None:
            raise ValueError("Data not loaded. Call `load_data()` first.")

        drop_cols = [Target_column]

        if drop_cols:
            drop_cols.extend(Drop_columns)

            #self.Y = self.df[Target_column]
            self.Y = np.log1p(self.df[Target_column])
            self.X = self.df.drop(columns=drop_cols)

            print(f"[+] Features shape: {self.X.shape}, Target Shape: {self.Y.shape}")

            return self


    def feature_engineering(self) -> "DataPreparer":

        self.X["delta_lat"] = (
        self.X["delivery_lat"] -
        self.X["pickup_lat"]
        )

        self.X["delta_lon"] = (
        self.X["delivery_lon"] -
        self.X["pickup_lon"]
        )

        print("[+] Feature engineered: delta_lat, delta_lon")

        return self

    def one_hot_encode(self, cat_features:Optional[List[str]]=None) -> "DataPreparer":
        """
        One-hot encode categorical features.
        """

        if self.X is None:
            raise ValueError("Features not separated. Call `separate_features_target()` first.")

        #
        if cat_features:
            self.X = pd.get_dummies(self.X, columns= cat_features, drop_first=True)
            self.X = self.X.astype(float)
            print(f"[+] Applied One-Hot encoding. New number of features are: {self.X.shape[1]}")

        

        return self

    def scale_data(self,X_train, X_val, numeric_features:Optional[List[str]]) -> "DataPreparer":
        """
        Scale the numerical features of dataset.
        """
        if self.X is None:
                    raise ValueError("Features not separated. Call `separate_features_target()` first.")

        if numeric_features:
            X_train = X_train.copy()
            X_val = X_val.copy()

            self.scaler.fit(X_train[numeric_features])
            joblib.dump(self.scaler, "scaler.pkl")

            X_train[numeric_features] = self.scaler.transform(
            X_train[numeric_features]
            )

            X_val[numeric_features] = self.scaler.transform(
            X_val[numeric_features]
            )
        return X_train, X_val

    def train_validation_split(self, val_size=0.2, random_state = 42) -> "DataPreparer":
        X_train, X_test, Y_train, Y_test = train_test_split(
            self.X,
            self.Y,
            test_size= val_size,
            random_state=random_state
        )

        print(f"[+] Train Samples      : {len(X_train)}")
        print(f"[+] Validation Samples : {len(X_test)}")

        print(f"Train X: {X_train.shape}")
        print(f"Train y: {Y_train.shape}")
        print(f"Val X: {X_test.shape}")
        print(f"Val y: {Y_test.shape}")

        return X_train, X_test, Y_train, Y_test



    def get_dataloader(self, X, Y, batch_size: int =32, shuffle: bool = True) -> "DataPreparer":
        """
        Convert prepared data to PyTorch tensors and wrap in Dataloaders
        """
        if self.X is None or self.Y is None:
            raise ValueError("Data preparation is incomplete")

        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        Y_tensor = torch.tensor(Y.values, dtype=torch.float32).unsqueeze(1)

        dataset = TensorDataset(X_tensor, Y_tensor)

        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 
