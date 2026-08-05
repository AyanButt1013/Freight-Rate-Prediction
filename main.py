import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

#--------------------------------------------------

from data_prepare import DataPreparer
from model import FreightRateModel
from trainer import FreightTrainer
from evaluator import FreightEvaluator
import numpy as np


# Initialize and run pipeline
pipeline = DataPreparer("train-test.csv")

pipeline =(
    pipeline
    .load_data()
    .date_process(date_features=['date'])
    .clean_na(numeric_features=[
            "pickup_lat",
            "pickup_lon",
            "delivery_lat",
            "delivery_lon",
            "distance",
            "weight",
            "market_index",
            "quote_signal",
        ],cat_features=["equipment"])
    .separate_features_target(Target_column = 'posted_rate', Drop_columns=['load_id','pickup','delivery'])
    .feature_engineering()
    .one_hot_encode(cat_features=["equipment"])
)


X_train, X_val, y_train, y_val = pipeline.train_validation_split(
    val_size=0.2
)

X_train, X_val = pipeline.scale_data(
    X_train,
    X_val,
    numeric_features=[
        "pickup_lat",
        "pickup_lon",
        "delivery_lat",
        "delivery_lon",
        "distance",
        "weight",
        "market_index",
        "quote_signal",
        "delta_lat",
        "delta_lon"
    ]
)

print(X_train.isna().sum())


train_loader = pipeline.get_dataloader(
    X_train,
    y_train,
    shuffle=True
)

validation_loader = pipeline.get_dataloader(
    X_val,
    y_val,
    shuffle=False
)




input_size = X_train.shape[1]

model = FreightRateModel(input_size)

print(model)

trainer = FreightTrainer(
    model,
    train_loader,
    validation_loader,
    epochs = 35
)

trainer.train()

trainer.save_model()



evaluator = FreightEvaluator(
    model= model, 
    dataloader=validation_loader
)

results = evaluator.evaluate()