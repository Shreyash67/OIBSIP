import pandas as pd
from src.Iris_Flower_Classification.logger import logging
from sklearn.preprocessing import LabelEncoder

class data_transform:
    def __init__(self,data):
        self.data=data

    def data_encoding(self):
        self.data = data.drop("Id",axis=1)
        le = LabelEncoder()
        self.data["Species"] = le.fit_transform(self.data["Species"])
        logging.info("Initializing Data Preprocessing")
        return self.data


data=pd.read_csv(r"D:\OI\Task_1\notebooks\data\Iris.csv")
obj = data_transform(data)