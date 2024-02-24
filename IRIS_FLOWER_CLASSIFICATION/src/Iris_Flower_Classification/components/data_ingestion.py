import pandas as pd
from src.Iris_Flower_Classification.logger import logging
from sklearn.model_selection import train_test_split

class data_ingestion:
    def __init__(self,data):
        self.data = data

    def x_y_split(self):
        x = self.data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]    
        y = self.data["Species"]
        logging.info("Splitting into x and y")
        return x,y
    
class Training_x_y(data_ingestion):
    def __init__(self,data):
        super().__init__(data)
    
    def train_test(self):
        x,y= self.x_y_split()
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
        logging.info("Splitting the data into train and test\n")
        return x_train,x_test,y_train,y_test