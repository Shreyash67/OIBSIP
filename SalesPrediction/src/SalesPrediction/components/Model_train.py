import pandas as pd
from sklearn.model_selection import train_test_split

class Spliting_x_y:
    def __init__(self, data):
        self.data = data
    
    def transform(self):
        # Check if the 'Unnamed: 0' column exists before dropping it
        if 'Unnamed: 0' in self.data.columns:
            self.data = self.data.drop("Unnamed: 0", axis=1)
        return self.data

    def split_x_y(self):
        data = self.transform()
        x = data[['TV', 'Radio', 'Newspaper']]
        y = data["Sales"]
        return x, y

    
class Training_x_y(Spliting_x_y):
    def __init__(self, data):
        super().__init__(data)

    def train_test(self):
        x, y = self.split_x_y()
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        return x_train, x_test, y_train, y_test