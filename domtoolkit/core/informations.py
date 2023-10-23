import pandas as pd
from tabulate import tabulate

class DataFrameDeepInfo:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def print_basic_info(self):
        print(f"Data Frame has {self.dataframe.shape[0]} rows and {self.dataframe.shape[1]} columns\n")
        print("These are the column names:")

    def collect_column_information(self):
        information = []
        for counter, column in enumerate(self.dataframe.columns):
            info = [
                f"{counter}: {column}", 
                f"{self.dataframe[column].nunique()} unique values", 
                f"{self.dataframe[column].isna().sum()} are NaN", 
                f"Type: {self.dataframe[column].dtype}"
            ]
            information.append(info)
        return information
    
    def print_detailed_column_info(self):
        for counter, column in enumerate(self.dataframe.columns):
            print(f"{counter}: {column} has {self.dataframe[column].nunique()} unique values. {self.dataframe[column].isna().sum()} are NaN.")
            print(f"It's {self.dataframe[column].dtype}. These are the unique values:")
            print(f"{self.dataframe[column].unique()}\n")

    def visualize(self, small=False):
        self.print_basic_info()
        information = self.collect_column_information()
        print(tabulate(information, headers=['Column', 'Values', 'NaN', 'Type']))
        print("\n")
        if not small:
            self.print_detailed_column_info()