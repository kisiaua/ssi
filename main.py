import pandas as pd
import random


csv = pd.read_csv('dataset/HeartDisease.csv')

columns_to_drop = ['target']
split_ratio = 0.7

class ProcessingData:
    @staticmethod
    def ShuffleData(data):
        for i in reversed(range(len(data))):
            j = random.randint(0, i)
            data.iloc[i], data.iloc[j] = data.iloc[j], data.iloc[i]
        return data.reset_index(drop=True)

    @staticmethod
    def NormalizeData(data):
        for column in data.drop(columns=columns_to_drop):
            minMaxValue = data[column].agg(['min', 'max']).tolist()
            for i, value in enumerate(data[column]):
                data.loc[i, column] = (value - minMaxValue[0]) / (minMaxValue[1] - minMaxValue[0])
        return data

    @staticmethod
    def SplitData(data):
        split = int(split_ratio * len(data))
        return data.iloc[:split, :].reset_index(drop=True), data.iloc[split:, :].reset_index(drop=True)