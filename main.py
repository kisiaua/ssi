import pandas as pd
import random


csv = pd.read_csv('dataset/HeartDisease.csv')

columns_to_drop = ['target']
target_column = ['target']
target_column_num = -1
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


class KNN:
    @staticmethod
    def minkowskiDistance(row_training, row_testing, m):
        distance = 0
        for i in range(len(row_training) - 1): # TODO: mozliwosc ze nie ostatnia kolumna bedzie etykietami
            distance += abs(row_training[i] - row_testing[i]) ** m
        return distance ** (1 / m)

    @staticmethod
    def knn(testing, training, m, k):
        distances = []
        classes = {}
        correct = 0

        for _, row_tst in testing.iterrows():
            row_testing = row_tst.tolist()

            distances = []
            for _, row in testing[target_column].iterrows():
                classes.update({row[0]: 0})

            for _, row_trn in training.iterrows():
                row_training = row_trn.tolist()
                distances.append([KNN.minkowskiDistance(row_training, row_testing, m), row_training[target_column_num]])
            distances = sorted(distances)

            for i in range(k):
                classes[distances[i][1]] += 1

            if max(classes, key=classes.get) == row_testing[target_column_num]:
                correct += 1

        return f'KNN wynik: {correct} na {len(testing)}, {(correct / len(testing) * 100):.2f}% (m={m}, k={k})'