import pandas as pd
import random
import math


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


class Bayes:
    def __init__(self, training, testing):
        self.training = training
        self.testing = testing
        self.separateByClass()
        self.calcStatistics()
        self.training[target_column] = training[target_column].astype(str)
        self.testing[target_column] = testing[target_column].astype(str)

    def separateByClass(self):
        self.firstSet = self.training.groupby(target_column).get_group(1).reset_index(drop=True)
        self.secondSet = self.training.groupby(target_column).get_group(0).reset_index(drop=True)

    def calcStatistics(self):

        valuesFirstSet = []
        valuesSecondSet = []

        for column in self.training.columns[:-1]:
            valuesFirstSet.append([self.firstSet[column].mean(), self.firstSet[column].std(), len(self.firstSet[column])])
            valuesSecondSet.append([self.secondSet[column].mean(), self.secondSet[column].std(), len(self.secondSet[column])])

        self.stats = {
            "1": valuesFirstSet,
            "0": valuesSecondSet,
        }

    def gaussianProb(self, x, mean, std):
        fraction = (1 / (std * math.sqrt(2 * math.pi)))
        exp = math.exp(-1/2 * ((x - mean) / std) ** 2)
        return fraction * exp

    def triangularDistribution(self, x, mean, std):
        if x < (mean - math.sqrt(6) * std) or x > (mean + math.sqrt(6) * std):
            return 0
        elif x >= (mean - math.sqrt(6) * std) or x <= mean:
            return ((x - mean) / (6 * std ** 2)) + (1 / (math.sqrt(6) * std))
        elif x <= (mean - math.sqrt(6) * std) or x >= mean:
            return (-1 *(x - mean) / (6 * std ** 2)) + (1 / (math.sqrt(6) * std))
    
    def calcProbabilities(self, row):
        probabilitiesByClass = {}
        for className, classValues in self.stats.items():
            probabilitiesByClass[className] = self.stats[className][0][2]/float(len(self.training))
        
            for idx in range(len(classValues)):
                mean, std, _ = classValues[idx]
                probabilitiesByClass[className] *= self.gaussianProb(row[idx], mean, std)

        return probabilitiesByClass

    def bayes(self):
        correct = 0

        for _, row_tst in self.testing.iterrows():
            row_testing = row_tst.tolist()

            className = self.calcProbabilities(row_testing[:-1])

            if max(className, key=className.get) == row_testing[-1]:
                correct += 1

        return f'Bayes wynik: {correct} na {len(self.testing)}, {(correct / len(self.testing) * 100):.2f}%'