from main import *

import matplotlib.pyplot as plt 
import statistics


csv = pd.read_csv('dataset/HeartDisease.csv')

# Basic KNN, Bayes, SVM tests
iterationNumber = 2
k = 3
minkowski = 2

knnValues = []

bayesValues= []
gaussianValues = []
multinomialValues = []
complementValues = []
bernoulliValues = []

svmLinearValues = []
svmPolyValues = []
svmRbfValues = []
svmSigmoidValues = []
for i in range(iterationNumber):
    training, testing = ProcessingData.SplitData(ProcessingData.NormalizeData(ProcessingData.ShuffleData(csv)))

    knnValues.append(KNN.knn(testing, training, minkowski, k))

    bayes = Bayes(training, testing)
    gaussian, multinomial, complement, bernoulli = BayesSK.bayes(training, testing)
    bayesValues.append(bayes.bayes())
    gaussianValues.append(gaussian)
    multinomialValues.append(multinomial)
    complementValues.append(complement)
    bernoulliValues.append(bernoulli)

    svmLinearValues.append(SVM.svm(training, testing, "linear"))
    svmPolyValues.append(SVM.svm(training, testing, "poly"))
    svmRbfValues.append(SVM.svm(training, testing, "rbf"))
    svmSigmoidValues.append(SVM.svm(training, testing, "sigmoid"))

print(f"Iterations: {iterationNumber}")
print(f"KNN result: {statistics.mean(knnValues) * 100:.2f}%, m={minkowski}, k={k}")
print(f"Bayes result: {statistics.mean(bayesValues) * 100:.2f}% (own),",
      f"{statistics.mean(gaussianValues) * 100:.2f}% (gaussian),",
      f"{statistics.mean(multinomialValues) * 100:.2f}% (multinomial),",
      f"{statistics.mean(complementValues) * 100:.2f}% (complement)",
      f"{statistics.mean(bernoulliValues) * 100:.2f}% (bernoulli)")
print(f"SVM result: {statistics.mean(svmLinearValues) * 100:.2f}% (linear),",
      f"{statistics.mean(svmPolyValues) * 100:.2f}% (poly),",
      f"{statistics.mean(svmRbfValues) * 100:.2f}% (rbf),",
      f"{statistics.mean(svmSigmoidValues) * 100:.2f}% (sigmoid)")


# Plot accuracy of knn depending on number of neighbours
iterationNumber = 1
neighboursNumber = 5
minkowski = 2

knn = []
for i in range(1, neighboursNumber + 1):
    knnValues = []
    for _ in range(iterationNumber):
        training, testing = ProcessingData.SplitData(ProcessingData.NormalizeData(ProcessingData.ShuffleData(csv)))
        knnValues.append(KNN.knn(testing, training, minkowski, i))

    k = statistics.mean(knnValues)
    knn.append(k)
    print("knn: ", k, "k=", i)

plt.plot([k for k in range(1,neighboursNumber + 1)], knn, color='blue')
for i in range(1, neighboursNumber + 1):
    plt.text(i, knn[i-1], f'{knn[i-1] * 100:.2f}%')

plt.xticks([i for i in range(1, neighboursNumber + 1)])
plt.xlabel('Number of neighbours (k)', color='Black', weight='bold', fontsize='12')
plt.ylabel('Accuracy', color='Black', weight='bold', fontsize='12')
plt.title('Accuracy of the kNN classifier depending on the number of neighbours', color='Black', weight='bold', fontsize='12')
plt.show()


svmDist = [statistics.mean(svmLinearValues), statistics.mean(svmPolyValues),  statistics.mean(svmRbfValues), statistics.mean(svmSigmoidValues)]

fig, ax = plt.subplots(figsize=(7.6, 4.8))
ax.bar(["linear", "poly", "rbf", "sigmoid"], svmDist, color ='steelblue')
ax.set_ylim(min(svmDist) - 0.01, max(svmDist) + 0.01)

for i, v in enumerate(svmDist):
    ax.text(i, v + 0.001, f"{v *100:.2f}%", color='black', ha='center')

plt.xlabel("Kernel functions", color='Black', weight='bold', fontsize='12')
plt.ylabel("Accuracy", color='Black', weight='bold', fontsize='12')
plt.title("Accuracy of the SVM classifier depending on the kernel functions", color='Black', weight='bold', fontsize='12')
plt.show()



bayesDist = [statistics.mean(gaussianValues), statistics.mean(multinomialValues),  statistics.mean(complementValues), statistics.mean(bernoulliValues)]

fig, ax = plt.subplots(figsize=(7.6, 4.8))
ax.bar(["gaussian", "multinomial", "complement", "bernoulli"], bayesDist, color ='steelblue')
ax.set_ylim(min(bayesDist) - 0.01, max(bayesDist) + 0.01)

for i, v in enumerate(bayesDist):
    ax.text(i, v + 0.001, f"{v *100:.2f}%", color='black', ha='center')

plt.xlabel('Probability distributions', color='Black', weight='bold', fontsize='12')
plt.ylabel('Accuracy', color='Black', weight='bold', fontsize='12')
plt.title('Naive Bayes accuracy depending on the probability distributions', color='Black', weight='bold', fontsize='12')
plt.show()