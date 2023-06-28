from main import *

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC


# Draw plot for KNN
csv = pd.read_csv('dataset/knnPlotData.csv')

feat = csv.columns
y = csv["target"]
X = csv[["x", "y"]].to_numpy()

def plot_circle():
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    radius = 1.5
    a = radius * np.cos(theta)
    b = radius * np.sin(theta)
    radius = 3
    aa = radius * np.cos(theta)
    bb = radius * np.sin(theta)
    figure, axes = plt.subplots(1)
    axes.plot(a, b, color="k")
    axes.plot(aa, bb, color="k")
    axes.set_aspect(1)
    plt.text(-0.45, -1.85, "$k = 3$", fontsize=14, color='k')
    plt.text(-0.45, -3.35, "$k = 7$", fontsize=14, color='k')

plt.rcParams['text.usetex'] = True
plot_circle()
plt.scatter(X[y == -1, 0], X[y == -1, 1], color='green', marker='o', label='class 1', s=210)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='o', label='class 1', s=210)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', label='class 2',s=210)
plt.axis([-3.7, 3.7, -3.7, 3.7])
plt.gca().axes.yaxis.set_ticklabels([])
plt.gca().axes.xaxis.set_ticklabels([])
plt.xlabel(feat[0])
plt.ylabel(feat[1])
plt.gca().tick_params(axis='x', which='both', length=0)
plt.gca().tick_params(axis='y', which='both', length=0)
ax = plt.gca()
ax.minorticks_on()
ax.set_xlabel('$x_2$', x=1, ha='right', fontsize=19)
ax.set_ylabel('$x_1$', y=1, ha='right', fontsize=19, rotation=0)

plt.savefig('img/plots/knn.png', dpi=200)
plt.close()


# Draw a plot for the SVM (the data is bad, so the graph also looks poor)
csv = pd.read_csv('dataset/svmPlotData.csv')

feat = csv.columns
y = csv["target"]
X = csv[["x", "y"]].to_numpy()

svm_clf = SVC(kernel='linear', C=10000000000000).fit(X, y)

def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    x1 = 0
    y1 = -b/w[1]
    x2 = 2
    y2 = (-w[0]/w[1])*x2 - b/w[1]

    tan_theta = (y2-y1)/(x2-x1)
    theta = np.arctan(tan_theta)*180/np.pi

    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]
    
    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.plot(x0, decision_boundary, "k-", linewidth=2, label="SVM")
    plt.text(1.16, 0.5, "$w * x - b = 1$", fontsize=16, rotation=34,color='green')
    plt.text(1.26, 0.4, "$w * x - b = 0$", fontsize=16, rotation=34,color='k')
    plt.text(1.36, 0.5, "$w * x - b = -1$", fontsize=16, rotation=34,color='red')
    plt.text(1.93, 0.6, r"$\leftarrow\frac{2}{||w||}\rightarrow$", fontsize=16, rotation=304,color='black')

    plt.plot(x0, gutter_up, "k--", linewidth=1.5)
    plt.plot(x0, gutter_down, "k--", linewidth=1.5)

    plt.fill_between(x0, gutter_up, gutter_down, where=gutter_up >= gutter_down, facecolor='#FFFFDE', interpolate=True)
    plt.fill_between(x0, gutter_up, gutter_down, where=gutter_up < gutter_down, facecolor='#FFFFDE', interpolate=True)
    plt.scatter(svs[:, 0], svs[:, 1], s=325, facecolors='black')

plt.rcParams['text.usetex'] = True
plot_svc_decision_boundary(svm_clf, 0, 3)
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='o', label='class 1', s=210)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', label='class 2',s=210)
plt.gca().axes.yaxis.set_ticklabels([])
plt.gca().axes.xaxis.set_ticklabels([])
plt.xlabel(feat[0])
plt.ylabel(feat[1])
plt.gca().tick_params(axis='x', which='both', length=0)
plt.gca().tick_params(axis='y', which='both', length=0)
ax = plt.gca()
ax.minorticks_on()
ax.set_xlabel('$x_2$', x=1, ha='right', fontsize=19)
ax.set_ylabel('$x_1$', y=1, ha='right', fontsize=19, rotation=0)

plt.savefig('img/plots/svm.png', dpi=200)