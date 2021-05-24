import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data.csv")

mean = []
stdPos = []
stds = []
stdNeg = []
meanX = []
stdX = []
best = []
for i in range(len(data.columns) - 1):
    x = [i] * data.shape[0]
    y = data[str(i)].tolist()
    maxScore = max(y)
    best.append(maxScore)
    meanScore = np.mean(y)
    standardDeviation = np.std(y)
    stds.append(standardDeviation)
    stdPos.append(meanScore + standardDeviation)
    stdNeg.append(meanScore - standardDeviation)
    mean.append(meanScore)
    meanX.append(i)
    stdX.append(i)
    size = []
    for j in y:
        counter = y.count(j)
        size.append(counter)
    size = [n * 1 for n in size]
    plt.scatter(x,y,s=size,c='C0')

print(np.mean(best))
print(np.mean(meanScore))
print(np.mean(standardDeviation))

plt.title("Fitness plot for the GA")
plt.ylabel("Fitness score")
plt.xlabel("Generations")
plt.show()

plt.plot(meanX, mean, 'o-',markersize=2)
plt.fill_between(stdX, stdNeg, stdPos, alpha=.1)

plt.title("Mean and Standard deviation plot for GA")
plt.ylabel("Fitness score")
plt.xlabel("Generations")
plt.show()