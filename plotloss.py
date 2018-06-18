import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

with open('./lossRecord/instrumentloss.txt') as f:
    content = f.readlines()
    content = [x.strip() for x in content]
    plotcon = np.around(np.array(content).astype(float),2)
    sns.set_style("darkgrid")
    plt.plot(plotcon)
    plt.xticks()
    plt.yticks()
    plt.ylim(0,6)
    #plt.show()
    #time.sleep(100)

    lr = linear_model.LinearRegression()
    x = np.arange(plotcon.shape[0]).reshape(-1, 1)
    y = plotcon
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    y_pred = regr.predict(x)
    # plt.scatter(x, y,  color='black')
    plt.plot(x, y_pred, color='blue', linewidth=3)
    plt.show()


