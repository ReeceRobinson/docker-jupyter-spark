# Jupyter with Apache Spark, Scala and pySpark on Docker

This is for people who want a friendly ipython/Jupyter browser experience for working with Apache Spark.

Included in this docker image are both pyspark and scala spark kernels so you can choose which is right for you.

# Pull the image from Docker Repository

`pull reecerobinson/docker-jupyter-spark`

# Building the image

`docker build -t [tag] .`

# Running the image

`docker run -d --name jupyter -p 8888:8888 -v /[your notebook path]:/notebooks reecerobinson/docker-jupyter-spark:latest`

In your browser go to `http://[host]:8888` to view the notebook.

# Versions

ipython/Jupyter 4, Apache Spark 1.5.1, numpy 1.8.2, matplotlib 1.4.2, pandas 0.16.2, bokeh 0.9.2, scikit-learn 0.16.1, scipy 0.14.0

# Bokeh Example
To use Bokeh in a notebook you need to create the output for inline display. To achieve this you use the `output_notebook()` feature of the `bokeh.io` API.

```python
import numpy as np
from bokeh.plotting import figure, show
from bokeh.io import output_notebook
N = 4000
x = np.random.random(size=N) * 100
y = np.random.random(size=N) * 100
radii = np.random.random(size=N) * 1.5
colors = ["#%02x%02x%02x" % (r,g,150) for r,g in zip(np.floor(50+2*x), np.floor(30+2*y))]
output_notebook()
p = figure()
p.circle(x,y,radius=radii, fill_color=colors,fill_alpha=0.6, line_color=None)
show(p)
```

# matplotlib Example from Scikit Learn

```python
from sklearn import datasets
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validated:
predicted = cross_val_predict(lr, boston.data, y, cv=10)
plt.close()
fig,ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()
pass
```

# pyspark matplotlib example

```python
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import log

# function for generating plot layout
def preparePlot(xticks, yticks, figsize=(10.5, 6), hideLabels=False, gridColor='#999999', gridWidth=1.0):
    plt.close()
    fig, ax = plt.subplots(figsize=figsize, facecolor='white', edgecolor='white')
    ax.axes.tick_params(labelcolor='#999999', labelsize='10')
    for axis, ticks in [(ax.get_xaxis(), xticks), (ax.get_yaxis(), yticks)]:
        axis.set_ticks_position('none')
        axis.set_ticks(ticks)
        axis.label.set_color('#999999')
        if hideLabels: axis.set_ticklabels([])
    plt.grid(color=gridColor, linewidth=gridWidth, linestyle='-')
    map(lambda position: ax.spines[position].set_visible(False), ['bottom', 'top', 'left', 'right'])
    return fig, ax

# generate layout and plot data
x = range(1, 50)
y = [log(x1 ** 2) for x1 in x]
fig, ax = preparePlot(range(5, 60, 10), range(0, 12, 1))
plt.scatter(x, y, s=14**2, c='#d6ebf2', edgecolors='#8cbfd0', alpha=0.75)
ax.set_xlabel(r'$range(1, 50)$'), ax.set_ylabel(r'$\log_e(x^2)$')
pass
```
