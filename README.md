# Jupyter with Apache Spark, Scala and pySpark on Docker

This is for people who want a friendly ipython/Jupyter browser experience for working with Apache Spark.

Included in this docker image are both pyspark and scala spark kernels so you can choose which is right for you.

#Pull the image from Docker Repository

`pull reecerobinson/docker-jupyter-spark`

# Building the image

`docker build -t [tag] .`

# Running the image

`docker run -d --name jupyter -p 8888:8888 -v /[your notebook path]:/notebooks reecerobinson/docker-jupyter-spark:latest`

In your browser go to `http://[host]:8888` to view the notebook.

# Versions

ipython/Jupyter 3.2.0, Apache Spark 1.4.1, numpy 1.8.2, matplotlib 1.4.2