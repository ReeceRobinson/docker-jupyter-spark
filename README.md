# docker-jupyter-spark
A docker project for building a jupyter docker image with both pyspark and scala spark kernels.

# Build
docker build -t tracit/jupyter-spark-1.4.0 .

# Run
docker run -d -p 8888:8888 --name jupyter -v /[your notebook path]:/notebooks tracit/jupyter-spark-1.4.0
