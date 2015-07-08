FROM java:8-jdk
MAINTAINER docker@reecerobinson.co.nz
 
ENV APACHE_SPARK_VERSION=1.4.0
 
RUN curl -s http://d3kbcqa49mib13.cloudfront.net/spark-1.4.0-bin-hadoop2.6.tgz | tar -xz -C /usr/local/
RUN cd /usr/local && ln -s spark-1.4.0-bin-hadoop2.6 spark
 
ENV SPARK_HOME /usr/local/spark
ENV PATH $PATH:$SPARK_HOME/bin

RUN     apt-get update && \
		apt-get install -y libc6 libc6-dev libc-dev udev && \
        apt-get install -y python-dev && \
        apt-get install -y python-pip && \
		apt-get install -y libfreetype6 libfreetype6-dev zlib1g-dev && \
		apt-get install -y python-matplotlib && \
		pip install py4j && \
		pip install numpy && \
   		pip install "ipython[All]" && \ 
        apt-get clean
 
COPY 	ipython /root/.ipython
COPY    kernels /root/.ipython/kernels
COPY	test_helper /tmp
RUN		cd /tmp && python setup.py install && rm -rf /tmp/*

RUN git clone https://github.com/ibm-et/spark-kernel.git
RUN echo "deb http://dl.bintray.com/sbt/debian /" | tee -a /etc/apt/sources.list.d/sbt.list && \
	apt-get update && \
	apt-get install --force-yes -y sbt
RUN cd spark-kernel && sbt compile && sbt pack && rm -rf /root/.ivy2
RUN (cd spark-kernel/kernel/target/pack && make install)

RUN     mkdir /notebooks
VOLUME ["/notebooks"]
EXPOSE 8888
 
# update boot script
COPY bootstrap.sh /etc/bootstrap.sh
RUN chown root.root /etc/bootstrap.sh
RUN chmod 700 /etc/bootstrap.sh

#CMD ["/bin/bash"]
ENTRYPOINT ["/etc/bootstrap.sh"]