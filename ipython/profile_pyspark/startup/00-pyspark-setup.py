import os
import sys

os.environ["SPARK_CLASSPATH"] = '/usr/local/spark/lib/postgresql-9.4-1201.jdbc41.jar'
spark_home = os.environ.get('SPARK_HOME', None)
if not spark_home:
    raise ValueError('SPARK_HOME environment variable is not set')
sys.path.insert(0, os.path.join(spark_home, 'python'))
sys.path.insert(0, os.path.join(spark_home, 'python/lib/py4j-0.8.2.1-src.zip'))

execfile(os.path.join(spark_home, 'python/pyspark/shell.py'))