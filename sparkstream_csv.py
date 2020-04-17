import findspark
findspark.init()
import os
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.streaming import StreamingContext

os.chdir("D:\\trentsemester2\\bigData\\the-movies-dataset")
sparkconfig = SparkConf().setAppName("stream").setMaster("local[*]")
sc = SparkContext(conf=sparkconfig)
sc.setLogLevel("ERROR")
ssc = StreamingContext(sc, 10)
# initiate streaming text from a TCP (socket) source:
socket_stream = ssc.socketTextStream("127.0.0.1", 5555)
# lines of tweets with socket_stream window of size 60, or 60 #seconds windows of time
rdds = socket_stream.window(10)

rdds.saveAsTextFiles("tweets.csv")
rdds.pprint()

ssc.start()            # Start the computation
ssc.awaitTermination()

print("end of code")