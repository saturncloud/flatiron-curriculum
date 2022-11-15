# Hadoop MapReduce Recap

### Objectives

- Reviewing Hadoop MapReduce Code in Java to further see the difference in PySpark

### Glossary 

- primary node : Primary node in a Hadoop cluster is responsible for delegating where to store files, the health status check of secondary nodes, and job tracking. There is only one primary node for each cluster.
- secondary node : Secondary node, or the worker node, is where the jobs are run. The secondary node receives job information from the primary node. There can be multiple secondary nodes per cluster.

### Introduction

MapReduce is an algorithm that sits on top of Hadoop ecosystem to handle big data processing through distributed computing. There are two important phases of MapReduce, which you can tell from the name, Map and Reduce. The mapper phase takes the input and converts it into key/value pairs. The reducer phase takes the output from the mapper phase as the input then aggregates the key/value pairs into even smaller tuples. The MapReduce job will be in sequence, as the name entails, the mapper job will always run before the reducer job.

The major advantage of MapReduce is that it is easy to scale jobs through distributed computing over multiple computing nodes. This simple scalability attracted many businesses to use the MapReduce model.


### Sample Code

We'll run through the sample code with the sample input data.

```java

package hadoop;

import java.util.*; 

import java.io.IOException; 
import java.io.IOException; 

import org.apache.hadoop.fs.Path; 
import org.apache.hadoop.conf.*; 
import org.apache.hadoop.io.*; 
import org.apache.hadoop.mapred.*; 
import org.apache.hadoop.util.*; 

public class SampleHadoopProcess {
	public static class SampleMapper extends MapReduceBase implements 
	Mapper<LongWritable, /*Input key Type */ 
	Text,                /*Input value Type*/ 
	Text,                /*Output key Type*/ 
	IntWritable>         /*Output value Type*/ 
	{

		public void map(LongWritable key, Text value, OutputCollector<Text, IntWritable> output, Reporter reporter) throws IOException {
			String line = value.toString();
			String lastToken = null;
			StringTokenizer s = new StringTokenizer(line, "\t");
			String key = s.nextToken();

			while (s.hasMoreTokens()) {
				lastToken = s.nextToken();
			}

			int avgPrice = Integer.parseInt(lastToken);
			output.collect(new Text(key));
		}

	public static class SampleReducer extens MapReduceBase implements Reducer<Text, IntWritable, Text, IntWritable> {
		public void reduce(Text key, Iterator <IntWritable> values, OutputCollector<Text, IntWritable> output, Reporter reporter) throws IOException { 
			int maxavg = 30; 
			int val = Integer.MIN_VALUE; 
            
        	while (values.hasNext()) { 
            	if ((val = values.next().get())>maxavg) { 
            		output.collect(key, new IntWritable(val)); 
            	} 
        	}
        } 
	}

	public static void main(String args[]) throws Exception {
		JobConf conf = new JobConf(SampleHadoopProcess.class);

		//sets the job name, this can be anything
		conf.setJobNames("hadoop_sample_process");
		//from the reducer class
		conf.setOutputKeyClass(Text.class);
		conf.setOutputValueClass(IntWritable);
		conf.setMapperClass(SampleMapper.class);
		conf.setCombinerClass(SampleReducer.class);
		conf.setReducerClass(SampleReducer.class);
		conf.setInputFormat(TextInputFormat.class);
		conf.setOutputFormat(TextOutputFormat.class);

		// we'll specify the input/output paths when we execute the jar file
		FileInputFormat.setInputPaths(conf, new Path(args[0]));
		FileOutputFormat.setOutputPaths(conf, new Path(args[1]));

		JobClient.runJob(conf);
	}
}
```

### Execution

#### Step 1

Compile the sample code using the `javac` command

```
javac -classpath hadoop-core-1.2.1.jar -d hadoop_process SampleHadoopProcess.java
jar -cvf hadoop_process.jar -C 
```

#### Step 2

We're assuming the input file already exists in the `~/input_dir/`. We can run the jar file using

```
$HADOOP_HOME/bin/hadoop jar hadoop_process.jar hadoop.SampleHadoopProcess input_dir output_dir
```



### Summary

The logs show pretty interesting output, especially which and how many CPUs it used. It also shows how much CPU time it used. Depending on the file input size, you'll be able to run some tests on how fast it can compute. Another way is to use different file formats, like avro or parquet to read and write much faster to reduce the CPU time. 
