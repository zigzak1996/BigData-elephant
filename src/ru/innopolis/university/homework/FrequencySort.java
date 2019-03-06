import java.io.IOException;
import java.util.StringTokenizer;
import java.lang.Integer;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

public class FrequencySort {

  public static class TokenizerMapper
	   extends Mapper<Object, Text, IntWritable, Text>{


	public void map(Object key, Text value, Context context
					) throws IOException, InterruptedException {
	  StringTokenizer itr = new StringTokenizer(value.toString());
	  while (itr.hasMoreTokens()) {
		String frequency = itr.nextToken();
		String w = itr.nextToken();
		context.write(new IntWritable(Integer.parseInt(w)), new Text(frequency));
	  }
	}
  }

  public static class WordReducer
		  extends Reducer<IntWritable,Iterable<Text>,Text,IntWritable> {

  	private IntWritable result = new IntWritable();

	public void reduce(IntWritable keys, Iterable<Text> values, Context context)
			throws IOException, InterruptedException {
	  for (Text value : values) {
		context.write(value, keys);
	  }

	}
  }

  public static void main(String[] args) throws Exception {
	Configuration conf = new Configuration();
	Job job = Job.getInstance(conf, "word count");
	job.setJarByClass(FrequencySort.class);
	job.setMapperClass(TokenizerMapper.class);
	job.setCombinerClass(WordReducer.class);
	job.setReducerClass(WordReducer.class);
	job.setOutputKeyClass(IntWritable.class);
	job.setOutputValueClass(Text.class);
	FileInputFormat.addInputPath(job, new Path(args[0]));
	FileOutputFormat.setOutputPath(job, new Path(args[1]));
	System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}