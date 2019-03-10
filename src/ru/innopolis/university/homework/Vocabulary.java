
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.json.JSONObject;

import java.io.IOException;
import java.util.HashSet;
import java.util.Set;
import java.util.StringTokenizer;

public class Vocabulary {

    public static class TokenizerMapper
            extends Mapper<Object, Text, Text, IntWritable> {

//        private final static IntWritable one = new IntWritable(1);

        private final static IntWritable docId = new IntWritable();

        private Text word = new Text();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            JSONObject json = new JSONObject(value.toString());

            docId.set(Integer.parseInt(json.getString("id")));

            StringTokenizer itr = new StringTokenizer(json.getString("text"));

            while (itr.hasMoreTokens()) {

                String now = itr.nextToken().toLowerCase();

                String parseWord = "";

                for (int i = 0; i < now.length(); i++) {

                    if ((now.charAt(i) >= 'a' && now.charAt(i) <= 'z') || now.charAt(i) == '-') {

                        parseWord += now.charAt(i);

                    }
                    else {
                        continue;
                    }
                }
                if (parseWord.length() > 0){
                    word.set(parseWord);
                    context.write(word, docId);
                }

            }

        }
    }

    public static class Combiner extends Reducer<Text, IntWritable, Text, IntWritable>
    {
        public void reduce(Text key,Iterable<IntWritable> value, Context context) throws IOException, InterruptedException
        {
            for (IntWritable v : value)
                context.write(key,v);
        }

    }

    public static class IntSumReducer
            extends Reducer<Text, IntWritable,Text,Text> {
        private static int index = 0;

        private final Set<Integer> uniqueIds = new HashSet<>();

        public void reduce(Text key, Iterable<IntWritable> values,
                           Context context
        ) throws IOException, InterruptedException {
            for (IntWritable val : values) {
                uniqueIds.add(val.get());
            }

            Text result = new Text();

            result.set(String.format("%1$d\t%2$d", index++, uniqueIds.size()));

            context.write(key, result);

            uniqueIds.clear();
        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "word count");
        job.setJarByClass(Vocabulary.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(Combiner.class);
        job.setReducerClass(IntSumReducer.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
