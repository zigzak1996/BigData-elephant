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

    public static final String MATCHED_WIKI_NAMES = "/AA*";

    public static class TokenizerMapper extends Mapper<Object, Text, Text, IntWritable> {

        private final IntWritable docId = new IntWritable();

        private final Text word = new Text();

        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            JSONObject json = new JSONObject(value.toString());

            docId.set(Integer.parseInt(json.getString("id")));

            String text = filterText(json.getString("text"));

            StringTokenizer itr = new StringTokenizer(text);

            while (itr.hasMoreTokens()) {

                String now = itr.nextToken();

                if (now.length() > 0) {
                    word.set(now);
                    context.write(word, docId);
                }

            }

        }
    }

    public static class Combiner extends Reducer<Text, IntWritable, Text, IntWritable> {

        private final IntWritable send = new IntWritable();

        private final Set<Integer> uniqueIds = new HashSet<>();

        public void reduce(Text key, Iterable<IntWritable> value, Context context)
                throws IOException, InterruptedException {

            for (IntWritable val : value) {
                uniqueIds.add(val.get());
            }

            for (Integer val : uniqueIds) {
                send.set(val);
                context.write(key, send);
            }

            uniqueIds.clear();
        }

    }

    public static class IntSumReducer extends Reducer<Text, IntWritable, Text, Text> {

        private static int index = 0;

        private final Set<Integer> uniqueIds = new HashSet<>();

        private static final String OUTPUT_FORMAT = "%1$d\t%2$d";

        public void reduce(Text key, Iterable<IntWritable> values, Context context)
                throws IOException, InterruptedException {

            for (IntWritable val : values) {
                uniqueIds.add(val.get());
            }

            Text result = new Text();

            result.set(String.format(OUTPUT_FORMAT, index++, uniqueIds.size()));

            context.write(key, result);

            uniqueIds.clear();
        }
    }

    static String filterText(final String rawText) {
        return rawText.toLowerCase()
                .replaceAll("[^a-z\\d\\s]", " ");
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Vocabulary");
        job.setJarByClass(Vocabulary.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(Combiner.class);
        job.setReducerClass(IntSumReducer.class);
        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0] + MATCHED_WIKI_NAMES));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }
}
