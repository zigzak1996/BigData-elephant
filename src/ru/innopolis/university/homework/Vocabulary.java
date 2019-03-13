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

    public static final String MATCHED_WIKI_NAMES = "/AA*"; //Special prefix for taking only documents which starts with AA*.

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

    /***
     * method for text processing with regular expression.
     *
     * @param rawText - text, which is going to be preprocessed and filtered by the regular expression.
     * @return lowercase text which contains only Latin letters, digits and spaces.
     */
    static String filterText(final String rawText) {
        return rawText.toLowerCase()
                .replaceAll("[^a-z\\d\\s]", " ");
    }

    /***
     * Method, which starts job by vocabulary creation, and after successful finishing of the job,
     * starts job by saving documents' information (id, title, URL) into the separate file. {@link ParseDocument}
     *
     * @param args: String array with arguments:
     *                  args[0] - Path to documents, which we are going to parse.
     *                  args[1] - Path into which vocabulary will be saved.
     *                  args[2] - Path into which documents' information will be saved.
     */
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

        boolean isSuccess = job.waitForCompletion(true); // is previous job finished successfully?

        if (isSuccess) {
            ParseDocument.startParsingDocuments(conf, args[0], args[2]); // Starting saving title, url, ids from docs.
        } else {
            System.exit(1); //UnSuccess closing
        }

    }
}
