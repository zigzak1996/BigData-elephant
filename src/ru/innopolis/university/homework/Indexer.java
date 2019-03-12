import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;

public class Indexer {

    public static class TFMapper extends Mapper<Object, Text, IntWritable, IntWritable> {

        private static final Map<String, Integer> vocab = new HashMap<>();

        private final IntWritable docIds = new IntWritable();

        private final IntWritable wordIds = new IntWritable();

        @Override
        protected void setup(Context context) throws IOException, InterruptedException {
            super.setup(context);

            Configuration conf = context.getConfiguration();

            String raw_path_to_vocabulary = conf.get("path_to_vocabulary");

            Path path_to_vocabulary = new Path(raw_path_to_vocabulary);
            FileSystem fileSystem = FileSystem.get(conf);

            try (BufferedReader bf = new BufferedReader(new InputStreamReader(fileSystem.open(path_to_vocabulary)))) {

                String line = bf.readLine();

                while (line != null) {
                    StringTokenizer itr = new StringTokenizer(line);

                    String word = itr.nextToken();

                    Integer word_id = Integer.parseInt(itr.nextToken());

                    vocab.putIfAbsent(word, word_id);

                    line = bf.readLine();
                }

            } catch (IOException ex) {
                System.out.println("Error reading file!");
                ex.printStackTrace();
            }

        }

        @Override
        public void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            JSONObject json = new JSONObject(value.toString());

            String text = Vocabulary.filterText(json.getString("text"));

            StringTokenizer itr = new StringTokenizer(text);

            int docId = json.getInt("id");

            docIds.set(docId);

            while (itr.hasMoreTokens()) {

                String word = itr.nextToken();

                if (vocab.containsKey(word)) {

                    wordIds.set(vocab.get(word));

                    context.write(docIds, wordIds);
                }
            }
        }
    }

    public static class TFReducer extends Reducer<IntWritable, IntWritable, IntWritable, Text> {

        private static final String OUTPUT_FORMAT = "%s\t%d";

        @Override
        public void reduce(IntWritable docId, Iterable<IntWritable> wordIds, Context context)
                throws IOException, InterruptedException {

            final Map<String, Integer> wordsTFcount = new HashMap<>();

            for (IntWritable wordId : wordIds) {  // put here wordId and count wordId met in spec. corpus (TF)
                wordsTFcount.merge(String.valueOf(wordId), 1, Integer::sum);
            }

            Text wordIdAndTF = new Text();

            for (String wordId : wordsTFcount.keySet()) {
                wordIdAndTF.set(String.format(OUTPUT_FORMAT, wordId, wordsTFcount.get(wordId)));
                context.write(docId, wordIdAndTF);
                wordIdAndTF.clear();
            }

        }
    }

    public static void main(String[] args) throws Exception {
        Configuration conf = new Configuration();
        conf.set("path_to_vocabulary", args[0]);
        Job job = Job.getInstance(conf, "Indexer");
        job.setJarByClass(Indexer.class);
        job.setMapperClass(TFMapper.class);
        job.setReducerClass(TFReducer.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(IntWritable.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(args[1] + Vocabulary.MATCHED_WIKI_NAMES));
        FileOutputFormat.setOutputPath(job, new Path(args[2]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }

}
