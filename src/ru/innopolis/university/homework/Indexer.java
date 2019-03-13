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

/***
 * This class creates separate file which contains TF of a word in concrete document. There are only words which
 * are in dictionary. Created file consist of next structure: docId - wordId - word TF for this docId. It will be
 * needed for calculating document's relevance when processing query.
 */
public class Indexer {

    public static class TFMapper extends Mapper<Object, Text, IntWritable, IntWritable> {

        private static final Map<String, Integer> vocab = new HashMap<>();

        private final IntWritable docId = new IntWritable();

        private final IntWritable wordId = new IntWritable();

        /***
         * Here vocabulary is read from the file and written into the HashMap, where specific word is key, and value is
         * specific wordId. Path to vocabulary is received here via configuration.
         *
         * @param context - MapReduce container.
         */
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

        /***
         * This method parse specific document, filter its text with regular expression (its in {@link Vocabulary}) and
         * writes in context pairs: (docId, wordId) if specific word in the document is met in vocabulary.
         *
         * @param key - unused.
         * @param document - document in JSON format.
         * @param context - MapReduce container.
         */
        @Override
        public void map(Object key, Text document, Context context) throws IOException, InterruptedException {

            JSONObject json = new JSONObject(document.toString());

            String text = Vocabulary.filterText(json.getString("text"));

            StringTokenizer itr = new StringTokenizer(text);

            docId.set(json.getInt("id"));

            while (itr.hasMoreTokens()) {

                String word = itr.nextToken();

                if (vocab.containsKey(word)) {

                    wordId.set(vocab.get(word));

                    context.write(docId, wordId);
                }
            }
        }
    }

    public static class TFReducer extends Reducer<IntWritable, IntWritable, IntWritable, Text> {

        private static final String OUTPUT_FORMAT = "%s\t%d";

        /***
         * Here is calculation of how much every specific word was met in the document
         * with received docId (TF calculation). After calculations every word id and TF is written in the context
         * where the key is docId.
         *
         * @param docId - id of specific document.
         * @param wordIds - array of identifiers of words, that were met in the document with received id.
         * @param context - MapReduce container.
         */
        @Override
        public void reduce(IntWritable docId, Iterable<IntWritable> wordIds, Context context)
                throws IOException, InterruptedException {

            final Map<String, Integer> wordsTFcount = new HashMap<>();

            for (IntWritable wordId : wordIds) {  // put here wordId and count of wordId which met in spec. corpus (TF)
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

    /***
     * Method, which starts job by calculating TF of every word which is met in the document and written in vocabulary.
     *
     * @param args: String array with arguments:
     *                  args[0] - Path to vocabulary that creates in {@link Vocabulary}.
     *                  args[1] - Path to documents, which we are going to parse.
     *                  args[2] - Path into which indexer(file with word's TF for each document) created here will be
     *                            saved.
     */
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
