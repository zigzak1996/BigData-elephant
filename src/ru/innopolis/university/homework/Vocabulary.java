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


/***
 * @TokenizerMapper is responsible for text extraction from json file by using its id.
 * Afterwards it processes it by regular expression method filter and saves each word with appropriate id of a document.
 * Finally it cleans dataset from empty places (ex. empty line)
 */
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

        private final IntWritable document_id = new IntWritable();

        private final Set<Integer> uniqueIds = new HashSet<>();

        public void reduce(Text word, Iterable<IntWritable> document_ids, Context context)
                throws IOException, InterruptedException {

            for (IntWritable val : document_ids) {
                uniqueIds.add(val.get());
            }

            for (Integer val : uniqueIds) {
                document_id.set(val);
                context.write(word, document_id);
            }

            uniqueIds.clear();
        }

    }

    /***
     * By using Combiner we are trying to decrase load on the Reducer by compressing repetitive words and doc ids
     * to one word per document.
     *
     * @DocumentReducer obtains Combiner output and makes final assigning of words' Ids and IDFs accordingly
     */

    public static class DocumentReducer extends Reducer<Text, IntWritable, Text, Text> {

        private static int wordId = 0;

        private final Set<Integer> uniqueDocIds = new HashSet<>();

        public void reduce(Text word, Iterable<IntWritable> document_id, Context context)
                throws IOException, InterruptedException {

            for (IntWritable val : document_id) {
                uniqueDocIds.add(val.get());
            }

            Text result = new Text();

            result.set(wordId++ +""+uniqueDocIds.size())

            context.write(word, result);

            uniqueDocIds.clear();
        }
    }

    /***
     * method for text processing with regular expression.
     *
     * @param rawText - text, which is going to be preprocessed and filtered by the regular expression.
     * @return lowercase text without any punctuation and specific symbols, and without words that consist only
     * digits, and words where some letter or digit repeats at least 4 times.
     */
    static String filterText(final String rawText) {
        return rawText.toLowerCase()
                .replaceAll("\\s*\\b(?=[a-z\\d]*([a-z\\d])\\1{3}|\\d+\\b)[a-z\\d]+|[^a-z\\d\\s]", " ");
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
