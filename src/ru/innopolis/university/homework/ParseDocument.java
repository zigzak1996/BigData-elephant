import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.json.JSONObject;

import java.io.IOException;

/***
 * Document parser saves documents' information: id, URL, title - into a separate file in order to extract it after
 * processing a query and write the information about most relevant documents.
 */
class ParseDocument {

    public static class ParseMapper extends Mapper<Object, Text, Text, Text> {

        private static final String MAP_OUTPUT_FORMAT = "%1$s %2$s";

        private final Text docId = new Text();

        private final Text docInfo = new Text();

        /***
         * Parse each document and write it's information to context by docId.
         *
         * @param key - unused.
         * @param document - document in JSON format.
         * @param context - MapReduce container.
         */
        @Override
        protected void map(Object key, Text document, Context context) throws IOException, InterruptedException {

            JSONObject json = new JSONObject(document.toString());

            docId.set(json.getString("id"));

            docInfo.set(String.format(MAP_OUTPUT_FORMAT, json.getString("url"), json.getString("title")));

            context.write(docId, docInfo);
        }
    }

    public static class ParseReducer extends Reducer<Text, Text, Text, Text> {

        /***
         * Passes docId, url and title into the context in order to save it on the disk.
         *
         * @param docId - id of concrete document, key.
         * @param docInfos - URL & title of document, written in 1 string.
         * @param context - MapReduce container.
         */
        @Override
        protected void reduce(Text docId, Iterable<Text> docInfos, Context context)
                throws IOException, InterruptedException {

            for (Text docInfo : docInfos) {
                context.write(docId, docInfo);
            }

        }
    }

    /***
     * This method starts MapReduce job for saving documents' information (only for docs which names begin from AA).
     *
     * @param conf - MapReduce job configuration.
     * @param pathToWiki - Path to documents, which we are going to parse.
     * @param parserOutput - directory into which results are saved.
     */
    static void startParsingDocuments(final Configuration conf,
                                      final String pathToWiki,
                                      final String parserOutput)
            throws IOException, ClassNotFoundException, InterruptedException {

        Job parseJob = Job.getInstance(conf, "Parse jsons");
        parseJob.setJarByClass(Vocabulary.class);
        parseJob.setMapperClass(ParseMapper.class);
        parseJob.setReducerClass(ParseReducer.class);

        parseJob.setMapOutputKeyClass(Text.class);
        parseJob.setMapOutputValueClass(Text.class);
        parseJob.setOutputKeyClass(Text.class);
        parseJob.setOutputValueClass(Text.class);

        FileInputFormat.addInputPath(parseJob, new Path(pathToWiki + Vocabulary.MATCHED_WIKI_NAMES));
        FileOutputFormat.setOutputPath(parseJob, new Path(parserOutput));
        System.exit(parseJob.waitForCompletion(true) ? 0 : 1);
    }

}
