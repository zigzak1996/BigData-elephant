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

class ParseDocument {

    public static class ParseMapper extends Mapper<Object, Text, Text, Text> {

        private final Text docId = new Text();

        private final Text docInfo = new Text();

        @Override
        protected void map(Object key, Text value, Context context) throws IOException, InterruptedException {

            JSONObject json = new JSONObject(value.toString());

            docId.set(json.getString("id"));

            docInfo.set(json.getString("url") + " " + json.getString("title"));

            context.write(docId, docInfo);
        }
    }

    public static class ParseReducer extends Reducer<Text, Text, Text, Text> {

        @Override
        protected void reduce(Text docId, Iterable<Text> docInfos, Context context) throws IOException, InterruptedException {

            for (Text docInfo : docInfos) {
                context.write(docInfo, docInfo);
            }

        }
    }

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
