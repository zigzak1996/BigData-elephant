import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.DoubleWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;


public class Query {

    private static final Map<String, Integer> vocab = new HashMap<>();
    private static final Map<Integer, Double> query_map = new HashMap<>();
    private static final Map<Integer, Double> idf = new HashMap<>();

    private static String query = "the good person";

    private static final int DEFAULT_RELEVANT_PAGE_COUNT = 10;

    private static int relevancePagesCount = DEFAULT_RELEVANT_PAGE_COUNT;

    private static void setArguments(final String rawPagesCount, final String rawQuery) {

        if (rawPagesCount != null && !rawPagesCount.isEmpty()) {
            try {
                relevancePagesCount = Integer.parseInt(rawPagesCount);
            } catch (NumberFormatException ex) {
                relevancePagesCount = DEFAULT_RELEVANT_PAGE_COUNT;
            }
        }

        if (rawQuery != null && !rawQuery.isEmpty()) {  // if specific query was entered - set it, otherwise use default
            query = rawQuery;
        }
    }

    private static void translateRequest() {
        StringTokenizer itr = new StringTokenizer(Vocabulary.filterText(query));
        while (itr.hasMoreTokens()) {

            String word = itr.nextToken();

            if (vocab.containsKey(word)) {  //map - word_id: it's count in query
                query_map.merge(vocab.get(word), 1.0, Double::sum);
            }
        }
    }

    private static void createVocabularyAndIdf(final String wayToVoc) throws IOException {
        Path pathToVoc = new Path(wayToVoc);
        FileSystem fileSystem = FileSystem.get(new Configuration());

        try (BufferedReader bf = new BufferedReader(new InputStreamReader(fileSystem.open(pathToVoc)))) {

            String line = bf.readLine();
            while (line != null) {
                StringTokenizer itr = new StringTokenizer(line);
                String word = itr.nextToken();
                int wordId = Integer.parseInt(itr.nextToken());
                double wordIdf = Double.parseDouble(itr.nextToken());

                vocab.putIfAbsent(word, wordId);
                idf.putIfAbsent(wordId, wordIdf);

                line = bf.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }

    public static class FrequencyMapper
            extends Mapper<Object, Text, IntWritable, DoubleWritable> {

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {

            StringTokenizer itr = new StringTokenizer(value.toString());

            int doc_id = Integer.parseInt(itr.nextToken());
            int word_id = Integer.parseInt(itr.nextToken());
            double doc_tf = Double.parseDouble(itr.nextToken());
            Configuration conf = context.getConfiguration();
            String raw_query_tf = conf.get(String.valueOf(word_id), "");  // извлечение TF по Query wordId
            String raw_idf = conf.get(word_id + "C", "");   // извлечение Idf по WordID + C
            if (!raw_query_tf.isEmpty() && !raw_idf.isEmpty()) {
                double query_tf = Double.parseDouble(raw_query_tf);
                double query_idf = Double.parseDouble(raw_idf);

                double doc_tf_idf = doc_tf / query_idf;
                double query_tf_idf = query_tf / query_idf;
                context.write(new IntWritable(doc_id), new DoubleWritable(doc_tf_idf * query_tf_idf));
            }
        }
    }

    public static class IntSumReducer
            extends Reducer<IntWritable, DoubleWritable, Text, DoubleWritable> {

        final DoubleWritable result = new DoubleWritable();

        final Text word = new Text();

        public void reduce(IntWritable key, Iterable<DoubleWritable> value, Context context)
                throws IOException, InterruptedException {

            double sum = 0;

            for (DoubleWritable val : value) {
                sum += val.get();
            }

            result.set(sum);
            word.set(key.get() + "");
            context.write(word, result);

        }
    }

    public static class IntSumCombiner
            extends Reducer<IntWritable, DoubleWritable, IntWritable, DoubleWritable> {

        final DoubleWritable result = new DoubleWritable();

        public void reduce(IntWritable doc_id, Iterable<DoubleWritable> value, Context context)
                throws IOException, InterruptedException {
            double sum = 0;
            for (DoubleWritable val : value) {
                sum += val.get();
            }
            result.set(sum);
            context.write(doc_id, result);
        }

    }

    public static void main(String[] args) throws Exception {
        final Configuration conf = new Configuration();

        setArguments(args[4], args[5]);
        createVocabularyAndIdf(args[0]);
        translateRequest();

        query_map.forEach((wordId, countInQuery) -> {
            conf.set(wordId.toString(), countInQuery.toString());
            conf.set(wordId.toString() + "C", idf.get(wordId).toString());
        });

        conf.setInt("$size$", relevancePagesCount);
        conf.set("$path_to_parse$", args[3]);
        Job processQueryJob = Job.getInstance(conf, "Query");
        processQueryJob.setJarByClass(Query.class);
        processQueryJob.setMapperClass(FrequencyMapper.class);
        processQueryJob.setCombinerClass(IntSumCombiner.class);
        processQueryJob.setReducerClass(IntSumReducer.class);

        processQueryJob.setMapOutputKeyClass(IntWritable.class);
        processQueryJob.setMapOutputValueClass(DoubleWritable.class);
        processQueryJob.setOutputKeyClass(Text.class);
        processQueryJob.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(processQueryJob, new Path(args[1]));
        FileOutputFormat.setOutputPath(processQueryJob, new Path(args[2]));

        System.exit(processQueryJob.waitForCompletion(true) ? 0 : 1);

    }

}