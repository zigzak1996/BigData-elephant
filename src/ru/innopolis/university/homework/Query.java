import org.apache.hadoop.conf.Configuration;
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
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;


public class OldQuery {

    public static Map<String, Integer> vocab = new HashMap<>();
    public static Map<Integer, Double> query_map = new HashMap<>();
    public static Map<Integer, Double> idf = new HashMap<>();

    public static String query = "the good person";

    private static void translateRequest() {
        StringTokenizer itr = new StringTokenizer(query);
        while (itr.hasMoreTokens()) {

            String now = itr.nextToken().toLowerCase();

            String parseWord = "";

            for (int i = 0; i < now.length(); i++) {

                if ((now.charAt(i) >= 'a' && now.charAt(i) <= 'z') || now.charAt(i) == '-') {

                    parseWord += now.charAt(i);

                }
            }
            if (vocab.containsKey(parseWord)) {
                if (query_map.containsKey(vocab.get(parseWord))) {
                    double tmp = query_map.get(vocab.get(parseWord)) + 1.0;
                    query_map.put(vocab.get(parseWord), tmp);
                } else {
                    query_map.put(vocab.get(parseWord), 1.0);
                }
            }
        }
    }

    public static void createVocabluary() {
        BufferedReader bf;
        try {
            bf = new BufferedReader(new FileReader("/home/zaraza/vocabulary/vocabulary"));
            String line = bf.readLine();

            while (line != null) {
                StringTokenizer itr = new StringTokenizer(line);
                String key = itr.nextToken();
                Integer wordId = Integer.parseInt(itr.nextToken());
                double wordIdf = Double.parseDouble(itr.nextToken());

                vocab.putIfAbsent(key, wordId);
                idf.putIfAbsent(wordId, wordIdf);

                line = bf.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }


    public static class TokenizerMapper
            extends Mapper<Object, Text, IntWritable, DoubleWritable> {

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {

            StringTokenizer itr = new StringTokenizer(value.toString());

            Integer doc_id = Integer.parseInt(itr.nextToken());
            Integer word_id = Integer.parseInt(itr.nextToken());
            Double frequency = Double.parseDouble(itr.nextToken());

            if (query_map.containsKey(word_id)) {
                double first = frequency / idf.get(word_id);
                double second = query_map.get(word_id) / idf.get(word_id);
                context.write(new IntWritable(doc_id), new DoubleWritable(first * second));
            }
        }
    }

    public static class IntSumReducer
            extends Reducer<IntWritable, DoubleWritable, Text, DoubleWritable> {

        DoubleWritable result = new DoubleWritable();
        Text word = new Text();

        public void reduce(IntWritable key, Iterable<DoubleWritable> value, Context context)
                throws IOException, InterruptedException {

            double sum = 0;

            for (DoubleWritable val : value) {
                sum += val.get();
            }

            result.set(sum);
            word.set("DocID:" + key.get());
            context.write(word, result);

        }
    }

    public static class Combiner
            extends Reducer<IntWritable, DoubleWritable, IntWritable, DoubleWritable> {
        DoubleWritable result = new DoubleWritable();

        public void reduce(IntWritable key, Iterable<DoubleWritable> value, Context context)
                throws IOException, InterruptedException {
            double sum = 0;
            for (DoubleWritable val : value) {
                sum += val.get();
            }
            result.set(sum);
            context.write(key, result);

        }

    }

    public static void main(String[] args) throws Exception {
        createVocabluary();
        translateRequest();

//        for (int key : query_map.keySet()) {
//            System.out.println(key);
//            System.out.println(query_map.get(key));
//        }

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Query");
        job.setJarByClass(OldQuery.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(Combiner.class);
        job.setReducerClass(IntSumReducer.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(DoubleWritable.class);
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(DoubleWritable.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);

    }
}
