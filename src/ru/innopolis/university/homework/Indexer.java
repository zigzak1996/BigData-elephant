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

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.StringTokenizer;

public class Indexer {

    public static Map vocab = new HashMap<String, Integer>();

    public static final String PATH_TO_VOC = "/home/misha/IdeaProjects/WordCount/vocabulary/vocabulary";

    public static void createVocabulary() {
        BufferedReader bf;
        try {
            System.out.println("START");
            bf = new BufferedReader(new FileReader(PATH_TO_VOC));
            String line = bf.readLine();

            while (line != null) {
                StringTokenizer itr = new StringTokenizer(line);
                String key = itr.nextToken();
                Integer value = Integer.parseInt(itr.nextToken());
                vocab.putIfAbsent(key, value);

                line = bf.readLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

    }


    public static class TokenizerMapper
            extends Mapper<Object, Text, IntWritable, Text> {

        private Text word = new Text();

        public void map(Object key, Text value, Context context
        ) throws IOException, InterruptedException {
            JSONObject json = new JSONObject(value.toString());

            StringTokenizer itr = new StringTokenizer(json.getString("text"));

            while (itr.hasMoreTokens()) {

                String word = itr.nextToken().toLowerCase();

                StringBuilder parseWord = new StringBuilder();

                for (int i = 0; i < word.length(); i++) {

                    if ((word.charAt(i) >= 'a' && word.charAt(i) <= 'z') || word.charAt(i) == '-') {

                        parseWord.append(word.charAt(i));

                    }

                }
                if (vocab.containsKey(parseWord.toString())) {
                    int k = json.getInt("id"); //id doc-a;
                    int v = (int) vocab.get(parseWord.toString()); // id slova v doke;
                    this.word.set(Integer.toString(v));
                    context.write(new IntWritable(k), this.word);
                }

            }
        }
    }

    public static class IntSumReducer
            extends Reducer<IntWritable, Text, IntWritable, Text> {

        public void reduce(IntWritable key, Iterable<Text> values, Context context) {
            Map<String, Integer> count = new HashMap<>();
            System.out.print(key.get());
            for (Text word_id_in_doc : values) {

                count.merge(word_id_in_doc.toString(), 1, Integer::sum);

            }
            count.forEach((k, v) -> {
                Text text = new Text();
                text.set(String.format("%s\t%s", k, Integer.toString(v)));
                try {
                    context.write(key, text);
                } catch (IOException | InterruptedException e) {
                    e.printStackTrace();
                }
            });


        }
    }

    public static class Combiner extends Reducer<IntWritable, Text, IntWritable, Text> {
        public void reduce(IntWritable key, Iterable<Text> value, Context context) throws IOException, InterruptedException {
            for (Text v : value)
                context.write(key, v);
        }

    }

    public static void main(String[] args) throws Exception {
        createVocabulary();

        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf, "Indexer");
        job.setJarByClass(Indexer.class);
        job.setMapperClass(TokenizerMapper.class);
        job.setCombinerClass(Combiner.class);
        job.setReducerClass(IntSumReducer.class);
        job.setMapOutputKeyClass(IntWritable.class);
        job.setMapOutputValueClass(Text.class);
        job.setOutputKeyClass(IntWritable.class);
        job.setOutputValueClass(Text.class);
        FileInputFormat.addInputPath(job, new Path(args[0]));
        FileOutputFormat.setOutputPath(job, new Path(args[1]));
        System.exit(job.waitForCompletion(true) ? 0 : 1);
    }

}
