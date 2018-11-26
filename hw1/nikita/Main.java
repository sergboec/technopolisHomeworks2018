package nikita;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {

        SparkConf conf = new SparkConf().setAppName("MyApp").setMaster("local");
        JavaSparkContext sc = new JavaSparkContext(conf);
        sc.setLogLevel("OFF");

        JavaRDD<String> excluded = sc.textFile("./hw1/excluded.txt");
        JavaRDD<String> logins = sc.textFile("./hw1/logins.txt").subtract(excluded);

        JavaPairRDD<String, Long> counts = logins
                .flatMap(s -> Arrays.asList(s.split("\n")).iterator())
                .mapToPair(word -> new Tuple2<>(word, 1L))
                .reduceByKey((a, b) -> a + b);

        //1. top N
        System.out.println("Top of ua:");
        long half = logins.count() / 2;
        counts
                .mapToPair(p -> p.swap())
                .sortByKey(false)
                .mapToPair(p -> p.swap())
                .fold(new Tuple2<>("ua", 0L), (acc, ua) -> {
                    if (acc._2 < half) {
                        acc = new Tuple2<>("ua", acc._2 + ua._2);
                        System.out.println(ua._2 + " " + ua._1);
                    }
                    return acc;
                });

        //2. unique
        long unique = counts.count();
        System.out.println("unique = " + unique);

        //3. total logins
        long total = logins.count();
        System.out.println("total = " + total);

        //4. average
        double average = (double) total / (double) unique;
        System.out.println("average = " + average);

    }
}
