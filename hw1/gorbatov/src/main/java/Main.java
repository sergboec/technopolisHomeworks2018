import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import scala.Tuple2;

import java.util.Arrays;

public class Main {

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("HW-1").setMaster("local");
        SparkContext context = new SparkContext(conf);

        JavaRDD<String> logins = context.textFile("hw1/logins.txt", 1).toJavaRDD();
        JavaRDD<String> excluded = context.textFile("hw1/excluded.txt", 1).toJavaRDD();

        JavaRDD subTractedMap = logins.
                flatMap(s -> Arrays.asList(s.split("\n")).iterator())
                .subtract(excluded.flatMap(s -> Arrays.asList(s.split("\n")).iterator()));

        long loginsAmount = subTractedMap.count();

        JavaPairRDD<String, Integer> pairs = subTractedMap
                .mapToPair(e -> new Tuple2<String, Integer>((String) e , 1))
                .reduceByKey((v1, v2) -> (int)v1 + (int)v2);

        long uniqueAmonut = pairs.count();

        JavaPairRDD sorted = pairs
                .mapToPair(e -> new Tuple2<Integer, String>(e._2, e._1))
                .sortByKey(false);

        final long half = loginsAmount / 2;
        long counter = 0;

        sorted.aggregate(counter, (c, e) -> {
            Tuple2<Integer, String> tuple = (Tuple2<Integer, String>) e;
            if (half > (long)c) System.out.println(((Tuple2<Integer, String>) e)._2);
            return ((long)c) + tuple._1;
        }, (a,b)-> ((long)a) + ((long)b));

        System.out.println("Unique logins: " + uniqueAmonut);
        System.out.println("Logins: " + loginsAmount);
        System.out.println("Average: " + loginsAmount / uniqueAmonut);

    }
}
