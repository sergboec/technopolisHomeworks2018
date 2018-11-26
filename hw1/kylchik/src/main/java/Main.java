import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function2;
import scala.Serializable;
import scala.Tuple2;

import java.util.Arrays;
import java.util.Comparator;
import java.util.concurrent.atomic.AtomicInteger;

public class Main {

    private static volatile AtomicInteger i = new AtomicInteger(0);

    interface SerializableComparator<T> extends Comparator<T>, Serializable {
        static <T> SerializableComparator<T> serialize(SerializableComparator<T> comparator) {
            return comparator;
        }
    }

    public static void main(String[] args) {
        SparkConf conf = new SparkConf().setAppName("HW1").setMaster("local");

        JavaSparkContext sc = new JavaSparkContext(conf);

        JavaRDD<String> logins = sc.textFile(System.getProperty("user.dir") + "/hw1/logins.txt");
        JavaRDD<String> excluded = sc.textFile(System.getProperty("user.dir") + "/hw1/excluded.txt");

        JavaRDD<String> loginWithoutExcluded =  logins.subtract(excluded);

        //1
        JavaPairRDD<String, Integer> counts = loginWithoutExcluded.flatMap(s -> Arrays.asList(s.split("\n")).iterator())
                .mapToPair(line -> new Tuple2<>(line, 1))
                .reduceByKey((a, b) -> a + b);

        long size = loginWithoutExcluded.count();
        counts.values().sortBy(f -> f, true, 1).aggregate(0,
                (Function2<Integer, Integer, Integer>) (v1, v2) -> {
                    if(v1 + v2 < size/2){
                        i.addAndGet(1);
                    }
                    return v1 + v2;
                },
                (v1, v2) -> null);

        counts.top((int) (counts.count() - i.get()), SerializableComparator.serialize((a, b) -> a._2.compareTo(b._2)))
                .forEach(t -> System.out.println("Item : " + t._1  + " Count : " + t._2));

        //2
        System.out.println("Количество уникальных UA = " + loginWithoutExcluded.distinct().count());
        //3
        System.out.println("Общее количество UA = " + loginWithoutExcluded.count());
        //4
        System.out.println("Среднее количество логинов = " + (double) loginWithoutExcluded.count() / loginWithoutExcluded.distinct().count());

        sc.close();
    }
}

