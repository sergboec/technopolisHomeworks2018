import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object Main {
  def main(args: Array[String]): Unit = {

    var sc: SparkContext = new SparkContext(new SparkConf().setAppName("technopolis").setMaster("local"))

    val logins: RDD[String] = sc.textFile("hw1/logins.txt").subtract(sc.textFile("hw1/excluded.txt"))

    val uniqueLogins = logins
      .map(key => (key, 1))
      .reduceByKey(_ + _)

    val halfOfReduceLogins = uniqueLogins.count() / 2

    println("Top logins in frequency which make uo 50% of all logins: ")
    uniqueLogins
      .filter(i => i._2.>(halfOfReduceLogins))
      .sortBy(k => k._2, false)
      .foreach(x => println(x))

    println("\nCount of unique logins: %d".format(uniqueLogins.count()))

    println("\nCount of logins: %d".format(logins.count()))

    println("\nAverage count of logins: %d".format(logins.count()./(uniqueLogins.count())))
  }
}