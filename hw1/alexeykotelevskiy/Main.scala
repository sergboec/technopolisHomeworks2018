package alexeykotelevskiy

import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object Main{
  def main(args:Array[String]): Unit = {
    val conf:SparkConf = new SparkConf().setAppName("Hw1").setMaster("local")
    val sc:SparkContext = new SparkContext(conf)
    val excludes:RDD[String] = sc.textFile("hw1/excluded.txt")
    val logins:RDD[String] = sc.textFile("hw1/logins.txt").subtract(excludes)
    val counts = logins.map(line => (line, 1)).reduceByKey((a, b) => a + b)
    val half = counts.count() / 2

    //1.Top N
    println("Top N:")
    counts.map(v => v.swap).filter(v => v._1 > half).sortByKey(false).foreach(x => println(x._2 + " " + x._1))

    //2.Unique
    println("Unique: " + counts.count())

    //3.All logins:
    println("All logins: " + logins.count())

    //4.Average:
    println("Average: " +  logins.count() / counts.count())
  }
}
