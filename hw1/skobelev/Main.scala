import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

object Main{
  def main(args:Array[String]): Unit = {

    val conf:SparkConf = new SparkConf().setAppName("hw1").setMaster("local")
    val sc:SparkContext = new SparkContext(conf)

    val excludes:RDD[String] = sc.textFile("hw1/excluded.txt")
    val logins:RDD[String] = sc.textFile("hw1/logins.txt")
    val workingSel = logins.subtract(excludes)

    val counts = workingSel.map(obj => (obj, 1)).reduceByKey((a, b) => a + b)
    val half = counts.count() / 2
    val top = counts.map(obj => obj.swap).filter(obj => obj._1 > half)

    println("Top N in frequency ua which make up 50% of all logins >> ")
    top.foreach(obj => println(obj._1 + " " + obj._2))

    println("The number of unique ua >> " + counts.count())

    println("Total logins >> " + workingSel.count())

    println("The number of average logins >> " +  workingSel.count() / counts.count())
  }
}