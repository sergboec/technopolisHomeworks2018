import java.util

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._

import scala.collection.mutable.ListBuffer

object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("ht-2")
      .master("local")
      .getOrCreate()

    val df = spark
      .read
      .option("header", true)
      .option("inferSchema", "true")
      .csv("/dataset_simple.csv")
      .toDF()

    df.show()

    val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2))

    val labelCol = "label"

    val types = new ListBuffer[String]

    df.schema.fields.foreach(f => if (f.name != labelCol) types.+=(f.name))

    val assembler = new VectorAssembler()
      .setInputCols(types.toArray)
      .setOutputCol("features")

    val gbt = new GBTRegressor()
      .setLabelCol(labelCol)
      .setFeaturesCol("features")
      .setPredictionCol("Predicted " + labelCol)
      .setMaxIter(150)

    val stages = Array(
      assembler,
      gbt
    )

    val pipeline = new Pipeline().setStages(stages)

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    val evaluator = new RegressionEvaluator()
      .setLabelCol(labelCol)
      .setPredictionCol("Predicted " + labelCol)
      .setMetricName("rmse")

    val error = evaluator.evaluate(predictions)

    println(error)

    spark.stop()
  }
}
