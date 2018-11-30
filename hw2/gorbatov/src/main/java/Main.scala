import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.GBTRegressor
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.types.{ArrayType, DoubleType, StructField, StructType}

object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession
      .builder()
      .appName("ht-2")
      .master("local")
      .getOrCreate()
    val schema = StructType(
      StructField("features", ArrayType(DoubleType))::
      StructField("label", DoubleType):: Nil
    );
    val df = spark.read.option("header", true).schema(schema).csv("/home/gorbatov/dataset_simple.csv").na.drop()
    val Array(trainingData: DataFrame, testData: DataFrame) = df.randomSplit(Array(0.8, 0.2))
    val labelCol = "label"
    val gbt = new GBTRegressor()
      .setLabelCol(labelCol)
      .setFeaturesCol("features")
      .setPredictionCol("Predicted " + labelCol)
      .setMaxIter(50)
    val stages = Array(gbt)
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