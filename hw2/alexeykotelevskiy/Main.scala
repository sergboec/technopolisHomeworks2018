package alexeykotelevskiy

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ArrayBuffer

object Main{
  def main(args:Array[String]): Unit = {
    val file = "D:\\dataset_simple.csv"
    val spark = SparkSession
      .builder()
      .appName("hw-2")
      .master("local")
      .getOrCreate()
    val df = spark
      .read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv(file)
      .toDF()
    val featuresCols = new ArrayBuffer[String]
    df
      .schema
      .fields
      .foreach(f => if (f.name != "label") {featuresCols += f.name})
    val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2))
    val assembler = new VectorAssembler()
      .setInputCols(featuresCols.toArray)
      .setOutputCol("features")


    val lr = new LogisticRegression().setMaxIter(10)
    val pipeline = new Pipeline().setStages(Array(assembler, lr))
    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.05, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
       //80% of the data will be used for training and the remaining 20% for validation.
      .setTrainRatio(0.8)

    val model = trainValidationSplit.fit(trainingData).bestModel

    val predictions = model.transform(testData)

    val evaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")

    println("Area under ROC = " +  evaluator.evaluate(predictions))
  }
}