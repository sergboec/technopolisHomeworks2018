import java.util

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{GBTRegressor, LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._

import scala.collection.mutable.ListBuffer

object Main {

  def main(args: Array[String]): Unit = {
    val pathToDataset = "/dataset_simple.csv"
    val spark = SparkSession.builder().appName("hw-2")
      .master("local").getOrCreate()

    val df = spark.read.option("header", true)
      .option("inferSchema", "true")
      .csv(pathToDataset).toDF()

    val types = new ListBuffer[String]

    df.schema.fields
      .foreach(f => if (f.name != "label") types.+=(f.name))

    val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2))

    val assembler = new VectorAssembler().setInputCols(types.toArray).setOutputCol("features")

    val lr = new LogisticRegression().setMaxIter(10)

    var pipeline = new Pipeline().setStages(Array(assembler, lr))

    val paramGridBuilder = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.05, 0.01))
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    val evaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGridBuilder)
      .setTrainRatio(0.8)

    val pipelineModel = trainValidationSplit.fit(trainingData).bestModel

    val model = pipelineModel.asInstanceOf[PipelineModel]

    val predictions = model.transform(testData)

    val precision = evaluator.evaluate(predictions)

    println(precision)
  }
}
