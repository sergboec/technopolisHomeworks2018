import java.util

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.regression.{GBTRegressor, LinearRegression, LinearRegressionModel}
import org.apache.spark.ml.tuning.{CrossValidator, CrossValidatorModel, ParamGridBuilder}
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

    val Array(trainingData, testData) = df.randomSplit(Array(0.8, 0.2))

    val labelCol = "label"
    val featuresCol = "features"

    val types = new ListBuffer[String]

    df
      .schema
      .fields
      .foreach(f => if (f.name != labelCol) types.+=(f.name))

    val assembler = new VectorAssembler()
      .setInputCols(types.toArray)
      .setOutputCol("features")

    val lr = new LogisticRegression()
      .setLabelCol(labelCol)
      .setFeaturesCol(featuresCol)
      .setPredictionCol("Predicted " + labelCol)
      .setMaxIter(10)

    val stages = Array(
      assembler,
      lr
    )

    val pipeline = new Pipeline().setStages(stages)

    val paramGrid = new ParamGridBuilder().addGrid(lr.regParam, Array(0.1, 0.05, 0.01)).addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0)).build()

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(2)

    val pipelineModel = cv.fit(trainingData).bestModel

    val model = pipelineModel.asInstanceOf[PipelineModel]

//    model.write.overwrite().save("/home/gorbatov/LogisticRegressionModel")
//    val model = PipelineModel.load("/home/gorbatov/LogisticRegressionModel")

    val predictions = model.transform(testData)

    val evaluator = new BinaryClassificationEvaluator()
      .setLabelCol(labelCol)
      .setRawPredictionCol("Predicted " + labelCol)
      .setMetricName("areaUnderROC")

    val precision = evaluator.evaluate(predictions)

    println("Is larger better: " + evaluator.isLargerBetter)
    println("Precision: " + precision)

    println("\n" + model.stages(1).explainParams() + "\n")

    spark.stop()
  }
}
