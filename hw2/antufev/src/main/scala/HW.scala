import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ListBuffer

object HW {
  def main(args: Array[String]): Unit = {

    val spark = SparkSession
      .builder()
      .appName("technopolis")
      .master("local[4]")
      .getOrCreate()

    val labelColumn = "label"

    val featuresColumn = "features"

    val dataFrame = spark
      .read
      .option("header", true)
      .option("inferSchema", "true")
      .csv("hw2/dataset_simple.csv")
      .toDF()

    val Array(trainingData, testData) = dataFrame
      .randomSplit(Array(0.8, 0.2))

    val types = new ListBuffer[String]

    dataFrame
      .schema
      .fields
      .foreach(f => if (f.name != labelColumn) types += (f.name))

    val vectorAssembler = new VectorAssembler()
      .setInputCols(types.toArray)
      .setOutputCol(featuresColumn)

    val logisticRegression = new LogisticRegression()
      .setMaxIter(10)
      .setElasticNetParam(0.8)

    val paramGrid = new ParamGridBuilder()
      .addGrid(logisticRegression.regParam, Array(0.1, 0.05, 0.01))
      .addGrid(logisticRegression.fitIntercept)
      .addGrid(logisticRegression.elasticNetParam, Array(0.0, 0.5, 1.0))
      .build()

    val pipeline = new Pipeline().setStages(Array(vectorAssembler, logisticRegression))

    val crossValidator = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)

    val model = crossValidator.fit(trainingData).bestModel.asInstanceOf[PipelineModel]

    val evaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")

    println("AUC precision " + evaluator.evaluate(model.transform(testData)))
  }
}
