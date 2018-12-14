import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.sql.SparkSession

import scala.util.Random

object Main {
  def main(args: Array[String]): Unit = {
    // initialise spark context
    val spark: SparkSession = SparkSession.builder()
      .appName("Pipeline")
      .master("local[*]")
      .getOrCreate()

    val sc = spark.sparkContext

    val df = spark.read
      .format("csv")
      .option("header", "true") //reading the headers
      .option("inferSchema", "true")
      .option("mode", "DROPMALFORMED")
      .load("./hw2/kylchik/src/main/resources/dataset_simple.csv")

    val assembler = new VectorAssembler()
      .setInputCols(df
        .schema
        .fields
        .map(f => f.name)
        .slice(0, 1000))
      .setOutputCol("features")
    val lr = new LogisticRegression()
    val pipeline = new Pipeline()
      .setStages(Array(assembler, lr))

    val Array(training, test) = df.randomSplit(Array(0.9, 0.1), seed = new Random().nextInt())

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(0.1, 0.01))
      .addGrid(lr.fitIntercept)
      .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
      .addGrid(lr.maxIter, Array(1, 5, 10, 20))
      //.addGrid(lr.standardization)
      //.addGrid(lr.aggregationDepth, Array(2, 3, 5))
      //.addGrid(lr.family, Array("auto", "binomial", "multinomial"))
      //.addGrid(lr.threshold, Array(0.1, 0.5, 1))
      .build()

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(new BinaryClassificationEvaluator)
      .setEstimatorParamMaps(paramGrid)
      // 80% of the data will be used for training and the remaining 20% for validation.
      .setTrainRatio(0.8)

    // Run train validation split, and choose the best set of parameters.
    val model = trainValidationSplit.fit(training).bestModel

    val predictions = model.transform(test)
    val evaluator = new BinaryClassificationEvaluator()
      .setMetricName("areaUnderROC")
    println("Area under ROC = " + evaluator.evaluate(predictions))

    println("Model hyper-parameters: " + model.asInstanceOf[PipelineModel].stages(1).explainParams())

    sc.stop()
  }
}