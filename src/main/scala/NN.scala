package org.apache.spark.mllib

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.Row

/**
  * Created by Shilpika on 7/11/16.
  */

case class Config(iter: Option[Int] = Some(60000))

object NN {


  def parseCommandLine(args: Array[String]): Option[Config] = {
    val parser = new scopt.OptionParser[Config]("scopt") {
      head("LineCount", "1.0")
      opt[Int]('i', "iter") action { (x, c) =>
        c.copy(iter = Some(x))
      } text ("iteration is an Int property")
      help("help") text ("-i enter max iteration")

    }
    parser.parse(args, Config())
  }

  def main(args: Array[String]):Unit={

    val appConfig = parseCommandLine(args).getOrElse(Config())
    val iterM:Int = appConfig.iter.getOrElse(60000)

    val conf = new SparkConf().setAppName("Multilayer-Perceptron-Classifier")
    val sc: SparkContext = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    // Load training data
    val data = MLUtils.loadLibSVMFile(sc, "libsvm_batch2.txt").toDF()
    // Split the data into train and test
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1000L)
    val train = splits(0)
    val test = splits(1)
    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4 and output of size 3 (classes)
    val layers = Array[Int](3072, 80, 50, 10)
    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(2500)
      .setSeed(1000L)
      .setMaxIter(iterM)
    // train the model
    val model = trainer.fit(train)
    // compute precision on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val pl = (result.select("prediction"), result.select("label"))


    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("precision")
    println("Precision:" + evaluator.evaluate(predictionAndLabels))

    val evaluator1 = new MulticlassClassificationEvaluator()
      .setMetricName("recall")
    println("Recall:" + evaluator1.evaluate(predictionAndLabels))
  }

}
