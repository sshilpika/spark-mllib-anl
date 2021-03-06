package org.apache.spark.mllib

import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
/**
  * Created by Shilpika on 7/12/16.
  */
object NN1 {

  def main1(args: Array[String]): Unit = {


    val conf = new SparkConf().setAppName("Multilayer-Perceptron-Classifier-EXAMPLE")
    val sc: SparkContext = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._

    // Load the data stored in LIBSVM format as a DataFrame.
    val data = sqlContext.read.format("libsvm")
      .load("sample_multiclass_classification_data.txt")
    // Split the data into train and test
    val splits = data.randomSplit(Array(0.6, 0.4), seed = 1000L)
    val train = splits(0)
    val test = splits(1)
    // specify layers for the neural network:
    // input layer of size 4 (features), two intermediate of size 5 and 4
    // and output of size 3 (classes)
    val layers = Array[Int](4, 5, 4, 3)
    // create the trainer and set its parameters
    val trainer = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1000L)
      .setMaxIter(100)
    // train the model
    val model = trainer.fit(train)
    // compute precision on the test set
    val result = model.transform(test)
    val predictionAndLabels = result.select("prediction", "label")
    val evaluator = new MulticlassClassificationEvaluator()
      .setMetricName("precision")
    println("Precision:" + evaluator.evaluate(predictionAndLabels))

  }
}
