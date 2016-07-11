
package org.apache.spark.mllib

import org.apache.spark.{SparkConf, SparkContext}
// $example on$
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.mllib.tree.model.RandomForestModel
import org.apache.spark.mllib.util.MLUtils
// $example off$

object SparkRandomForestClassification {
  def main1(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("RandomForestClassification")
    val sc = new SparkContext(conf)
    // $example on$
    // Load and parse the data file.
    val data = MLUtils.loadLibSVMFile(sc, "/home/shilpika/anl/mllibSpark/code/batch_1_libsvm.txt")
    // Split the data into training and test sets (30% held out for testing)
    val splits = data.randomSplit(Array(0.7, 0.3))
    val (trainingData, testData) = (splits(0), splits(1))

    // Train a RandomForest model.
    // Empty categoricalFeaturesInfo indicates all features are continuous.
    val numClasses = 10
    val categoricalFeaturesInfo = Map[Int, Int]()
    val numTrees = 500 // Use more in practice.
    val featureSubsetStrategy = "auto" // Let the algorithm choose.
    val impurity = "gini"
    val maxDepth = 20
    val maxBins = 100

    val model = RandomForest.trainClassifier(trainingData, numClasses, categoricalFeaturesInfo,
      numTrees, featureSubsetStrategy, impurity, maxDepth, maxBins)

    // Evaluate model on test instances and compute test error
    val labelAndPreds = testData.map { point =>
      val prediction = model.predict(point.features)
      (point.label, prediction)
    }
    val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()
    println("Test Error = " + testErr)
    import java.io._
    val writer = new PrintWriter(new File("testresult.txt" ))

    writer.write(testErr+" "+numTrees)
    writer.close()

    //println("Learned classification forest model:\n" + model.toDebugString)

    // Save model
    model.save(sc, "myRandomForestClassificationModel1")
   // val sameModel = RandomForestModel.load(sc, "myRandomForestClassificationModel")
    // $example off$
  }
}