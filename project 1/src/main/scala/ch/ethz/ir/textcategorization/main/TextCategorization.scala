package ch.ethz.ir.textcategorization.main
 
import ch.ethz.dal.tinyir.io.ReutersRCVStream
import ch.ethz.dal.tinyir.processing.Tokenizer
import ch.ethz.dal.tinyir.processing.XMLDocument
import ch.ethz.dal.tinyir.processing.ReutersRCVParse

import ch.ethz.ir.textcategorization.processing._
import ch.ethz.ir.textcategorization.train._
import scala.collection.mutable._
import com.github.aztek.porterstemmer.PorterStemmer
import ch.ethz.dal.tinyir.processing.StopWords

import java.util.Calendar

import breeze.linalg._

object TextCategorization {
  
  def main(args: Array[String]){
    
    //execution parameters
    //irrelevant parameters are ignored, sorry for that ugliness :)
    val algorithm = args(0)
    val learningRate = args(1).toFloat
    val lambda = args(2).toFloat
    val codeThreshold = args(3).toInt
    val vocabularyThreshold = args(4).toInt
    val maxEpochs = args(5).toInt
    val maxNoImprovement = args(6).toInt
    val bayesAlpha = args(7).toFloat
    
    //dataset loading
    val streamTrain = new ReutersRCVStream("data/zips/train").stream
    val xmlListTrain = streamTrain.toList
    
    val streamValid = new ReutersRCVStream("data/zips/validation").stream
    val xmlListValid = streamValid.toList
    
    val streamPred = new ReutersRCVStream("data/zips/test").stream
    val xmlListPred = streamPred.toList
    
    //vocabulary generation
    val vocabulary = Utils.generateVocabulary(xmlListTrain,vocabularyThreshold)
    val vocabSize = vocabulary.size
    println(Calendar.getInstance().getTime() + ": vocabulary size = " + vocabSize)
    
    //code set generation
    val codesTrain = xmlListTrain.flatMap(_.codes)
                                .groupBy(identity)
                                .mapValues(_.length)
                                .filter{case (code,count) => count > codeThreshold}
                                .keys
                                .toSet
    val codesEval = xmlListTrain.flatMap(_.codes).toSet ++ xmlListValid.flatMap(_.codes).toSet                                
    println(Calendar.getInstance().getTime() + ": codes size = " + codesTrain.size)
    
    if (algorithm == "bayes") {
      //training of naive bayes model
      println(Calendar.getInstance().getTime() + ": training Naive Bayes...")
      val naiveBayes = new NaiveBayes(xmlListTrain, codesTrain, vocabulary, bayesAlpha)
                     
      //evaluation of naive bayes model on validation set
      println(Calendar.getInstance().getTime() + ": evaluating Naive Bayes...")
      val (p,r,f1) = Utils.evaluate(xmlListValid,naiveBayes,codesEval)
      println(Calendar.getInstance().getTime() + ": Terminated, final P/R/F1 = " + (p,r,f1))
         
      //prediction of labels for the test set
      println(Calendar.getInstance().getTime() + ": prediciting unseen labels with Naive Bayes...")
      Utils.predict(xmlListPred,naiveBayes,"ir-2016-1-project-18-nb.txt")
    }
                                
    else if (algorithm == "lr") {
      //feature generation for each dataset                         
      println(Calendar.getInstance().getTime() + ": calculating features for training set...")
      val featureGeneratorTrain = new FeatureGenerator(xmlListTrain,vocabulary)
                                  
      println(Calendar.getInstance().getTime() + ": calculating features for validation set...")
      val featureGeneratorValid = new FeatureGenerator(xmlListValid,vocabulary)
                                  
      println(Calendar.getInstance().getTime() + ": calculating features for test set...")
      val featureGeneratorPred = new FeatureGenerator(xmlListPred,vocabulary)
      
      //training of logistic regression model
      println(Calendar.getInstance().getTime() + ": training logistic regression...")
      val logisticRegression = new LogisticRegression(featureGeneratorTrain,
                                                      featureGeneratorValid,
                                                      featureGeneratorPred,
                                                      learningRate,
                                                      maxEpochs,
                                                      maxNoImprovement,
                                                      xmlListTrain,
                                                      xmlListValid,
                                                      codesTrain,
                                                      codesEval,
                                                      vocabulary)          
                          
      logisticRegression.train()
      
      //prediction of labels for the test set
      println(Calendar.getInstance().getTime() + ": prediciting unseen labels with Logistic Regression...")
      Utils.predict(xmlListPred,logisticRegression,"ir-2016-1-project-18-lr.txt")
    }
    else if (algorithm == "svm") {
      //feature generation for each dataset                         
      println(Calendar.getInstance().getTime() + ": calculating features for training set...")
      val featureGeneratorTrain = new FeatureGenerator(xmlListTrain,vocabulary)
                                  
      println(Calendar.getInstance().getTime() + ": calculating features for validation set...")
      val featureGeneratorValid = new FeatureGenerator(xmlListValid,vocabulary)
                                  
       println(Calendar.getInstance().getTime() + ": calculating features for test set...")
      val featureGeneratorPred = new FeatureGenerator(xmlListPred,vocabulary)
  
      //training of SVM model
      println(Calendar.getInstance().getTime() + ": training svm...")
      val svm = new SVM(featureGeneratorTrain,
                        featureGeneratorValid,
                        featureGeneratorPred,
                        lambda,
                        xmlListTrain,
                        xmlListValid,
                        codesTrain,
                        vocabulary)          
                          
      svm.train()
         
      //prediction of labels for the test set
      println(Calendar.getInstance().getTime() + ": prediciting unseen labels with SVM...")
      Utils.predict(xmlListPred,svm,"ir-2016-1-project-18-svm.txt")
    }
    else { println(Calendar.getInstance().getTime() + ": ERROR : unsupported classification algorithm ") }                         
                              
  }
  
}