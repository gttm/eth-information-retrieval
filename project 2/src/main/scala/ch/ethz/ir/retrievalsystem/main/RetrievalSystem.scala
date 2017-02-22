package ch.ethz.ir.retrievalsystem.main

import ch.ethz.dal.tinyir.io.TipsterStream
import scala.collection.mutable
import java.util.Calendar
import ch.ethz.ir.retrievalsystem.processing.Utils
import ch.ethz.ir.retrievalsystem.processing.Evaluation
import ch.ethz.ir.retrievalsystem.models.LanguageModel
import ch.ethz.ir.retrievalsystem.models.TermBasedModel
import ch.ethz.ir.retrievalsystem.models.PLSA
import scala.collection.parallel.ParSet
import scala.collection.parallel.ParMap
import scala.collection.mutable.ListBuffer
import com.madhukaraphatak.sizeof.SizeEstimator
import ch.ethz.dal.tinyir.processing.XMLDocument

import java.io._

object RetrievalSystem {
  
  def main(args: Array[String]) = {
    val model = args(0) //term or language
    val useIndex = args(1).toBoolean //true or false
    val vocabularyThreshold = args(2).toInt //vocabulary pruning threshold
    val characterThreshold = args(3).toInt //character pruning threshold
    val numTopics = args(4).toInt //number of topics for the language model
    val numIter = args(5).toInt //number of iterations for csizar-tusnady algorithm
    val mu = args(6).toFloat //mu value for bayes smoothing in the term based model
    val lambda = args(7).toFloat //lambda value for mercer smoothing in the term based model
    val smoothing = args(8) //bayes or mercer
    
    val batchSize = 1000
    val batchIterations = 100
    
    println(Calendar.getInstance().getTime() + ": loading dataset...")
    def tipsterIter = new TipsterStream("data").stream.iterator
    val pass1 = tipsterIter
    
    println(Calendar.getInstance().getTime() + ": building vocabulary...")
    val vocabulary = Utils.generateVocabulary(pass1,vocabularyThreshold,characterThreshold)
    
    val pass2 = tipsterIter
    
    println(Calendar.getInstance().getTime() + ": creating index and hashToDoc...")
    val (index,invertedIndex,hashToDoc) = Utils.generateIndexAndHashToDoc(tipsterIter,vocabulary,batchIterations,batchSize)
    
    println(Calendar.getInstance().getTime() + " : vocabulary size: " + vocabulary.size)
    println(Calendar.getInstance().getTime() + " : index size: " + index.size)
    println(Calendar.getInstance().getTime() + " : invertedIndex size: " + invertedIndex.size)
    println(Calendar.getInstance().getTime() + " : hashToDoc size: " + hashToDoc.size)
    
    if (model == "language") {
      println(Calendar.getInstance().getTime() + ": training language model...")
      val languageModel = new LanguageModel(vocabulary, numTopics, invertedIndex, index, hashToDoc, numIter)
      languageModel.train()
      
      println(Calendar.getInstance().getTime() + ": evaluating language model...") 
      val evaluation = new Evaluation("data/questions-descriptions.txt", "data/test-questions.txt", "data/relevance-judgements.csv", vocabulary, hashToDoc) 
      evaluation.evaluate(languageModel,useIndex)
      
      println(Calendar.getInstance().getTime() + ": producing test set predictions with languae model...")      
      evaluation.predict(languageModel,"l")
    }
    else if (model == "term") {
      println(Calendar.getInstance().getTime() + ": training term-based model...")
      val termModel = new TermBasedModel(vocabulary, invertedIndex, index, mu, lambda, smoothing)
      termModel.train()
                               
      println(Calendar.getInstance().getTime() + ": evaluating term-based model...")
      val evaluation = new Evaluation("data/questions-descriptions.txt", "data/test-questions.txt", "data/relevance-judgements.csv", vocabulary, hashToDoc) 
      evaluation.evaluate(termModel,useIndex)
      
      println(Calendar.getInstance().getTime() + ": producing test set predictions with term-based model...")
      evaluation.predict(termModel,"t")
     
    }
    else {
      throw new IllegalArgumentException("Unknown scoring model")
    }
        
  }
  
}