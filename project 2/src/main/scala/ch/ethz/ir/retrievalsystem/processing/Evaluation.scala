package ch.ethz.ir.retrievalsystem.processing

import scala.io.Source
import scala.collection.parallel.ParSet
import scala.collection.parallel.ParMap
import ch.ethz.ir.retrievalsystem.models.Model
import scala.math._
import ch.ethz.dal.tinyir.processing.Tokenizer

import ch.ethz.ir.retrievalsystem.models

import java.util.Calendar

class Evaluation(queriesFilePath: String, 
                 queriesTestFilePath: String, 
                 qrelsFilePath:String, 
                 vocabulary: ParSet[String], 
                 hashToDoc: Map[Int, String]) {
  
  val queryIDs: List[Int] = Source.fromFile(queriesFilePath).getLines.toList
              .filter(_.startsWith("<num>"))
              .map(_.split("\\s+").last.toInt)
  val queryTokens: List[List[String]] = Source.fromFile(queriesFilePath).getLines.toList
              .filter(_.startsWith("<title>"))
              .map(Tokenizer.tokenize(_).drop(2))
              .map(tokenList => Utils.docToTokens(tokenList, vocabulary))              
  val queries: Map[Int, List[String]] = queryIDs.zip(queryTokens).toMap
  
  val queryTestIDs: List[Int] = Source.fromFile(queriesTestFilePath).getLines.toList
              .filter(_.startsWith("<num>"))
              .map(_.split("\\s+").last.toInt)
  val queryTestTokens: List[List[String]] = Source.fromFile(queriesTestFilePath).getLines.toList
              .filter(_.startsWith("<title>"))
              .map(Tokenizer.tokenize(_).drop(2))
              .map(tokenList => Utils.docToTokens(tokenList, vocabulary))              
  val queriesTest: Map[Int, List[String]] = queryTestIDs.zip(queryTestTokens).toMap
  
  val qrels: Map[Int, List[Int]] = Source.fromFile(qrelsFilePath).getLines.toList
              .map(_.split(" ").toList)
              .filter(_(3) == "1")
              .map(lineList => (lineList(0).toInt, lineList(2).replaceAll("-", "").hashCode()))
              .groupBy(_._1)
              .mapValues(_.map(_._2))
              .toMap

  println(Calendar.getInstance.getTime() + ": Queries: " + queries)
  val readableQrels = qrels.mapValues(_.map(hashToDoc.getOrElse(_,"NotInTraining")))
  println(Calendar.getInstance.getTime() + ": Qrels: " + readableQrels)

  def printMetrics(results: Map[Int, List[Int]]) = {
    var pSum = 0f
    var rSum = 0f
    var f1Sum = 0f
    var apSum = 0f
        
    for ((queryID, retrievedDocs) <- results.seq) {
      val queryIDqrels = qrels(queryID)  
      val tpSize = retrievedDocs.filter(queryIDqrels.contains(_)).size
      val p = if (retrievedDocs.size == 0f) 0f else 1f*tpSize/retrievedDocs.size
      val r = if (queryIDqrels.size == 0f || retrievedDocs.size == 0f) 0f else 1f*tpSize/min(queryIDqrels.size, retrievedDocs.size) 
      val f1 = if ((p == 0f)||(r == 0f)) 0f else (2f*p*r/(p + r))
      pSum += p
      rSum += r
      f1Sum += f1
      
      var tp = 0f
      var ap = 0f
      var pos = 1f
      for (docHash <- retrievedDocs) {
        if (queryIDqrels.contains(docHash)) {
          tp += 1f
          ap += tp/pos
        }
        pos += 1
      }
      ap = if (retrievedDocs.size == 0f) 0f else ap/min(queryIDqrels.size, retrievedDocs.size) // bounded AP
      apSum += ap
      
      println("Query %d TP=%d, retrieved=%d, relevant=%d".format(queryID, tpSize, retrievedDocs.size, queryIDqrels.size))
      println("Query %d scores: P=%f, R=%f, F1=%f, AP=%f".format(queryID, p, r, f1, ap))
      
      println("Query ID: " + queryID + " Query: " + queries.getOrElse(queryID,List()))
      println("Retrieved Docs: " + retrievedDocs.map(hashToDoc(_)).sorted)
      println("Relevant documents: " + queryIDqrels.map(hashToDoc.getOrElse(_,"not in training")).sorted)
      println("---------------------------------------------------------------------------------------------------")
    }
    
    val map = apSum/results.size // bounded MAP
    println("---------------------------------------------------------------------------------------------------")
    println("Averages: P=%f, R=%f, F1=%f".format(pSum/results.size, rSum/results.size, f1Sum/results.size))
    println("MAP=" + map)
  }
  
  def evaluate(model: Model, useIndex: Boolean) = {
    
    //evaluates the model on the training set
    val resultTuples = {
      if (useIndex == true) {
        println(Calendar.getInstance().getTime() + ": perform queries with index...")
        queries.mapValues(model.performQuery(_)).map(identity)
      }
      else {
        println(Calendar.getInstance().getTime() + ": perform queries without index...")
        queries.mapValues(model.performQueryNoIndex(_)).map(identity)
      }
    }
    
    val results = resultTuples.mapValues(_._1).map(identity)
    val resultTimes = resultTuples.mapValues(_._2).map(identity)
    val readableResults = results.mapValues(_.map(hashToDoc(_)))
    println("Query results: " + readableResults)
    println("Query times in nsec: " + resultTimes)
    printMetrics(results)
    println("Average query time: " + resultTimes.values.sum/resultTimes.size)
  }
  
  //produces the predictions for the test queries
  def predict(model: Model, modelID: String){
    val resultTestTuples = queriesTest.mapValues(model.performQuery(_)).map(identity)
    val resultsTest = resultTestTuples.mapValues(_._1).map(identity)
    Utils.writeResults(resultsTest, "ranking-" + modelID + "-18.run", hashToDoc)
  }
  
}
