package ch.ethz.ir.retrievalsystem.processing

import ch.ethz.dal.tinyir.processing.XMLDocument
import ch.ethz.dal.tinyir.processing.StopWords
import com.github.aztek.porterstemmer.PorterStemmer
import scala.io.Source
import ch.ethz.dal.tinyir.io.TipsterStream
import scala.collection.parallel.ParSeq
import scala.util.Random
import scala.collection.mutable.ListBuffer
import breeze.linalg._
import java.util.Calendar
import scala.collection.parallel.ParSet
import scala.collection.parallel.ParMap
import java.io.PrintWriter
import scala.collection.mutable.ListBuffer

import ch.ethz.dal.tinyir.io.TipsterStream

import com.madhukaraphatak.sizeof.SizeEstimator

object Utils {
    
  //strips irrelevant characters
  def preprocess(text: String): List[String] = {
    val textFiltered = text.replaceAll("[^A-Za-z0-9]+", "").toLowerCase()
    (textFiltered.split("[a-z]") ++ textFiltered.split("[0-9]")).filter(_ != "").toList
  }
  
  //extracts tokens from a document
  def docToTokens(tokens: List[String], vocabulary: ParSet[String]): List[String] = {
    tokens.flatMap(preprocess(_))
          .flatMap(token => SpellChecker.spellCheck(token).split(" "))
          .map(PorterStemmer.stem(_))
          .filter(vocabulary.contains(_))
  }
  
  //extracts tokens from all documents
  def docsToTokens(xmlList: List[XMLDocument], vocabulary: ParSet[String]): ParSeq[(String,List[String])] = {
    xmlList.par.map(xmlDoc => xmlDoc.name -> xmlDoc.tokens)
               .map{case (docName,tokens) => (docName,docToTokens(tokens,vocabulary))}
  }
  
  //generates and prunes the vocabulary
  def generateVocabulary(xmlIter: Iterator[XMLDocument], vocabularyThreshold: Int, characterThreshold: Int): ParSet[String] = {
    var vocabularyUnpruned = ParMap[String,Int]()
    
    //generate vocabulary in batches so that intermediate results can fit into memory
    for (i <- 1 to 10) {
      val xmlListBuffer = ListBuffer[XMLDocument]()
      for (j <- 1 to 10000) {
        xmlListBuffer += xmlIter.next()
      }
      
      val vocabularyUnprunedIter = xmlListBuffer.toList
                                            .par
                                            .flatMap(_.tokens
                                                      .flatMap(preprocess)
                                                      .flatMap(token => SpellChecker.spellCheck(token).split(" ").filter(_ != ""))
                                                      .filter(token => ((token.size <= characterThreshold) && !StopWords.stopWords(token) && (token != "")))
                                                      .map(PorterStemmer.stem(_)))
                                            .groupBy(identity)
                                            .mapValues(_.length)
                                          
      vocabularyUnpruned = mergeVocabularies(vocabularyUnpruned,vocabularyUnprunedIter)
      println(Calendar.getInstance().getTime() + ": generated vocabulary (" + i + "/10), vocabulary size = " + vocabularyUnpruned.size)
    }
    vocabularyUnpruned.filter{case (word,count) => count > vocabularyThreshold}
                      .keys
                      .toSet
  }
  
  //used during the generation of vocabulary to merge the results of two batches
  def mergeVocabularies(map1: ParMap[String,Int], map2: ParMap[String,Int]): ParMap[String,Int] = {
    map1 ++ map2.map{ case (k,v) => k -> (map1.getOrElse(k,0) + v) }
  }
 
  //generates index and a map from the names of the documents to a 4-byte hash
  def generateIndexAndHashToDoc(tipsterIter: Iterator[XMLDocument],
                                vocabulary: ParSet[String],
                                batchIterations: Int,
                                batchSize: Int): (ParMap[Int,Map[Int,Short]],ParMap[Int,List[(Int,Short)]],Map[Int,String]) = {
        
    var index = ParMap[Int,Map[Int,Short]]()
    var hashToDoc = Map[Int, String]()
    for (i <- 1 to batchIterations) {
      
      val batchListBuffer = ListBuffer[XMLDocument]()
        for (j <- 1 to batchSize) {
          batchListBuffer += tipsterIter.next()
        }
      
      hashToDoc ++= batchListBuffer.map(xmlDoc => xmlDoc.name.hashCode() -> xmlDoc.name).toMap
      
      val indexIter = Utils.docsToTokens(batchListBuffer.toList, vocabulary)
                           .map{case (docName, tokens) => docName.hashCode() -> tokens.groupBy(identity).map{case (token,docNameList) => (token.hashCode(),docNameList.length.toShort)}}.toMap
      index ++= indexIter
      
      println(Calendar.getInstance().getTime() + ": created index (" + i + "/" + batchIterations + ")")
      
    }
    
    val invertedIndex = index.flatMap{case (docHash, tokenTfList) => tokenTfList.map{case (token, tokenTf) => (token, docHash, tokenTf)}}
                    .groupBy(_._1)
                    .mapValues(_.map(triple => (triple._2, triple._3)).seq.toList)
    
    (index,invertedIndex,hashToDoc)
  }
  
  //writes the ranking results for the test set into a file
  def writeResults(results: Map[Int, List[Int]], resultsFile: String, hashToDoc: Map[Int, String]) = {
    val writer = new PrintWriter(resultsFile)
    for ((queryID, retrievedDocs) <- results.toList.sortBy(_._1)) {
      var counter = 1
      for (docHash <- retrievedDocs) {
        writer.write("%d %d %s\n".format(queryID, counter, hashToDoc(docHash)))
        counter += 1
      }
    }
    writer.close()
  }
  
  def writeTimes(resultTimes: Map[Int, Long], timesFile: String) = {
    val writer = new PrintWriter(timesFile)
    for ((queryID, queryTime) <- resultTimes.toList.sortBy(_._1)) {
        writer.write("%d %d\n".format(queryID, queryTime))
    }
    writer.close()
  }
    
}