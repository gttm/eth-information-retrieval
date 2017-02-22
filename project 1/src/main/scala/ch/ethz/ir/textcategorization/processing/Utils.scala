package ch.ethz.ir.textcategorization.processing

import ch.ethz.dal.tinyir.processing.XMLDocument

import ch.ethz.dal.tinyir.processing.StopWords
import com.github.aztek.porterstemmer.PorterStemmer
import scala.io.Source

import ch.ethz.dal.tinyir.io.ReutersRCVStream

import scala.collection.parallel.immutable.ParSet
import scala.collection.parallel.ParMap

import scala.util.Random

import scala.collection.mutable.ListBuffer

import breeze.linalg._

import java.util.Calendar

import ch.ethz.ir.textcategorization.train.Classifier

import scala.collection.parallel.immutable.ParSet

import scala.collection.mutable

import java.io.PrintWriter

object Utils {
  
  def log2(x: Float) = { (Math.log(x) / Math.log(2)).toFloat }
  
  //strips irrelevant characters
  def preprocess(text: String): String = {
    text.replaceAll("[-+1234567890()\\/]+", "").toLowerCase()
  }
  
  //extracts tokens from a document
  def docToTokens(tokens: List[String], vocabulary: ParSet[String]) = {
    tokens.map(token => PorterStemmer.stem(preprocess(token))).filter(vocabulary(_))
  }
  
  //extracts tokens from all documents
  def docsToTokens(xmlList: List[XMLDocument], vocabulary: ParSet[String]): ParMap[Int,List[String]] = {
    xmlList.par.map(xmlDoc => xmlDoc.ID -> xmlDoc.tokens).toMap
               .mapValues(docToTokens(_,vocabulary))           
  }
  
  //generates and prunes the vocabulary
  def generateVocabulary(xmlList: List[XMLDocument], vocabularyThreshold: Int) = {
    xmlList.par.flatMap(_.tokens)
               .map(Utils.preprocess(_))
               .filter(!StopWords.stopWords(_))
               .map(PorterStemmer.stem(_))
               .filter(_ != "")
               .groupBy(identity)
               .mapValues(_.length)
               .filter{case (word,count) => count > vocabularyThreshold}
               .keys
               .toSet
  }
  
  //shuffles the dataset for SGD based training
  def shuffleDataset(docIds: Set[Int],
                     numEpochs: Int): Array[Int] = {
    
    val random = new Random(13)
    
    val docIdsShuffled: ListBuffer[Int] = ListBuffer[Int]()
    
    for (i <- 1 to numEpochs) { docIdsShuffled ++= random.shuffle(docIds)}
    
    docIdsShuffled.toArray
  }
  
  //loads category codes
  def getCodes(dir: String) = {
    val codes: scala.collection.mutable.Set[String] = scala.collection.mutable.Set[String]()
    
    for (line <- Source.fromFile(dir + "/industry_codes.txt").getLines().drop(2)) {
      val code = line.split("\t").take(1)(0)
      if ((code != " ") && (code != "")) codes += code
    }
    for (line <- Source.fromFile(dir + "/region_codes.txt").getLines().drop(2)) { 
      val code = line.split("\t").take(1)(0)
      if ((code != " ") && (code != "")) codes += code
    }
    for (line <- Source.fromFile(dir + "/topic_codes.txt").getLines().drop(2)) { 
      val code = line.split("\t").take(1)(0)
      if ((code != " ") && (code != "")) codes += code
    }
    
    codes
  }
  
  //evaluates on the validation set
  def evaluate(xmlList: List[XMLDocument],
               classifier: Classifier,
               codes: Set[String]) = {
    var tp = 0f
    var fp = 0f
    var tn = 0f
    var fn = 0f
    
    var pSum = 0f
    var rSum = 0f
    var f1Sum = 0f
        
    println(Calendar.getInstance().getTime() + ": classifying documents...")
    for (doc <- xmlList) {
      tp = 0f
      fp = 0f
      tn = 0f
      fn = 0f

      val docResult = classifier.classifyDoc(doc)
      
      for (code <- codes) {
        val docCodeResult = docResult.getOrElse(code,false)
        
        if ((doc.codes contains code) && (docCodeResult)){
          tp += 1f
        }
        else if ((doc.codes contains code) && (!docCodeResult)){
          fn += 1f
        }
        else if (!(doc.codes contains code) && (docCodeResult)){
          fp += 1f
        }
        else if (!(doc.codes contains code) && (!docCodeResult)){
          tn += 1f
        }
      }
      
      val p = if (tp + fp == 0f) 0f else tp / (tp + fp)
      val r = if (tp + fn == 0f) 0f else tp / (tp + fn)
      val f1 = if ((p == 0f) || (r == 0f)) 0f else (2f*p*r / (p+r))
      
      pSum += p
      rSum += r
      f1Sum += f1
      
    }
    
    val pAvg = pSum / xmlList.length.toFloat
    val rAvg = rSum / xmlList.length.toFloat
    val f1Avg = f1Sum / xmlList.length.toFloat
    
    (pAvg,rAvg,f1Avg)
  }
  
  //predicts labels for of the test set and generates output file
  def predict(xmlList: List[XMLDocument],
              classifier: Classifier,
              predFile: String) {
    println(Calendar.getInstance().getTime() + ": classifying documents...")
    val writer = new PrintWriter(predFile)
    for (doc <- xmlList) {
      val docResult = classifier.classifyDoc(doc).filter{case (code,pred) => pred}.keys
      writer.write(doc.ID + " " + docResult.mkString(" " ) + "\n")
    }
    writer.close()
  }
  
}