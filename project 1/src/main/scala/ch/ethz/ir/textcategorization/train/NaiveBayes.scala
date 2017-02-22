package ch.ethz.ir.textcategorization.train

import scala.util.Random

import ch.ethz.dal.tinyir.processing.XMLDocument
import ch.ethz.dal.tinyir.io.ReutersRCVStream
import ch.ethz.ir.textcategorization.processing.Utils
import com.github.aztek.porterstemmer.PorterStemmer
import ch.ethz.dal.tinyir.processing.StopWords

import scala.collection.parallel.immutable.ParSet

import scala.collection.mutable

import java.util.Calendar

class NaiveBayes(xmlListTrain: List[XMLDocument],
                 codes: Set[String],
                 vocabulary: ParSet[String],
                 alpha: Double) extends Classifier {
  
  def tf(doc: List[String]) : Map[String, Int] = doc.groupBy(identity).mapValues(l => l.length)
  
  val (wordCondProbPos,wordCondProbNeg,catProb) = {
    val vocabSize = vocabulary.size
    
    //term frequences per document
    println(Calendar.getInstance().getTime() + ": calculating tf...")
    val docTfs = xmlListTrain.par.map(doc => doc.ID -> tf(doc.tokens.map(Utils.preprocess(_)).filter(!StopWords.stopWords(_)).map(PorterStemmer.stem(_)).filter(vocabulary(_)))).toMap
  
    //length frequences per document
    println(Calendar.getInstance().getTime() + ": calculating lenD...")
    val lenD = xmlListTrain.par.map(doc => doc.ID -> doc.tokens.map(Utils.preprocess(_)).filter(!StopWords.stopWords(_)).map(PorterStemmer.stem(_)).filter(vocabulary(_)).length).toMap
  
    //document Ids
    println(Calendar.getInstance().getTime() + ": calculating docIds...")
    val docIds = xmlListTrain.map(_.ID).toSet
    
    //positive documents per category
    println(Calendar.getInstance().getTime() + ": calculating posDocs...")
    val posDocs = codes.par.map(code => (code -> xmlListTrain.filter(_.codes(code)).map(_.ID).toSet)).toMap
      
    //negative documents per category, subsampled
    println(Calendar.getInstance().getTime() + ": calculating negDocs...")
    val negDocs = codes.par.map(code => (code -> Random.shuffle(docIds.filter(!posDocs.getOrElse(code,Set[Int]())(_))).take(posDocs.getOrElse(code,Set()).size * 15))).toMap
    
    //category probabilities
    println(Calendar.getInstance().getTime() + ": calculating catProb...")
    val catProb = codes.par.map(code => code -> (posDocs.getOrElse(code,Set()).size.toDouble / (posDocs.getOrElse(code,Set()).size.toDouble + negDocs.getOrElse(code,Set()).size.toDouble))).toMap
    
    //denominator of word conditional probabilities per category (positive documents)
    println(Calendar.getInstance().getTime() + ": calculating denominatorPos...")
    val denominatorPos = codes.par.map(code => code -> (posDocs.getOrElse(code,Set()).map(docID => lenD.getOrElse(docID,0)).sum + alpha * vocabSize)).toMap
    
    //denominator of word conditional probabilities per category (negative documents)
    println(Calendar.getInstance().getTime() + ": calculating denominatorNeg...")
    val denominatorNeg = codes.par.map(code => code -> (negDocs.getOrElse(code,Set()).map(docID => lenD.getOrElse(docID,0)).sum + alpha * vocabSize)).toMap
    
    //word conditional probabilities per category (positive documents)
    println(Calendar.getInstance().getTime() + ": calculating wordCondProbPos...")
    val wordCondProbPos = codes.par.map(code => code -> (posDocs.getOrElse(code,List()).map(docID => docTfs.getOrElse(docID,Map[String,Int]())))
                                                                                     .flatMap(_.toSeq)
                                                                                     .groupBy(_._1)
                                                                                     .mapValues(_.map(_._2))
                                                                                     .mapValues(tfVal => (tfVal.sum + alpha) / denominatorPos.getOrElse(code,1.0)))
                                 .flatMap{case (code,tfList) => for (tf <- tfList) yield ((code,tf._1) -> tf._2)}.toMap
    
    //word conditional probabilities per category (negative documents)
    println(Calendar.getInstance().getTime() + ": calculating wordCondProbNeg...")
    val wordCondProbNeg = codes.par.map(code => code -> (negDocs.getOrElse(code,List()).map(docID => docTfs.getOrElse(docID,Map[String,Int]())))
                                                                                     .flatMap(_.toSeq)
                                                                                     .groupBy(_._1)
                                                                                     .mapValues(_.map(_._2))
                                                                                     .mapValues(tfVal => (tfVal.sum + alpha) / denominatorNeg.getOrElse(code,1.0)))
                                 .flatMap{case (code,tfList) => for (tf <- tfList) yield ((code,tf._1) -> tf._2)}.toMap
  
    (wordCondProbPos,wordCondProbNeg,catProb)
  }
                                
  //classifies a document for all category codes
  def classifyDoc(doc: XMLDocument): mutable.Map[String,Boolean] = {
    val retMap = mutable.Map[String,Boolean]()
    val docToTokens = Utils.docToTokens(doc.tokens,vocabulary)
    for (code <- codes) {
      val catProbVal = catProb.getOrElse(code,0.0)
      
      val posProb = Math.log(catProbVal) + docToTokens.map(x => Math.log(wordCondProbPos.getOrElse((code,x),alpha))).sum
      val negProb = Math.log(1 - catProbVal) + docToTokens.map(x => Math.log(wordCondProbNeg.getOrElse((code,x),alpha))).sum
      if (posProb > negProb) { retMap(code) = true }
    }
    retMap
  }
  
}