package ch.ethz.ir.textcategorization.processing

import ch.ethz.dal.tinyir.processing.XMLDocument

import com.github.aztek.porterstemmer.PorterStemmer
import ch.ethz.dal.tinyir.processing.StopWords

import scala.collection.parallel.immutable.ParSet
import scala.collection.parallel.ParMap
import scala.collection.parallel.mutable.ParArray

import breeze.linalg._
import breeze.stats.mean

//static indexing of words, so that a vector offeset corresponds to a specific word
object FeatureGenerator {
  var vocabularyIndex:ParMap[String,Int] = null
  def getVocabularyIndex(vocabulary: ParSet[String]) = {
    if (vocabularyIndex == null) { vocabularyIndex = vocabulary.zipWithIndex.toMap }
    vocabularyIndex
  }
}

class FeatureGenerator(xmlList: List[XMLDocument],
                       vocabulary: ParSet[String]) {
  
  val docIds = xmlList.map(_.ID).toSet
  
  val nDocuments = xmlList.length
  val vocabSize = vocabulary.size
  
  val vocabularyIndex = FeatureGenerator.getVocabularyIndex(vocabulary)
  val documentIndex = xmlList.map(_.ID).zipWithIndex.toMap
  
  def tf(doc: List[String]) : Map[String, Int] = doc.groupBy(identity).mapValues(l => l.length)
    
  def logtf(doc: List[String]): Map[String, Float] = {
    tf(doc).mapValues(v => Utils.log2(v + 1.0f))
  }
  
  //tfidf for each token per document
  val tfidf: ParMap[Int,Map[String,Float]] = {
    val docsToTokens = Utils.docsToTokens(xmlList,vocabulary)
    
    val df = new Array[Int](vocabSize)
    for ((docId,tokens) <- docsToTokens ; token <- tokens.distinct) 
      df(vocabularyIndex(token)) += 1
              
    val idf = df.map(Utils.log2(nDocuments) - Utils.log2(_))
    
    docsToTokens.par.mapValues(tokens => logtf(tokens).map { case (k,v) => (k,v * idf(vocabularyIndex(k))) })
  }
  
  //generation of a tfidf vector for every document
  def tfidfVec(docId: Int): DenseVector[Float] = {
    val docTfidf = tfidf(docId)
    val docTfidfVec = DenseVector.zeros[Float](vocabSize+1)
    
    docTfidf.par.foreach{case (token,tfidf) => docTfidfVec(vocabularyIndex(token)) = tfidf}
    docTfidfVec(vocabSize) = 1f
    
    docTfidfVec
  }
}