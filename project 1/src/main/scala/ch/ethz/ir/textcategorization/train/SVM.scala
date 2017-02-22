package ch.ethz.ir.textcategorization.train

import ch.ethz.dal.tinyir.processing.XMLDocument
import ch.ethz.dal.tinyir.io.ReutersRCVStream

import ch.ethz.ir.textcategorization.processing._

import breeze.linalg._
import breeze.numerics._

import scala.collection.mutable
import scala.collection.parallel.immutable.ParSet

import java.util.Calendar

class SVM(featureGeneratorTrain: FeatureGenerator,
          featureGeneratorValid: FeatureGenerator,
          featureGeneratorPred: FeatureGenerator,
          lambda: Float,
          xmlListTrain: List[XMLDocument],
          xmlListValid: List[XMLDocument],
          codesTrain: Set[String],
          vocabulary: ParSet[String]) extends Classifier {

  val numCodes = codesTrain.size
  val numDocs = xmlListTrain.size
  
  val vecOnes = DenseVector.ones[Float](numCodes)

  //document IDs
  println(Calendar.getInstance().getTime() + ": calculating docIds...")
  val docIds = xmlListTrain.map(_.ID).toSet

  //indexing of codes, so that a vector offeset corresponds to a specific code
  println(Calendar.getInstance().getTime() + ": calculating codes index...")
  val codesIdx = codesTrain.zipWithIndex.toMap
  val codesIdxInv = new Array[String](numCodes)
  codesIdx.foreach { case (code, codeid) => codesIdxInv(codeid) = code }

  //positive documents per category
  println(Calendar.getInstance().getTime() + ": calculating posDocs...")
  val posDocs = codesTrain.par.map(code => (code -> xmlListTrain.filter(_.codes(code)).map(_.ID).toSet)).toMap

  //negative documents per category
  println(Calendar.getInstance().getTime() + ": calculating negDocs...")
  val negDocs = codesTrain.par.map(code => (code -> docIds.filter(!posDocs.getOrElse(code, Set[Int]())(_)))).toMap

  val vocabSize = vocabulary.size

  //svm weight vector
  var theta = DenseMatrix.zeros[Float](numCodes, vocabSize + 1)
  
  //+1 or -1 according to class
  val y = {
    val y = mutable.Map[Int, DenseVector[Float]]()
    for (docId <- xmlListTrain.map(_.ID)) {
      y(docId) = DenseVector.ones[Float](numCodes)
    }
    for ((code, docIds) <- negDocs; docId <- docIds) { y(docId)(codesIdx(code)) = -1f }
    y
  }
  
  //training of svm classifier
  def train() {
    val docIdsShuffled = Utils.shuffleDataset(docIds, 1)
    var iter = 0

    while (iter < numDocs) {
      val docId = docIdsShuffled(iter)

      val docY = y(docId)

      val docTfidfVec = featureGeneratorTrain.tfidfVec(docId)

      val thetaShrink = (1 - 1f / (iter + 1f)) * theta
      
      val margin = docY :* (theta * docTfidfVec)
      val marginSelect = DenseVector.zeros[Float](numCodes)
      marginSelect(margin :< 1f) := 1f
      
      val gradientPart = (1f / (lambda * (iter + 1f))) * ((marginSelect :* docY) * docTfidfVec.t)
      
      theta = thetaShrink + gradientPart

      if (iter % 1000 == 0) { println(Calendar.getInstance().getTime() + ": iter = " + iter) }

      iter += 1
    }
  }
  
  //classifies a document for each category code
  def classifyDoc(doc: XMLDocument) = {
    val docId = doc.ID
    val docTfidfVec = if (featureGeneratorValid.docIds.contains(docId)) featureGeneratorValid.tfidfVec(docId) else featureGeneratorPred.tfidfVec(docId)
    val docProbs = theta * docTfidfVec

    val retMap = mutable.Map[String, Boolean]()

    for (i <- 0 to (docProbs.size - 1)) { retMap(codesIdxInv(i)) = (docProbs(i) > 0.0) }

    retMap
  }

}
