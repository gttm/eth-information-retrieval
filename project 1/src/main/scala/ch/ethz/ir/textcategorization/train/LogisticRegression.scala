package ch.ethz.ir.textcategorization.train

import ch.ethz.dal.tinyir.processing.XMLDocument

import ch.ethz.ir.textcategorization.processing._
import breeze.linalg._
import breeze.numerics._

import scala.collection.mutable

import scala.collection.parallel.immutable.ParSet

import java.util.Calendar

import ch.ethz.dal.tinyir.io.ReutersRCVStream

class LogisticRegression(featureGeneratorTrain: FeatureGenerator,
                         featureGeneratorValid: FeatureGenerator,
                         featureGeneratorPred: FeatureGenerator,
                         learningRate: Float,
                         numEpochs: Int,
                         maxNoImprovement: Int,
                         xmlListTrain: List[XMLDocument],
                         xmlListValid: List[XMLDocument],
                         codesTrain: Set[String],
                         codesEval: Set[String],
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
  codesIdx.foreach{case (code,codeid) => codesIdxInv(codeid) = code}
  
  //positive documents per category
  println(Calendar.getInstance().getTime() + ": calculating posDocs...")
  val posDocs = codesTrain.par.map(code => (code -> xmlListTrain.filter(_.codes(code)).map(_.ID).toSet)).toMap
  
  //negative documents per category
  println(Calendar.getInstance().getTime() + ": calculating negDocs...")
  val negDocs = codesTrain.par.map(code => (code -> docIds.filter(!posDocs.getOrElse(code,Set[Int]())(_)))).toMap
  
  //category probabilities
  println(Calendar.getInstance().getTime() + ": calculating catProb...")
  val catProb = codesTrain.par.map(code => code -> (posDocs.getOrElse(code,Set()).size.toFloat / (posDocs.getOrElse(code,Set()).size.toFloat + negDocs.getOrElse(code,Set()).size.toFloat))).toMap
  
  //class weighting coefficients
  println(Calendar.getInstance().getTime() + ": calculating aPlus and aMinus...")
  var aPlus = DenseVector.zeros[Float](numCodes)
  var aMinus = DenseVector.zeros[Float](numCodes)
  
  catProb.foreach{case (code,prob) => aPlus(codesIdx(code)) = prob}
  aMinus = 1f - aPlus
  
  val vocabSize = vocabulary.size
  
  //logistic regression weight vector
  var theta = DenseMatrix.zeros[Float](numCodes,vocabSize+1)
  
  //auxiliary vectors for vectorizing the update step
  val (auxVecPos,auxVecNeg) = {
     val auxVec = mutable.Map[Int,DenseVector[Float]]()
     val auxVecPos = mutable.Map[Int,DenseVector[Float]]()
     val auxVecNeg = mutable.Map[Int,DenseVector[Float]]() 
           
     for (docId <- xmlListTrain.map(_.ID)) {
       auxVec(docId) = DenseVector.zeros[Float](numCodes)
       auxVecPos(docId) = DenseVector.zeros[Float](numCodes)
       auxVecNeg(docId) = DenseVector.zeros[Float](numCodes)
     }
     
     for ((code,docIds) <- posDocs ; docId <- docIds) { auxVec(docId)(codesIdx(code)) = 1f }
     for (docId <- xmlListTrain.map(_.ID)) { 
       auxVecPos(docId) = aMinus :* auxVec(docId) 
       auxVecNeg(docId) = aPlus :* (1f - auxVec(docId))
     }
     
     (auxVecPos,auxVecNeg)
  }

  //training of logistic regression classifier
  def train() {
    val docIdsShuffled = Utils.shuffleDataset(docIds, numEpochs)
    var iter = 0
    var epoch = 0
    
    //tracking of best P/R/F1
    var bestP = 0f
    var bestR = 0f
    var bestF1 = 0f
    
    var noImprovement = 0
    
    while (epoch < numEpochs) {
      while (iter <  numDocs) { 
        val docId = docIdsShuffled(iter)
        
        val docAuxVecPos = auxVecPos(docId)
        val docAuxVecNeg = auxVecNeg(docId)
        
        val docTfidfVec = featureGeneratorTrain.tfidfVec(docId)
              
        val probs = vecOnes :/ (vecOnes + exp(-theta*docTfidfVec))
        
        val gradient = -(((docAuxVecPos - (docAuxVecPos :* probs)) + (-docAuxVecNeg :* probs)) * docTfidfVec.t)
        
        val epochLearningRate = if (epoch < 10) learningRate else learningRate / (epoch-9)
                
        theta -= epochLearningRate * gradient
        
        if (iter % 1000 == 0) {println(Calendar.getInstance().getTime() + ": iter = " + iter)}
        
        iter += 1
      }
      val (p,r,f1) = Utils.evaluate(xmlListValid,this,codesEval)
      println(Calendar.getInstance().getTime() + ": epoch : " + epoch + " P/R/F1 = " + (p,r,f1))
      
      //early stopping
      if (f1 > bestF1) {
        bestP = p
        bestR = r
        bestF1 = f1
        noImprovement = 0
      }
      else {
        noImprovement += 1
        if (noImprovement == maxNoImprovement) {
          epoch = numEpochs
          println(Calendar.getInstance().getTime() + ": Terminated, final P/R/F1 = " + (p,r,f1))
        }
      }
      iter = 0
      epoch += 1
      
    }
  }
  
  //classifies a document for each category code
  def classifyDoc(doc: XMLDocument) = {
    val docId = doc.ID
    val docTfidfVec = if (featureGeneratorValid.docIds.contains(docId)) featureGeneratorValid.tfidfVec(docId) else featureGeneratorPred.tfidfVec(docId)
    val docProbs = DenseVector.ones[Float](numCodes) :/ (DenseVector.ones[Float](numCodes) + exp(-theta*docTfidfVec))
    
    val retMap = mutable.Map[String,Boolean]()
    
    for(i <- 0 to (docProbs.size-1)) { retMap(codesIdxInv(i)) = (docProbs(i) > 0.5) }
    
    retMap
  }
    
}