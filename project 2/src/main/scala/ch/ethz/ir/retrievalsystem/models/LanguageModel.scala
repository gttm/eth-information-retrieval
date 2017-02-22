package ch.ethz.ir.retrievalsystem.models

import scala.collection.parallel.ParMap
import scala.collection.parallel.ParSet
import scala.math._
import java.util.Calendar
import collection.mutable.{Map => MutMap}
import ch.ethz.dal.tinyir.math.ProbVector
import ch.ethz.dal.tinyir.io.TipsterStream

import ch.ethz.ir.retrievalsystem.processing.Utils

class LanguageModel (vocabulary: ParSet[String], 
                     numTopics: Int,
                     invertedIndex: ParMap[Int, List[(Int, Short)]], 
                     index: ParMap[Int, Map[Int, Short]], 
                     hashToDoc: Map[Int, String],
                     numIter: Int) extends Model {
  
  // for each word (string) an array of topic probabilities p(w|t)
  // current and new values
  var Pwt = MutMap[Int,ProbVector]()
  var Ptd = MutMap[Int,ProbVector]()
  
  vocabulary.seq.map(_.hashCode()).foreach { Pwt(_) = ProbVector.random(numTopics).normalize }
  index.seq.keys.foreach { Ptd(_) = ProbVector.random(numTopics).normalize }
      
  // compute and store topic distribution for a document  
  def topics(docHash:Int, doc: Map[Int,Short], num: Int, random: Boolean) = {
    var PtdNew = if (random == true) ProbVector.random(numTopics) else Ptd(docHash)
    for (i <- 0 until num ) PtdNew = iteration(PtdNew,doc)
    Ptd(docHash) = PtdNew
  }
    
  // one iteration of the generalized Csizar-Tusnady algorithm 
  private def iteration(Ptd : ProbVector, doc: Map[Int,Short]) : ProbVector = {
    val newPtd = ProbVector(new Array[Double](numTopics)) 
    for ((w,f) <- doc) newPtd += (Pwt(w) * Ptd).normalize(f)
    newPtd.normalize
  }  
  
  // compute updates for word distributions from single document
  def update (docHash:Int, doc: Map[Int,Short], num: Int) : MutMap[Int,ProbVector] = {
    val result = MutMap[Int,ProbVector]()
    for ((w,f) <- doc) result += w -> (Pwt(w) * Ptd(docHash)).normalize(f)
    result
  }
  
  def train() = {
    
    val docStream = index.toStream
    
  	val newPwt = MutMap[Int,ProbVector]()
  	var i = 1
  	
  	// run the Csizar-Tusnady algorithm in parallel for each document
    for ( doc <- docStream.par) {
      if ((i-1) % 1000 == 0) println(Calendar.getInstance().getTime() + ": i = " + (i-1)) ; i += 1
      topics(doc._1,doc._2,20,true)
    }
    
    // 
    for ( doc <- docStream ) {
      i = 0
      if ((i-1) % 1000 == 0) println(Calendar.getInstance().getTime() + ": i = " + (i-1)) ; i += 1
      val result = update(doc._1,doc._2,5)
      val increment = result.map{ 
        case (k,v) => k -> ( if (newPwt.contains(k)) v + newPwt(k) else v) 
      }
      increment.foreach{ case (k,a) => newPwt(k) = a }
    }
    
    //Pwt.clear; newPwt.foreach{ case (k,v) => Pwt += k->v }
     Pwt = newPwt
     val sums = Pwt.values.reduce((v1,v2) => v1 + v2)
     Pwt.foreach{ case (s,a) => Pwt.update(s,a/sums) } 
     i = 0
     for ( doc <- docStream.par) {
      if ((i-1) % 1000 == 0) println(Calendar.getInstance().getTime() + ": i = " + (i-1)) ; i += 1
      topics(doc._1,doc._2,20,true)
     }
 }  
    
  def getLogPwd(word: String, docHash: Int): Double = {
    log((Pwt(word.hashCode())*Ptd(docHash)).arr.sum)
  }

  // can be used for test queries too
  def performQueries(queries: Map[Int, List[String]]): ParMap[Int, List[Int]] = {
    queries.par.mapValues(queryTokens => (queryTokens,queryTokens.flatMap(token => invertedIndex(token.hashCode()).map(t => (t._1,token)))))
               .mapValues{case (t1,t2) => (t1,t2.groupBy(_._1))}.mapValues{case (t1,t2) => (t1,t2.flatMap{case (_,docTokenList) => docTokenList.map{case (t1,t2) => (t1,docTokenList.size)} }.toList)}
               .mapValues{case (queryTokens1,candidateDocuments) => candidateDocuments.map{ case (doc,numTokens) => (queryTokens1,doc,numTokens)}}
               .mapValues(_.map{case (tokens,docHash,numTokens) => (docHash,tokens.map(getLogPwd(_,docHash)).reduce(_ + _),numTokens)}
                           .sortBy(t => (t._3,t._2))
                           .reverse
                           .take(100)
                           .map(_._1))
  }
  
  def performQuery(query: List[String]): (List[Int], Long) = {
    println(Calendar.getInstance().getTime() + ": perform query with index " + query)
    val startTime = System.nanoTime
    
    val queryResult = query.flatMap(token => invertedIndex(token.hashCode()).map(_._1).map(docHash => (docHash,getLogPwd(token,docHash))))
                           .groupBy(_._1)
                           .mapValues{x => x.map(t => (t._1,t._2,x.size))}
                           .values
                           .map(_.reduce((t1,t2) => (t1._1,t1._2 + t2._2,t1._3)))
                           .toList
                           .sortBy(t => (t._3,t._2))
                           .reverse
                           .take(100)
                           .map(_._1)
   
    val queryTime = (System.nanoTime - startTime)/1000
    (queryResult, queryTime)
  }
  
  def performQueryNoIndex(query: List[String]): (List[Int], Long) = {
    println(Calendar.getInstance().getTime() + ": perform query without index " + query)
    val startTime = System.nanoTime
    
    val tipsterStream = new TipsterStream("data").stream
    
    //does not sort the results based on how many tokens of the query appear in the document, should produce worse results
    val queryResult = tipsterStream.filter(xmlDoc => Utils.docToTokens(xmlDoc.tokens,vocabulary).toSet.intersect(query.toSet).size > 0)
                                   .map(doc => (doc.name.hashCode(),query.map(token => getLogPwd(token,doc.name.hashCode())).reduce(_ + _)))
                                   .sortBy(_._2).reverse.take(100).map(_._1).toList
    
    val queryTime = (System.nanoTime - startTime)/1000
    (queryResult, queryTime)
  }
  
}