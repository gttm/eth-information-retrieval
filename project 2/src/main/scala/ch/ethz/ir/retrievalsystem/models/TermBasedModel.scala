package ch.ethz.ir.retrievalsystem.models
import scala.collection.parallel.ParMap
import scala.collection.parallel.ParSet
import scala.math._
import ch.ethz.dal.tinyir.io.TipsterStream
import ch.ethz.ir.retrievalsystem.processing.Utils
import ch.ethz.ir.retrievalsystem.main.RetrievalSystem
import java.util.Calendar
class TermBasedModel(vocabulary: ParSet[String], 
                     invertedIndex: ParMap[Int, List[(Int, Short)]], 
                     index: ParMap[Int, Map[Int, Short]],
                     mu: Float,
                     lambda: Float,
                     smoothing: String) extends Model {
  
  var logPwdFinal = Map[Int, Map[Int, Double]]()
  var lambdas = Map[Int, Float]()
  
  def train() = {
    val docSizes = index.mapValues(_.map(_._2.toInt).sum).map(identity)
    
    if (smoothing == "mercer") {
       // Jelinek Mercer smoothing
       lambdas = docSizes.mapValues(_ => lambda).seq.toMap
    }
    else if (smoothing == "bayes") {
       // Bayes smoothing with Dirichlet priors
       lambdas = docSizes.mapValues(docSize => mu/(docSize + mu)).seq.toMap
    }
    else {
      throw new IllegalArgumentException("unknown smoothing method")
    }
    
    logPwdFinal = {
      val lambdaFactors = lambdas.mapValues{lambda => (1f - lambda)/lambda}.map(identity)    
      val Pwd = invertedIndex.mapValues(_.map{case (docHash, tokenTf) => docHash -> 1f*tokenTf/docSizes(docHash)}.toMap)
     
      val Pw = {
         val cfTotal = invertedIndex.flatMap(_._2.map(_._2.toInt)).sum
         invertedIndex.mapValues(1f*_.map(_._2).sum/cfTotal).map(identity)
      }
      Pwd.map{case (word, docMap) => word -> docMap.map{case (docHash, pwd) => docHash -> log(1f + lambdaFactors(docHash)*pwd/Pw(word))}}.seq.toMap
    }
  }
  
  def getLogPwd(word: Int, docHash: Int): Double = {
    logPwdFinal(word)(docHash)
  }
  
  //performs a query using the index
  def performQuery(query: List[String]): (List[Int], Long) = {
    println(Calendar.getInstance().getTime() + ": perform query with index " + query)
    val startTime = System.nanoTime
    
    val queryResult = query.map(_.hashCode()).flatMap(token => invertedIndex(token).map{case (docHash, _) => (docHash, getLogPwd(token, docHash))})
            .groupBy(_._1)
            .map{case (docHash, pwdList) => docHash -> (pwdList.map(_._2).sum + pwdList.size*log(lambdas(docHash)))}
            .toList
            .sortBy(-_._2)
            .take(100)
            .map(_._1)
           
    val queryTime = (System.nanoTime - startTime)/1000
    (queryResult, queryTime)
  }
  
  //performs a query without using the index
  def performQueryNoIndex(query: List[String]): (List[Int], Long) = {
    println(Calendar.getInstance().getTime() + ": perform query without index " + query)
    val startTime = System.nanoTime
    
    val tipsterStream = new TipsterStream("data").stream
    
    //does not work properly
    val queryResult = tipsterStream.filter(xmlDoc => Utils.docToTokens(xmlDoc.tokens,vocabulary).toSet.intersect(query.toSet).size > 0)
                                   .map(doc => (doc.name.hashCode(),query.map(token => getLogPwd(token.hashCode(),doc.name.hashCode())).reduce(_ + _)))
                                   .sortBy(_._2).reverse.take(100).map(_._1).toList
    
    val queryTime = (System.nanoTime - startTime)/1000
    (queryResult, queryTime)
  }
  
}