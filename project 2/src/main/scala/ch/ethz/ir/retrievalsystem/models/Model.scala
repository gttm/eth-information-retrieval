package ch.ethz.ir.retrievalsystem.models

import scala.collection.parallel.ParMap

trait Model {
  
  def train()
  
  def performQuery(query: List[String]): (List[Int], Long)
  
  def performQueryNoIndex(query: List[String]): (List[Int], Long)
  
}