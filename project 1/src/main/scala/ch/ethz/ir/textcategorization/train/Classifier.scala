package ch.ethz.ir.textcategorization.train

import scala.collection.mutable
import ch.ethz.dal.tinyir.processing.XMLDocument

trait Classifier {
  def classifyDoc(doc: XMLDocument): mutable.Map[String,Boolean]
}