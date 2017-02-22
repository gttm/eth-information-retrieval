package ch.ethz.ir.retrievalsystem.processing

object SpellChecker {
  val spellingCorrections = Map[String,String]("^the" -> "the ","the$" -> " the","^at" -> "at ","^by" -> "by ","by$" -> " by",
                                               "^and" -> "and ","andm$" -> " and m","this$" -> " this","^this" -> "this ",
                                               "under$" -> " under", "^under" -> "under ","does$" -> " does","^does" -> "does ",
                                               "did$" -> " did","^did" -> "did ", "^for" -> "for ","to$" -> " to","of$" -> " of",
                                               "more$" -> " more","^more" -> "more ","but$" -> " but","^but" -> "but ",
                                               "^one" -> "one ","two$" -> " two","^two" -> "two ","^three" -> "three ",
                                               "^four" -> "four ","^five" -> "five ","^six" -> "six ","six$" -> " six",
                                               "^seven" -> "seven ","^eight" -> "eight ","^nine" -> "nine ", "^that" -> "that ",
                                               "that$" -> " that", "with$" -> " with", "^each" -> "each ", "^year" -> "year ",
                                               "year$" -> " year")
                                               
  def spellCheck(text: String) = {
    var newText = text
    spellingCorrections.foreach{ case(wrong,correct) =>
      if (wrong.r.findFirstIn(newText) != None) { 
        newText = newText.replaceAll(wrong,correct)
      }
    }
    newText
  }
}