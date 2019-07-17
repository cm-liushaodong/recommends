import scala.util.control.Breaks._
object Test {
  def main(args:Array[String]): Unit ={
    val score_arr = Array(1,3,4,5)
    val maxNum =10
    breakable{
      for(i <- 0 until maxNum){
        var max = (0,score_arr(0))
        for(j <- 0 until score_arr.length){
          if(max._2 < score_arr(j)){
            max = (j,score_arr(j))
          }
        }
  /*      if(max._2 == 0){
          break
        }*/
        println(max._1,max._2)
        score_arr(max._1) = 0
      }
    }

  }
}
