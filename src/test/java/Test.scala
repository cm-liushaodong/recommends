object Test {
  def main(args:Array[String]): Unit ={
    val str = "1ratings_test101::5::1000"
    val arr = str.split("::")
    arr.foreach(println)
  }
}
