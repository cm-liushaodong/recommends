import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.sql.SparkSession

object UserCF {
  case class Rating(user:String, item:String, score:Double)

  def main(args:Array[String]): Unit ={
    val spark = SparkSession
      .builder()
      .appName(this.getClass.getSimpleName)
      .enableHiveSupport()
      .master("local[*]")
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark.sqlContext.setConf("spark.sql.files.ignoreCorruptFiles","true")
//    导入数据
    import spark.implicits._
    var data = spark.sparkContext
      .textFile("movielens\\ratings.dat")
      .map(_.split("::"))
      .map(attributes => Rating(attributes(0), attributes(1), attributes(2).trim.toDouble))
      .toDF()
//    数据预处理部分，对user和item添加索引
    val userIndex = new StringIndexer().setInputCol("user").setOutputCol("userIndex")
    data = userIndex.fit(data).transform(data)
    val itemIndex = new StringIndexer().setInputCol("item").setOutputCol("itemIndex")
    data = itemIndex.fit(data).transform(data)
//    建立临时表 user -> userIndex, item -> itemIndex的映射关系
    val user_index_map = data.select("user","userIndex").distinct()
    user_index_map.createGlobalTempView("user_map")
    val item_index_map = data.select("item","itemIndex").distinct()
    item_index_map.createGlobalTempView("item_map")
    data = data.select("userIndex","itemIndex","score")

  }
}
