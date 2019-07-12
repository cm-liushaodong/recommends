package itemcf

import org.apache.spark.ml.feature.{IndexToString, StringIndexer}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, IndexedRow, MatrixEntry}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

/**
  * itemcf算法的scala实现，数据集采用的是movielens数据集
  *
  * @author liushaodong
  *         created by 2019-07-08
  */
object ItemCF {
  case class Rating(user:String, item:String, score:Double)

  def main(args:Array[String]): Unit ={
    val spark = SparkSession
      .builder()
      .appName(this.getClass.getSimpleName)
//      .master("local[*]")
      .enableHiveSupport()
      .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    spark.sqlContext.setConf("spark.sql.files.ignoreCorruptFiles","true")
//  导入数据，并转成dataFrame的格式
    import spark.implicits._
    var data = spark.sparkContext
      .textFile("C:\\Users\\cm\\IdeaProjects\\recommends\\movielens\\ratings.dat")
      .map(_.split("::"))
      .map(attributes => Rating(attributes(0), attributes(1), attributes(2).trim.toDouble))
      .toDF()
//    数据预处理，为字符串格式的user和item添加索引。形成新的data作为算法的输入
    val userIndex = new StringIndexer().setInputCol("user").setOutputCol("userIndex")
    data = userIndex.fit(data).transform(data)
    val itemIndex = new StringIndexer().setInputCol("item").setOutputCol("itemIndex")
    data = itemIndex.fit(data).transform(data)
//    临时表用来存储user->userIndex   item->itemIndex 的映射关系
    val user_index_map = data.select("user","userIndex").distinct()
    user_index_map.createOrReplaceTempView("user_map")
    val item_index_map = data.select("item","itemIndex").distinct()
    item_index_map.createOrReplaceTempView("item_map")
    data = data.select("userIndex","itemIndex","score")
//    data.printSchema()
    val similarityMatrix = getSimilarityMatrix(data)
//    为所有的用户进行推荐
    var result = recommend4All(data,similarityMatrix,0).toDF("userIndex","itemIndex","score")
    result.createOrReplaceTempView("result_table")
//    将数据预处理过程中添加的索引，再还原成原字符串
    result = spark.sql(
      """
        |SELECT user, item, max(score) AS score FROM (SELECT user, item, score FROM
        |result_table table1 LEFT JOIN user_map table2
        |ON table1.userIndex = table2.userIndex
        |LEFT JOIN item_map table3
        |ON table1.itemIndex = table3.itemIndex) table
        |GROUP BY user, item
      """.stripMargin)
    println(" recommends results  ")
    result.show()
  }
  /**
    * 生成一个item * item 的相似性矩阵
    * @param ratings 评分矩阵
    */
  def getSimilarityMatrix(ratings: DataFrame) = {
    val parseData = parseDataEntry(ratings)
    val ratings_coorM = new CoordinateMatrix(parseData)
    val ratings_rowM = ratings_coorM.toRowMatrix()
//    similarity_matrix_top是一个上三角阵 similarity_matrix_bottom是一个下三角阵
    val similarity_matrix_top = ratings_rowM.columnSimilarities()
    val similarity_rdd_top = similarity_matrix_top.toIndexedRowMatrix().rows.map{
      case IndexedRow(index, vector) => (index.toInt, vector)
    }
    val similarity_matrix_bottom = similarity_matrix_top.transpose()
    val similarity_rdd_bottom = similarity_matrix_bottom.toIndexedRowMatrix().rows.map{
      case IndexedRow(index, vector) => (index.toInt, vector)
    }
    val similarity_arr = Array.ofDim[Double](similarity_matrix_top.numCols().toInt,similarity_matrix_top.numCols().toInt)
    similarity_rdd_top.collect().foreach(row =>{
      val item_index = row._1
      val item_similarity = row._2
      for(i <- 0 until item_similarity.size){
        similarity_arr(item_index)(i) = item_similarity(i)
      }
    })
    similarity_rdd_bottom.collect().foreach(row=>{
      val item_index = row._1
      val item_similarity = row._2
      for(i <- 0 until item_similarity.size){
        if(item_similarity(i) > 0 ){
          similarity_arr(item_index)(i) = item_similarity(i)
        }
      }
    })
    similarity_arr
  }
  /**
    * 将dataframe转成RDD[MatrixEntry]
    * @param ratings
    * @return
    */
  def parseDataEntry(ratings:DataFrame): RDD[MatrixEntry] ={
    val parseData = ratings.rdd.mapPartitions(par =>{
      val data_arr = new ArrayBuffer[MatrixEntry]()
      par.foreach(line =>{
        val userIndex = line(0).toString.toDouble.toLong
        val itemIndex = line(1).toString.toDouble.toLong
        val score = line(2).toString.toDouble
        data_arr.+=(MatrixEntry(userIndex, itemIndex, score))
      })
      data_arr.iterator
    })
    parseData
  }
  /**
    * 为所有的用户进行推荐
    * @param data
    * @param similarityMatrix
    */
  def recommend4All(data: DataFrame, similarityMatrix: Array[Array[Double]], threshold:Double) = {
    val parseData = parseDataEntry(data)
    val ratings_coorMatrix = new CoordinateMatrix(parseData)
    val users_recommends = ratings_coorMatrix.toIndexedRowMatrix().rows.repartition(300).mapPartitions(par =>{
      val users_array = new ArrayBuffer[(Double,Double,Double)]()
      par.foreach(user =>{
        val userId = user.index //用户的userIndex
        val user_vector = user.vector //该用户访问过的历史item以及打分数据
        val user_hisItems = new mutable.HashSet[Double]()
        for(i<- 0 until user_vector.size){
          if(user_vector(i)!=0){
            user_hisItems.add(i)
          }
        }//用户访问过的items，已经存储到user_hisItems
        //为用户生成推荐列表
        for(i<- 0 until user_vector.size){
          if(user_vector(i)!=0){
            val item = similarityMatrix(i) //与主题i相似的所有主题
            for(j<- 0 until item.length){
              if(item(j)>=threshold & !user_hisItems.contains(j)){
                users_array.+=((userId,j,item(j)))
              }
            }
          }
        }
      })
      users_array.iterator
    })
    users_recommends
  }
}
