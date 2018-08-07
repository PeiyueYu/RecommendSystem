import java.io.{File, PrintWriter}

import breeze.linalg._
import org.apache.spark.{SparkConf, SparkContext}
import java.util.Random

import scala.collection.mutable.ArrayBuffer
import scala.util.Sorting
object Peiyue_Yu_task1_Jaccard {
  //create hashfunction
  def hash(denseVector: DenseVector[Int], m: Int): DenseVector[Int] = {
    val a = (new Random).nextInt(3000)
    var b = (new Random).nextInt(m)
    while(b%2==0|b%7==0){
      b = (new Random).nextInt(m)
    }
    val hashv = DenseVector.zeros[Int](m)
    for (i <- denseVector) {
      var t = ((a * i + b)%3943)%m
      hashv(i) = t
    }
    hashv
  }

  //minhash
  def minhash(mmatrix: DenseMatrix[Int], num: Int, pc: Int, uc: Int): DenseMatrix[Int] = {
    val res: DenseMatrix[Int] = DenseMatrix.fill(num, pc) {
      pc * 10
    }
    val denseV: DenseVector[Int] = DenseVector.range(0, uc)
    var p = mmatrix
    for (i <- 1 to num) {
      val hashv = hash(denseV, uc).toDenseMatrix.t
      p = DenseMatrix.horzcat(p, hashv)
    }
    for (i <- 0 until uc) {
      for (j <- 0 until pc) {
        if (p(i, j) == 1) {
          //res(0:2,j)->p(i,pcount:pcount+2)
          for (q <- 0 until num) {
            if (res(q, j) >= p(i, pc + q)) {
              res(q, j) = p(i, pc + q)
            }
          }
        }
      }
    }
    res
  }

  //candidate of each band
  def createcandidate(minhm: DenseMatrix[Int], pc: Int): Set[List[Int]] = {
    var candi = Set.empty[List[Int]]
    for (i <- 0 until pc - 1) {
      for (j <- i + 1 until pc) {
        if (minhm(::, i) == minhm(::, j)) {
          candi = candi + List(i, j).sorted
        }
      }
    }
    candi
  }

  def jaccard(candi:Set[List[Int]],data:List[(Int,Iterable[Int])]):Array[(Int,Int,Double)]={
    var ff = ArrayBuffer[(Int,Int,Double)]()
    for(i<-candi){
      val c1= data(i(0))._2.toSet
      val c2 = data(i(1))._2.toSet
      val inter:Double = (c1&c2).size*1.0
      val uni:Double= (c1|c2).size*1.0
      val j = inter/uni
      if(j>=0.5){
        ff.append((i(0),i(1),j))
      }
    }
    ff.toArray
  }

  def main(arg: Array[String]): Unit = {
    val start_time = System.currentTimeMillis()
    val conf = new SparkConf().setAppName("LSH").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val raw_data = sc.textFile(arg(0))
    val header = raw_data.first()
    val rdd = raw_data.filter(x => x != header).map(_.split(",")).cache()
    val data = rdd.map(x => (x(1).toInt, x(0).toInt)).groupByKey()
    val rdata:List[(Int,Iterable[Int])] = data.collect().toList.sorted
    val pcount = data.count().toInt
    val ucount = rdd.map(x => x(0).toInt).distinct().count().toInt
    //original matrix m1
    val m1 = DenseMatrix.zeros[Int](ucount, pcount)
    for (i <- rdata) {
      for (j <- i._2)
        m1(j, i._1) = 1
    }

    val r = 5
    val b = 36
    val threshold = 0.5
    var candi = Set.empty[List[Int]]
    for (i <- 1 to b) {
      var hmatrix = minhash(m1, r, pcount, ucount)
      candi = candi ++ createcandidate(hmatrix, pcount)
    }


//
//    //calculate presicion and recall
//    var test_data = sc.textFile("/Users/yupeiyue/Desktop/553/homework/Assignment_3/Data/video_small_ground_truth_jaccard.csv")
//    var tt:Set[List[Int]]= test_data.map(_.split(",")).map(x=>List(x(0).toInt,x(1).toInt).sorted).collect().toSet
//    val tp:Double = (candi&tt).size*1.0
//    val fp:Double = (candi -- tt).size*1.0
//    val fn:Double = (tt -- candi).size*1.0
//    val precision:Double = tp/(tp+fp)
//    val recall:Double = tp/(tp+fn)
//    println("Precision is: "+precision+" and recall is: "+recall)

    val ja = jaccard(candi,rdata)
    Sorting.quickSort(ja)
    val writer = new PrintWriter(arg(1))
    var output = ""
    for(i<-ja){
      val pout = i._1.toString+"," + i._2.toString+","+i._3.toString+"\n"
      output = output+pout
    }
    writer.write(output)
    writer.close()

    val end = System.currentTimeMillis()
    println("Time: "+(end-start_time)/1000+" sec")

  }

}
