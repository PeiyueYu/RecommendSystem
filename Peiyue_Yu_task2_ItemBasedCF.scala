import java.io.{File, PrintWriter}
import java.util.Random

import breeze.linalg.{DenseMatrix, DenseVector}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.Rating

import scala.collection.mutable.ArrayBuffer
import scala.util.Sorting

object Peiyue_Yu_task2_ItemBasedCF {
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
      ff.append((i(0),i(1),j))
      ff.append((i(1),i(0),j))
    }
    ff.toArray
  }

  def main(arg: Array[String]): Unit = {
    val start_time = System.currentTimeMillis()
    val conf = new SparkConf().setAppName("LSH").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val raw_data1 = sc.textFile(arg(0))
    val raw_data2 = sc.textFile(arg(1))
    val header1 = raw_data1.first()
    val header2 = raw_data2.first()
    val rdd = raw_data1.filter(x => x != header1).map(_.split(",")).cache()
    val data = rdd.map(x => (x(1).toInt, x(0).toInt)).groupByKey()
    val rdata: List[(Int, Iterable[Int])] = data.collect().toList.sorted
    val pcount = data.count().toInt
    val ucount = rdd.map(x => x(0).toInt).distinct().count().toInt

    //itembased data preprocessing
    val data1 = raw_data1.filter(x=>x!=header1).map(_.split(",")).map(x=>((x(0).toInt,x(1).toInt),x(2).toDouble))
    val data2 = raw_data2.filter(x=>x!=header2).map(_.split(",")).map(x=>((x(0).toInt,x(1).toInt),x(2).toDouble))
    val intersections = data1.map(_._1).intersection(data2.map(_._1)).collect()
    val ratings = data1.filter(e=>{val flag=intersections.contains(e._1);!flag}).map(e=>Rating(e._1._1.toInt,e._1._2.toInt,e._2))


    val ratings2 = data2.map(x=>(x._1._1,x._1._2,x._2) match {case (user,item,rate) =>
      Rating(user.toInt, item.toInt, rate.toDouble)
    })


    val trainProductMap=ratings.map{case Rating(user, product, rate)=>(product,(user,rate))}.groupByKey().map(e=>(e._1,e._2.toSet)).collect().toMap
    val trainUserMap=ratings.map{case Rating(user, product, rate)=>(user,(product,rate))}.groupByKey().map(e=>(e._1,e._2.toSet)).collect().toMap


    //rating2 data
    val usersProducts = ratings2.map { case Rating(user, product, rate) =>
      (user, product)
    }.collect()


    //original matrix m1
    val m1 = DenseMatrix.zeros[Int](ucount, pcount)
    for (i <- rdata) {
      for (j <- i._2)
        m1(j, i._1) = 1
    }

    val r = 5
    val b = 35
    val threshold = 0.5
    var candi = Set.empty[List[Int]]
    for (i <- 1 to b) {
      var hmatrix = minhash(m1, r, pcount, ucount)
      candi = candi ++ createcandidate(hmatrix, pcount)
    }

    val ja = jaccard(candi, rdata)
    Sorting.quickSort(ja)
    val scja = sc.parallelize(ja).groupBy(_._1).map{
      case(item,similaritem)=>(item,similaritem.toArray.sortBy(x=>x._3).reverse.take(10))
    }
    val scjja = scja.collect().toMap
    //    val xd= scjja.get(4904).head.map(x=>x._2).toSet
    //    println(xd)




    var predictArray1 = ArrayBuffer[((Int,Int),Double)]()
    for(i<-usersProducts){
      val p =i._2
      val u = i._1
      var xd = Set.empty[Int]
      if(scjja.contains(p)) {
        xd = scjja.get(p).head.map(x => x._2).toSet
      }
      var wArray = ArrayBuffer[(Int,Double)]()
      val itemAndRate = trainUserMap.get(u).head
      val userSet1 = trainProductMap.get(p).head.map(x=>x._1)
      for(item<-itemAndRate){
        val userSet2 = trainProductMap.get(item._1).head.map(x=>x._1)
        val interset = userSet1&userSet2
        if(interset.nonEmpty){
          var sum1 = 0.0
          var sum2 = 0.0
          for(user<-interset){
            sum1 = sum1+trainUserMap.get(user).head.toMap.get(p).head
            sum2 = sum2+trainUserMap.get(user).head.toMap.get(item._1).head
          }
          var mean1 = sum1/interset.size
          var mean2 = sum2/interset.size
          var c = 0.0
          var pow1 = 0.0
          var pow2 = 0.0
          for(user<-interset){
            c = c+(trainUserMap.get(user).head.toMap.get(p).head-mean1)*(trainUserMap.get(user).head.toMap.get(item._1).head-mean2)
            pow1 = pow1 + Math.pow(trainUserMap.get(user).head.toMap.get(p).head-mean1,2)
            pow2 = pow2 + Math.pow(trainUserMap.get(user).head.toMap.get(item._1).head-mean2,2)
          }
          var w = 0.0
          if(c!=0 & xd.contains(item._1)){
            w = c/(Math.sqrt(pow1)*Math.sqrt(pow2))
          }
          wArray.append((item._1,w))
        }
      }
      val wwArray=wArray.toArray.sortBy(x=>x._2).reverse.take(2)
      var pus = 0.0
      var totalweight = 0.0
      for(w<-wwArray){
        totalweight = totalweight+Math.abs(w._2)
        pus = pus +trainUserMap.get(u).head.toMap.get(w._1).head*w._2
      }
      var sum4 = 0.0
      for(user<-userSet1){
        sum4 = sum4+trainUserMap.get(user).head.toMap.get(p).head
      }
      var predictionv = sum4/userSet1.size
      if(totalweight!=0){
        predictionv = pus/totalweight
      }
      predictArray1.append(((u,p),predictionv))
    }
    //println(predictArray.toList)
    val pre1 = sc.parallelize(predictArray1)
    val max1 = pre1.map{case ((user,product),rate)=>rate}.max()
    val min1 = pre1.map{case ((user,product),rate)=>rate}.min()
    val refineprel = pre1.map{case ((user,product),rate)=>((user,product),5*(rate-min1)/(max1-min1))}
    val refineprel2 = pre1.map{case ((user,product),rate)=>{
      if(rate<0)
        ((user,product),0.0)
      else if(rate>5)
        ((user,product),5.0)
      else
        ((user,product),rate)
    }}

    val ratesAndPreds1 = ratings2.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(refineprel2)

    val ratesAndPredsl1 = ratesAndPreds1.collect()

    var ll1 = 0
    var ll2 = 0
    var ll3 = 0
    var ll4 = 0
    var bb4 = 0
    var terr1 = 0.0

    for(i<-ratesAndPredsl1){
      var err0 = i._2._1-i._2._2

      var err = Math.abs(err0)

      if(err<1){
        ll1 = ll1+1
      }
      else if(err<2){
        ll2 = ll2+1
      }
      else if(err<3){
        ll3 = ll3+1
      }
      else if(err<4){
        ll4 = ll4+1
      }
      else{
        bb4 =bb4+1
      }

      terr1 = terr1+err*err
    }

    println("ItemBasedCF integrating LSH:")
    println(">=0 and <1: "+ll1)
    println(">=1 and <2: "+ll2)
    println(">=2 and <3: "+ll3)
    println(">=3 and <4: "+ll4)
    println(">=4: "+bb4)
    println("RMSE: "+Math.sqrt(terr1/ratesAndPredsl1.length))

    val ibresult = refineprel2.map(x=>(x._1._1,x._1._2,x._2)).collect()
    Sorting.quickSort(ibresult)
    val writer = new PrintWriter(new File(arg(2)))
    var output = ""
    for(i<-ibresult){
      val pout = i._1.toString+"," + i._2.toString+","+i._3.toString+"\n"
      output = output+pout
    }
    writer.write(output)
    writer.close()

  }


}
