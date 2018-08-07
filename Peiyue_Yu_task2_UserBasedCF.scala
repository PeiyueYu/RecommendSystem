import java.io.{File, PrintWriter}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.Rating

import scala.collection.mutable.ArrayBuffer
import scala.util.Sorting

object Peiyue_Yu_task2_UserBasedCF {
  def main(args: Array[String]): Unit = {
    val start_time = System.currentTimeMillis()
    val conf = new SparkConf().setAppName("USER_BASED").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val raw_data1 = sc.textFile("/Users/yupeiyue/Desktop/553/homework/Assignment_3/Data/video_small_num.csv")
    val raw_data2 = sc.textFile("/Users/yupeiyue/Desktop/553/homework/Assignment_3/Data/video_small_testing_num.csv")
    val header1 = raw_data1.first()
    val header2 = raw_data2.first()
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

    var predictArray = ArrayBuffer[((Int,Int),Double)]()

    for(i<-usersProducts){
      val p = i._2
      val u = i._1
      var wArray = ArrayBuffer[(Int,Double,Double,Double)]()
      val userAndRate= trainProductMap.get(p).head
      val np = userAndRate.size*1.0
      val productionSet1 = trainUserMap.get(u).head.map(x=>x._1)
      for(j<-userAndRate){
        val productionSet2 = trainUserMap.get(j._1).head.map(x=>x._1)
        //println("user "+j._1+" "+productionSet2)
        val interset = productionSet1&productionSet2
        //println(interset)
        if(interset.nonEmpty){
          var sum1 = 0.0
          var sum2 = 0.0
          var sum3 = 0.0
          val productionSet3 = productionSet2--Set(p)
          //println(j._1+"Set3"+productionSet3)
          for(item<-productionSet3){
            sum3 = sum3+trainUserMap.get(j._1).head.toMap.get(item).head
          }
          var mean3 = sum3/productionSet3.size
          for(item<-interset){
            sum1 = sum1+trainUserMap.get(u).head.toMap.get(item).head
            sum2 = sum2+trainUserMap.get(j._1).head.toMap.get(item).head
          }
          var mean1 = sum1/interset.size
          var mean2 = sum2/interset.size
          var c = 0.0
          var pow1 = 0.0
          var pow2 = 0.0
          for(item<-interset){
            c = c+(trainUserMap.get(u).head.toMap.get(item).head-mean1)*(trainUserMap.get(j._1).head.toMap.get(item).head-mean2)
            pow1 = pow1+Math.pow(trainUserMap.get(u).head.toMap.get(item).head-mean1,2)
            pow2 = pow2+Math.pow(trainUserMap.get(j._1).head.toMap.get(item).head-mean2,2)
          }
          var w = 0.0
          if(c!=0){
            w = c/(Math.sqrt(pow1)*Math.sqrt(pow2))
          }
          //println(u+" "+j._1+" "+w)
          //wArray.append((j._1,mean3,w*Math.pow(Math.abs(w),0.5)))
          //wArray.append((j._1,mean3,w))
          wArray.append((j._1,mean3,w,np)) //1.438
        }
      }
      var pus = 0.0
      var totalweight = 0.0
      var nnp = 0.0
      for(w<-wArray.toArray){
        pus = pus+(trainUserMap.get(w._1).head.toMap.get(p).head-w._2)*w._3
        totalweight = totalweight+Math.abs(w._3)
        nnp = w._4
      }
      var sum4 = 0.0
      for(item<-productionSet1){
        sum4=sum4+trainUserMap.get(u).head.toMap.get(item).head

      }
      var mean4 = sum4/productionSet1.size
      //println(mean4)
      var predictionv = mean4
      if(totalweight!=0){
        predictionv = (mean4+pus/totalweight)*Math.log(3374/nnp)
      }
      predictArray.append(((u,p),predictionv))
    }

    val pre = sc.parallelize(predictArray)
    val max = pre.map{case ((user,product),rate)=>rate}.max()
    val min = pre.map{case ((user,product),rate)=>rate}.min()
    val refinepre = pre.map{case ((user,product),rate)=>((user,product),5*(rate-min)/(max-min))}
    val refinepre2 = pre.map{case ((user,product),rate)=>{
      if(rate<0)
        ((user,product),0.0)
      else if(rate>5)
        ((user,product),5.0)
      else
        ((user,product),rate)
    }}

    val ratesAndPreds = ratings2.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(refinepre2)

    val ratesAndPreds1 = ratesAndPreds.collect()

    var l1 = 0
    var l2 = 0
    var l3 = 0
    var l4 = 0
    var b4 = 0
    var terr = 0.0

    for(i<-ratesAndPreds1){
      var err0 = i._2._1-i._2._2

      var err = Math.abs(err0)

      if(err<1){
        l1 = l1+1
      }
      else if(err<2){
        l2 = l2+1
      }
      else if(err<3){
        l3 = l3+1
      }
      else if(err<4){
        l4 = l4+1
      }
      else{
        b4 =b4+1
      }

      terr = terr+err*err
    }
    println(">=0 and <1: "+l1)
    println(">=1 and <2: "+l2)
    println(">=2 and <3: "+l3)
    println(">=3 and <4: "+l4)
    println(">=4: "+b4)
    println("RMSE: "+Math.sqrt(terr/ratesAndPreds1.length))
    val ubresult =refinepre2.collect().map(x=>(x._1._1,x._1._2,x._2))
    Sorting.quickSort(ubresult)
    val writer = new PrintWriter(new File("/Users/yupeiyue/Desktop/test.txt")
    var output = ""
    for(i<-ubresult){
      val pout = i._1.toString+"," + i._2.toString+","+i._3.toString+"\n"
      output = output+pout
    }
    writer.write(output)
    writer.close()
    val end = System.currentTimeMillis()
    println("Time: "+(end-start_time)/1000+" sec")


  }

}
