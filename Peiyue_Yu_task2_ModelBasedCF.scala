import java.io.{File, PrintWriter}

import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.mllib.recommendation.{ALS, Rating}

import scala.util.Sorting

object Peiyue_Yu_task2_ModelBasedCF {
  def main(arg: Array[String]): Unit = {
    val start_time = System.currentTimeMillis()
    val conf = new SparkConf().setAppName("LSH").setMaster("local[4]")
    val sc = new SparkContext(conf)
    val raw_data1 = sc.textFile(arg(0))
    val raw_data2 = sc.textFile(arg(1))
    val header1 = raw_data1.first()
    val header2 = raw_data2.first()
    val data1 = raw_data1.filter(x=>x!=header1).map(_.split(",")).map(x=>((x(0).toInt,x(1).toInt),x(2).toDouble))
    val data2 = raw_data2.filter(x=>x!=header2).map(_.split(",")).map(x=>((x(0).toInt,x(1).toInt),x(2).toDouble))
    val intersections = data1.map(_._1).intersection(data2.map(_._1)).collect()
    val ratings = data1.filter(e=>{val flag=intersections.contains(e._1);!flag}).map(e=>Rating(e._1._1.toInt,e._1._2.toInt,e._2))

    //    val ratings= rdd.map(_.split(",")).map(x=>(x(0),(x(1),x(2)))).groupByKey().filter(x=>x._2.size>0).
    //      map(x=>(x._1,x._2.toList(0)._1,x._2.toList(0)._2) match {case (user,item,rate) =>
    //        Rating(user.toInt, item.toInt, rate.toDouble)})

    //    val ratings = rdd.map(_.split(",")).map(x=>(x(0),(x(1),x(2))))


    val ratings2 = data2.map(x=>(x._1._1,x._1._2,x._2) match {case (user,item,rate) =>
      Rating(user.toInt, item.toInt, rate.toDouble)
    })

    //rating2 data
    val usersProducts = ratings2.map { case Rating(user, product, rate) =>
      (user, product)
    }

    val rank = 3
    val numIterations = 10
    val model = ALS.train(ratings, rank, numIterations,0.5,1,1)
    val predictions =
      model.predict(usersProducts).map { case Rating(user, product, rate) =>
        ((user, product), rate)
      }
    val min = predictions.map(e=>e._2).min()
    val max = predictions.map(e=>e._2).max()
    val predictionRefine = predictions.map(elem=>(elem._1,5*(elem._2-min)/(max-min)))

    val ratesAndPreds = ratings2.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictionRefine)

    val ratesAndPreds1 = ratings2.map { case Rating(user, product, rate) =>
      ((user, product), rate)
    }.join(predictionRefine).collect()

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
    val mresult = predictionRefine.map(x=>(x._1._1,x._1._2,x._2)).collect()
    Sorting.quickSort(mresult)
    val writer = new PrintWriter(new File(arg(2)))
    var output = ""
    for(i<-mresult){
      val pout = i._1.toString+"," + i._2.toString+","+i._3.toString+"\n"
      output = output+pout
    }
    writer.write(output)
    writer.close()
    val end = System.currentTimeMillis()
    println("Time: "+(end-start_time)/1000+" sec")

  }

}
