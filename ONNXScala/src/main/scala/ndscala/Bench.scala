package org.sciscala.ndscala

object Bench extends App{

import scala.language.postfixOps
import scala.collection.immutable.ArraySeq
import org.sciscala.ndscala._
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.Tensors.Tensor._
import scala.concurrent._
import scala.concurrent.duration._
import scala.concurrent.ExecutionContext.Implicits.global
import scala.util.Random
import ONNXScalaOps._
import io.kjaer.compiletime._
import org.emergentorder.compiletime._

//Note: Allocation takes quite a while (on Java 11); works fine on Java 14+
val thisRandom = new Random(42)
val iters = 5
val lr = 0.000001f

type TT = "TensorTypeDenotation"
type TD = "TensorShapeDenotation" ##: TSNil

type TENKXTENK = 10000 #: 10000 #: SNil

type TENKXONE = 10000 #: 1000 #: SNil


val lrs:Tensor[Float, (TT, TD, TENKXONE)] = Tensor(Array.fill(10000000)(lr),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 10000 #: 1000 #: SNil)
val moreLrs:Tensor[Float, (TT,TD, TENKXTENK)] = Tensor(Array.fill(100000000)(lr),"TensorTypeDenotation","TensorShapeDenotation" ##: TSNil, 10000 #: 10000 #: SNil)

val ones = Tensor(Array.fill(10000000)(1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 10000 #: 1000 #: SNil)
val moreOnes = Tensor(Array.fill(100000000)(1.0f),"TensorTypeDenotation","TensorShapeDenotation" ##: TSNil, 10000 #: 10000 #: SNil)
//For scaling

val some10ks = Tensor(Array.fill(10000000)(10000.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 10000 #: 1000 #: SNil)
val more10ks = Tensor(Array.fill(100000000)(10000.0f),"TensorTypeDenotation","TensorShapeDenotation" ##: TSNil, 10000 #: 10000 #: SNil)

//TODO: check without type annotations when it's working
val arrX:Array[Float] = (Array.fill(100000000)(thisRandom.nextFloat)).map(_.toFloat)
val arrY:Array[Float] = (Array.fill(10000000)(thisRandom.nextFloat)).map(_.toFloat)
val arrW0:Array[Float] = (Array.fill(100000000)(thisRandom.nextFloat)).map(_.toFloat)
val arrW1:Array[Float] = (Array.fill(10000000)(thisRandom.nextFloat)).map(_.toFloat)

val y = Tensor(arrY,"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 10000 #: 1000 #: SNil)
val x = Tensor(arrX,"TensorTypeDenotation","TensorShapeDenotation" ##: TSNil, 10000 #: 10000 #: SNil)
//TODO: call recursively
var w0:Tensor[Float, (TT, TD, TENKXTENK)] = (Tensor(arrW0,"TensorTypeDenotation","TensorShapeDenotation" ##: TSNil, 10000 #: 10000 #: SNil) - moreOnes) / more10ks
var w1:Tensor[Float, (TT, TD, TENKXONE)] = (Tensor(arrW1,"TensorTypeDenotation","TensorShapeDenotation" ##: TSNil, 10000 #: 1000 #: SNil) - ones ) /some10ks


//Disabling broadcasting for ones may have slowed things down
//also lr multiplications
def train = {
//     val future = async {
      val l1 =  (x.matmul(w0)).sigmoid() // one / ((-(x dot w0)).exp() + one)
      val l2 = (l1.matmul(w1)).sigmoid() // one / ((-(l1 dot w1)).exp() + one)

      val error = y - l2
//      println("error: " + error.data.sum)
      val l2Delta = (error) * (l2 * (ones - l2))
      val l1Delta =  (l2Delta.matmul(w1.transpose)) * (l1 * (moreOnes - l1))

      //Simulate in-place += op here 
      
      w0 = w0 + (((x.transpose).matmul(l1Delta))) //*moreLrs)
      w1 = w1 + (((l1.transpose).matmul(l2Delta))) //*lrs)
    
}
val before = System.nanoTime; for (j <- 0 until iters) {
  val result = train
}; val after = System.nanoTime

println(after-before)

}

//For dex/ futhark comparison
/*
def pairwiseL1[Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, N <: Dimension, D <: Dimension](tens:Tensor[Float, (Tt, Td, N #: D #: SNil)]): Tensor[Float, (Tt, Td, N #: N #: SNil)] = {
((tens.transpose - tens.unsqueeze(Array(2))).abs).reduceSum(axis=1)

}

//def pairwiseL1[SomeTT, SomeTD, N #: D #: SNil](tens:Tensor[Float, (SomeTT, SomeTD, N #: D #: SNil)]): Tensor[Float, (SomeTT, SomeTD, N #: N #: SNil)] =
}
*/

