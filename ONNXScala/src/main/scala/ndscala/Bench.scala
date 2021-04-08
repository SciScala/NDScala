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
import org.emergentorder.compiletime.TensorShapeDenotation.Reverse

import org.emergentorder.onnx.backends._

//Note: Allocation takes quite a while (on Java 11); works fine on Java 14+
val thisRandom = new Random(42)
val iters = 1 //5 //TODO RESTORE
val lr = 0.0000001f

type TT = "TensorTypeDenotation"
type TD = "TensorShapeDenotation" ##: TSNil

type TENKXTENK = 10000 #: 10000 #: SNil
val mat10kX10k = shapeOf[TENKXTENK]

type TENKXONEK = 10000 #: 1000 #: SNil
val mat10kX1k = shapeOf[TENKXONEK]

val axisLabels = tensorShapeDenotationOf[TD]

val lrs:Tensor[Float, (TT, TD, TENKXONEK)] = Tensor(Array.fill(10000000)(lr),"TensorTypeDenotation", axisLabels, mat10kX1k)
val moreLrs:Tensor[Float, (TT,TD, TENKXTENK)] = Tensor(Array.fill(100000000)(lr),"TensorTypeDenotation", axisLabels, mat10kX10k)

val ones = Tensor(Array.fill(10000000)(1.0f),"TensorTypeDenotation", axisLabels, mat10kX1k)
val moreOnes:Tensor[Float, (TT,TD, TENKXTENK)] = Tensor(Array.fill(100000000)(1.0f),"TensorTypeDenotation", axisLabels, mat10kX10k)
//For scaling
val some10ks = Tensor(Array.fill(10000000)(10000.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 10000 #: 1000 #: SNil)
val more10ks:Tensor[Float, (TT,TD, TENKXTENK)] = Tensor(Array.fill(100000000)(10000000.0f),"TensorTypeDenotation", axisLabels, mat10kX10k)


//Hacking in a bias node
val arrX:Array[Float] = (Array.fill(100000000)(thisRandom.nextFloat)).map(_.toFloat).zipWithIndex
  .map { case (e, i) => if((i) % 10000 == 0) 1.0f else e}
   

//println("arr X " + arrX(0) + " anotheR " + arrX(1))
val arrY:Array[Float] = (Array.fill(10000000)(thisRandom.nextFloat)).map(_.toFloat)
val arrW0:Array[Float] = (Array.fill(100000000)(thisRandom.nextFloat)).map(_.toFloat)
val arrW1:Array[Float] = (Array.fill(10000000)(thisRandom.nextFloat)).map(_.toFloat)

val y:Tensor[Float, (TT, TD, TENKXONEK)] = Tensor(arrY,"TensorTypeDenotation",  axisLabels, mat10kX1k) 
val x:Tensor[Float, (TT, TD, TENKXTENK)] = Tensor(arrX,"TensorTypeDenotation", axisLabels, mat10kX10k)

var w0:Tensor[Float, (TT, TD, TENKXTENK)] = (Tensor(arrW0,"TensorTypeDenotation", axisLabels, mat10kX10k) - moreOnes) / more10ks
var w1:Tensor[Float, (TT, TD, TENKXONEK)] = (Tensor(arrW1,"TensorTypeDenotation", axisLabels, mat10kX1k) - ones) / some10ks

//var w1:Tensor[Float, (TT, TD, TENKXONE)] = (Tensor(arrW1,"TensorTypeDenotation","TensorShapeDenotation" ##: TSNil, 10000 #: 1000 #: SNil) - ones ) /some10ks

//TODO: test tracing, exporting to onnx, and running the result here
//Disabling broadcasting for ones may have slowed things down
//also lr multiplications

val someOnes:Tensor[Float, (TT,TD, TENKXTENK)] = org.sciscala.ndscala.given_NDArrayOps_Tensor.ones[Float, TT, TD, TENKXTENK](shapeOf[TENKXTENK].toSeq.toArray)

def forwardLayer[Dim <: Dimension, Dim1 <: Dimension, Dim2 <: Dimension, S <: Dim #: Dim1 #: SNil, S1 <: Dim1 #: Dim2 #: SNil](input:Tensor[Float, (TT, TD, S)], weights:Tensor[Float, (TT, TD, S1)])(using s: ShapeOf[S], s1: ShapeOf[S1], s2: ShapeOf[Dim #: Dim2 #: SNil],vd:ValueOf[compiletime.ops.int.S[Dim]], vd1:ValueOf[compiletime.ops.int.S[Dim1]], vd2: ValueOf[compiletime.ops.int.S[Dim2]]): Tensor[Float, (TT,TD, Dim #: Dim2 #: SNil)]
                         = {
                          val res = input.matmul(weights)
                          res.sigmoid()
}

//update + request, combined
//
//accepting forward result as an input to prevent redundancy - found bug
def backwardLayer[Dim <: Dimension, Dim1 <: Dimension, Dim2 <: Dimension](input:Tensor[Float, (TT, TD,  Dim #: Dim1 #: SNil)], weights:Tensor[Float, (TT, TD,  Dim1 #: Dim2 #: SNil)], 
  output:Tensor[Float, (TT, TD,  Dim #: Dim2 #: SNil)], forwardResult: Tensor[Float, (TT, TD,  Dim #: Dim2 #: SNil)], trans: Tensor[Float, (TT, TD, Dim1 #: Dim #: SNil)], someOnes: Tensor[Float, (TT, TD,  Dim #: Dim2 #: SNil)], ones: Tensor[Float, (TT, TD,  Dim #: Dim1 #: SNil)],  moreLrs:  Tensor[Float, (TT, TD,  Dim1 #: Dim2 #: SNil)])(using s: ShapeOf[ Dim #: Dim1 #: SNil], s1: ShapeOf[ Dim1 #: Dim2 #: SNil], s2: ShapeOf[ Dim #: Dim2 #: SNil], s3: ShapeOf[Dim1 #: Dim #: SNil],vd:ValueOf[compiletime.ops.int.S[Dim]], vd1:ValueOf[compiletime.ops.int.S[Dim1]], vd2: ValueOf[compiletime.ops.int.S[Dim2]], vd3: ValueOf[Dim1], vd4: ValueOf[Dim2]):Tuple2[Tensor[Float, (TT, TD, Dim1 #: Dim2 #: SNil)],Tensor[Float, (TT, TD,  Dim #: Dim1 #: SNil)]] = {
      val l = forwardResult //(input.matmul(weights)).sigmoid() //forwardResult
      val error = output - l
      //println("op by op pred: " + l.data(0))
      println("ERROR : " + error.data.sum)
      val onesSubL = someOnes.constant() - l //need this because of same input in one expression limitation .. ? nope
      val lDelta = (error) * (l * (onesSubL))
      //val trans = input.transpose(Array(1,0)) //TODO: looks like this duplicates in multiple runs, causing fail
      val constLrs = moreLrs.constant()
     // val requested = (l + lDelta) matmul (weights.transpose(Array(1,0)))
      val requested = (((lDelta) matmul (weights.transpose(Array(1,0)))) * (input*(ones.constant() - input))) + input //weights aren't last anymore
      val weightsUpd = weights + (((trans).matmul(lDelta)) * constLrs)
      println("the input : " + input.data(0) + " the requested input : " + requested.data(0))
      // (logit(output, someOnes):Tensor[Float, (TT, TD,  Dim #: Dim2 #: SNil)]) matmul ((weightsUpd.inverse()):Tensor[Float, (TT, TD,  Dim2 #: Dim1 #: SNil)]) 
      (weightsUpd, requested)
}

def logit[S <: Shape](input:Tensor[Float, (TT, TD, S)], someOnes:Tensor[Float, (TT, TD, S)])(using s: ShapeOf[S]) = {
  val result =  - ((someOnes/input) - someOnes).log()
  result
}

val trans:Tensor[Float, (TT,TD, TENKXTENK)] = x.transpose(Array(1,0))

//TODO: need a way to embed constant initializers or Constant op i.e. trained weights inside the graph
//constant op exposed here now, should work if you just declare the constant and it's input inside the fence
def train[Dim <: Dimension, Dim1 <: Dimension, Dim2 <: Dimension, Dim3 <: Dimension](input:Tensor[Float, (TT, TD, Dim #: Dim1 #: SNil)], weights:Tensor[Float, (TT, TD, Dim1 #: Dim2 #: SNil)], weights1 :Tensor[Float, (TT, TD, Dim2 #: Dim3 #: SNil)],
  output:Tensor[Float, (TT, TD, Dim #: Dim3 #: SNil)], trans: Tensor[Float, (TT, TD, Dim1 #: Dim #: SNil)], someOnes: Tensor[Float, (TT, TD, Dim #: Dim2 #: SNil)], someOnes1: Tensor[Float, (TT, TD, Dim #: Dim3 #: SNil)], someOnes2: Tensor[Float, (TT, TD, Dim #: Dim1 #: SNil)], moreLrs:  Tensor[Float, (TT, TD, Dim1 #: Dim2 #: SNil)], moreLrs1:  Tensor[Float, (TT, TD, Dim2 #: Dim3 #: SNil)], count: Int)(using s: ShapeOf[Dim #: Dim1 #: SNil], s1: ShapeOf[Dim1 #: Dim2 #: SNil], s2: ShapeOf[Dim #: Dim2 #: SNil], s3: ShapeOf[Dim2 #: Dim3 #: SNil], s4: ShapeOf[Dim #: Dim3 #: SNil], vd: ValueOf[compiletime.ops.int.S[Dim]], vd1: ValueOf[compiletime.ops.int.S[Dim1]], vd2: ValueOf[compiletime.ops.int.S[Dim2]], vd3: ValueOf[compiletime.ops.int.S[Dim3]], vd4: ValueOf[Dim], vd5: ValueOf[Dim1], vd6: ValueOf[Dim2], vd7: ValueOf[Dim3]): Tuple2[Tensor[Float, (TT, TD, Dim1 #: Dim2 #: SNil)],Tensor[Float, (TT, TD, Dim2 #: Dim3 #: SNil)]] = {
//(using s: ShapeOf[dim0.type #: dim2.type #: SNil], s1: ShapeOf[dim1.type #: dim2.type #: SNil], vd: ValueOf[scala.compiletime.ops.int.S[dim0.type]], vd1: ValueOf[scala.compiletime.ops.int.S[dim1.type]], vd2: ValueOf[scala.compiletime.ops.int.S[dim2.type]]) = {

if(count == 0) {
  (weights, weights1)
}
else{
  val l1: Tensor[Float, (TT, TD, Dim #: Dim2 #: SNil)] = forwardLayer(input, weights)
  val l2: Tensor[Float, (TT, TD, Dim #: Dim3 #: SNil)] = forwardLayer(l1, weights1)
  val l1trans: Tensor[Float, (TT, TD, Dim2 #: Dim #: SNil)] = l1.transpose(Array(1,0))
  val weightsAndRequest: Tuple2[Tensor[Float, (TT, TD, Dim2 #: Dim3 #: SNil)],Tensor[Float, (TT, TD, Dim #: Dim2 #: SNil)]] = backwardLayer(l1, weights1, output, l2, l1trans, someOnes1, someOnes, moreLrs1)
  val weightsUpd1 = weightsAndRequest._1
  val requested1 = weightsAndRequest._2
//  l1.shape.foreach(println)
//  println("SEP")
//  println(requested1)
//  println("Dim2 " + valueOf[Dim2])
//  requested1.shape.foreach(println)
  val weightsUpd:Tensor[Float, (TT, TD, Dim1 #: Dim2 #: SNil)] = backwardLayer(input, weights, requested1, l1, trans, someOnes, someOnes2, moreLrs)._1
  //only weightsUpd is returned in fused version due to 1 output limitation
  train[Dim,Dim1,Dim2,Dim3](input, weightsUpd, weightsUpd1, output, trans, someOnes, someOnes1, someOnes2, moreLrs, moreLrs1, count - 1)
}
//  (0 until 5).backwardLayer(x, w0, y)
 /*
  val requested = request(x,w0,y)
  println(x)
  println(x.data(0))
  println(requested)
  println(requested.data(0))
*/
}

//layerwise backprop iteration
  //Fencing with fuseOps calls
  fuseOps
  val before = System.nanoTime

  //  println("L1 : " + l1.data(0))
  val (w0upd: Tensor[Float, (TT, TD, TENKXTENK)], w1upd: Tensor[Float, (TT, TD, TENKXONEK)]) = train[10000, 10000, 10000, 1000] (x,w0,w1,y, trans, someOnes, ones, someOnes, moreLrs, lrs,1)
  val after = System.nanoTime

  println("train time , op by op " + (after-before))  
  println("DONE train " + w0upd.data(0))
  //Second pass just to see if error improved:
//  train(forwardLayer(x, w0upd), w1upd, y, 1)

  val fusedTraining = fuseOps //clear after train, constant call below is dying - working now, but still needs an input due to ORT weirdness
//  println(fusedTraining)
  val onnxBytesTraining = fusedTraining.toByteArray
  val fusedModelTraining = new ORTModelBackend(onnxBytesTraining)
  val beforeTrainFused = System.nanoTime
  val trainOut = fusedModelTraining.fullModel[Float, TT, TD, TENKXTENK](x,w0,w1,y, Tensor(Array(10000l, 1000l), 2 #: SNil), Tensor(Array(10000l, 1000l), 2 #: SNil), Tensor(Array(10000l, 10000l), 2 #: SNil),
                                                                                   Tensor(Array(10000l, 10000l), 2 #: SNil), Tensor(Array(10000l, 10000l), 2 #: SNil),Tensor(Array(10000l, 10000l), 2 #: SNil), trans) //orderinnggg
  val afterTrainFused = System.nanoTime
  //x,w0,w1,y, someOnes, moreLrs)


  println("train time , fused " + (afterTrainFused-beforeTrainFused))
  println("trained out " + trainOut.data(0))
//  val clonedInput = Tensor(x.data.clone, "TensorTypeDenotation", axisLabels, mat10kX10k).constant()
  val clonedResult:Tensor[Float, (TT,TD, TENKXTENK)] = Tensor(w0upd.data.clone, "TensorTypeDenotation", axisLabels, mat10kX10k).constant()
  val clonedResult1:Tensor[Float, (TT,TD, TENKXONEK)] = Tensor(w1upd.data.clone, "TensorTypeDenotation", axisLabels, mat10kX1k).constant()
//  val out1 = forwardLayer(x, clonedResult)
//  val output:Tensor[Float, (TT,TD, TENKXTENK)] = forwardLayer(out1, clonedResult1)
  //println("cloned pred " + output.data(0))

//NEWWWW




//


val fusedModelProto = fuseOps
//println(fusedModelProto)
val onnxBytes = fusedModelProto.toByteArray
val fusedModel = new ORTModelBackend(onnxBytes)
val fusedBefore = System.nanoTime

for (j <- 0 until 5) { val out = fusedModel.fullModel[Float, TT, TD, TENKXONEK]((Tensor(Array(10000l, 10000l), 2 #: SNil), Tensor(Array(10000l, 1000l), 2 #: SNil), x))
                                 //println("fused pred: " + out.data(0))
                                 } //(Tuple5(x,w0,y, someOnes, moreLrs)) }
val fusedAfter = System.nanoTime
println(fusedAfter - fusedBefore)
}

//For dex/ futhark comparison
/*
def pairwiseL1[Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, N <: Dimension, D <: Dimension](tens:Tensor[Float, (Tt, Td, N #: D #: SNil)]): Tensor[Float, (Tt, Td, N #: N #: SNil)] = {
((tens.transpose - tens.unsqueeze(Array(2))).abs).reduceSum(axis=1)

}

//def pairwiseL1[SomeTT, SomeTD, N #: D #: SNil](tens:Tensor[Float, (SomeTT, SomeTD, N #: D #: SNil)]): Tensor[Float, (SomeTT, SomeTD, N #: N #: SNil)] =
}
*/

