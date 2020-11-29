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
import DJLOps._
import io.kjaer.compiletime._
import org.emergentorder.compiletime._
import ai.djl._
import ai.djl.ndarray._
import DJLOps.toDJLNDArray
import spire.math.Numeric
import spire.implicits._

//Weird bug happening where allocation hangs forever
val thisRandom = new Random(42)
val iters = 5
val lr = 0.000001f

type TT = "TensorTypeDenotation"
type TD = "TensorShapeDenotation" ##: TSNil

type TENKXTENK = 10000 #: 10000 #: SNil

type TENKXONE = 10000 #: 1000 #: SNil


val lrs:DJLNDArray[Float, (TT, TD, TENKXONE)] = Tensor(Array.fill(10000000)(lr),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 10000 #: 1000 #: SNil)
val moreLrs:DJLNDArray[Float, (TT,TD, TENKXTENK)] = Tensor(Array.fill(100000000)(lr),"TensorTypeDenotation","TensorShapeDenotation" ##: TSNil, 10000 #: 10000 #: SNil)

val ones:DJLNDArray[Float, (TT, TD, TENKXONE)] = Tensor(Array.fill(10000000)(1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 10000 #: 1000 #: SNil)
val moreOnes:DJLNDArray[Float, (TT, TD, TENKXTENK)] = Tensor(Array.fill(100000000)(1.0f),"TensorTypeDenotation","TensorShapeDenotation" ##: TSNil, 10000 #: 10000 #: SNil)
//For scaling

val some10ks:DJLNDArray[Float, (TT, TD, TENKXONE)] = Tensor(Array.fill(10000000)(10000.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 10000 #: 1000 #: SNil)
val more10ks:DJLNDArray[Float, (TT, TD, TENKXTENK)] = Tensor(Array.fill(100000000)(10000.0f),"TensorTypeDenotation","TensorShapeDenotation" ##: TSNil, 10000 #: 10000 #: SNil)

//TODO: check without type annotations when it's working
val arrX:Array[Float] = (Array.fill(100000000)(thisRandom.nextFloat)).map(_.toFloat)
val arrY:Array[Float] = (Array.fill(10000000)(thisRandom.nextFloat)).map(_.toFloat)
val arrW0:Array[Float] = (Array.fill(100000000)(thisRandom.nextFloat)).map(_.toFloat)
val arrW1:Array[Float] = (Array.fill(10000000)(thisRandom.nextFloat)).map(_.toFloat)

val y:DJLNDArray[Float, (TT, TD, TENKXONE)] = Tensor(arrY,"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 10000 #: 1000 #: SNil)
val x:DJLNDArray[Float, (TT, TD, TENKXTENK)] = Tensor(arrX,"TensorTypeDenotation","TensorShapeDenotation" ##: TSNil, 10000 #: 10000 #: SNil)
//TODO: call recursively
val origW0:DJLNDArray[Float, (TT, TD, TENKXTENK)] = Tensor(arrW0,"TensorTypeDenotation","TensorShapeDenotation" ##: TSNil, 10000 #: 10000 #: SNil)
var w0:DJLNDArray[Float, (TT, TD, TENKXTENK)] = ((origW0:DJLNDArray[Float, (TT, TD, TENKXTENK)]) - moreOnes) / more10ks


val origW1:DJLNDArray[Float, (TT, TD, TENKXONE)] = Tensor(arrW1,"TensorTypeDenotation","TensorShapeDenotation" ##: TSNil, 10000 #: 1000 #: SNil)
var w1:DJLNDArray[Float, (TT, TD, TENKXONE)] = ((origW1:DJLNDArray[Float, (TT, TD, TENKXONE)]) - ones ) /some10ks


//Disabling broadcasting for ones may have slowed things down
//also lr multiplications
def train = {
//     val future = async {
      val l1:DJLNDArray[Float, (TT, TD, TENKXTENK)] =  (x.matmul[Float, 10000, 10000, 10000, TT,TD, TENKXTENK, TT, TD,  TENKXTENK](w0)).sigmoid() // one / ((-(x dot w0)).exp() + one)
      val l2:DJLNDArray[Float, (TT, TD, TENKXONE)] = (l1.matmul[Float, 10000, 10000, 1000, TT,TD, TENKXTENK, TT, TD,  TENKXONE](w1)).sigmoid() // one / ((-(l1 dot w1)).exp() + one)

      val error = (y:DJLNDArray[Float, (TT, TD, TENKXONE)]) - l2
//      println("error: " + error.data.sum)
      val l2Delta = (error) * (l2 * ((ones:DJLNDArray[Float, (TT, TD, TENKXONE)]) - (l2:DJLNDArray[Float, (TT, TD, TENKXONE)])))
      val l1Delta =  (l2Delta.matmul[Float, 10000,1000,10000, TT,TD, TENKXONE, TT, TD,  1000 #: 10000 #: SNil](w1.transpose)) * ((l1:DJLNDArray[Float, (TT, TD, TENKXTENK)]) * ((moreOnes - (l1:DJLNDArray[Float, (TT, TD, TENKXTENK)]))))

      //Simulate in-place += op here 
      
      w0 = w0 + (((x.transpose).matmul[Float, 10000,10000,10000, TT,TD, TENKXTENK, TT, TD,  TENKXTENK](l1Delta))) //*moreLrs)
      w1 = w1 + (((l1.transpose).matmul[Float, 10000,10000,1000, TT,TD, TENKXTENK, TT, TD,  TENKXONE](l2Delta))) //*lrs)
   
      println(w0)
      println(w1)
}
val before = System.nanoTime; for (j <- (0l until iters)) {
  val result = train
}; val after = System.nanoTime

println(after-before)

}
