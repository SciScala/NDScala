package org.sciscala.ndscala

import scala.collection.immutable.ArraySeq

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import ai.djl._
import ai.djl.ndarray._
import ai.djl.ndarray.types.Shape
import ai.djl.ndarray.index.NDIndex

//TODO: consistent naming 
class DJLNDArraySpec extends AnyFlatSpec with Matchers {

  import DJLOps._

  import NDArrayOps.ops._
 
  implicit val ndarrayOps: NDArrayOps[DJLNDArray] = new DJLOps()


  /*
  "DJLNDArray" should "zero" in {
    fromDJLNDArray(ndarrayOps.zeros[Int](ArraySeq(4))) shouldEqual (ArraySeq(0,0,0,0), ArraySeq(4))
  }

  "DJLNDArray" should "one" in {
    fromDJLNDArray(ndarrayOps.ones[Int](ArraySeq(4))) shouldEqual (ArraySeq(1,1,1,1), ArraySeq(4))
  }

  "DJLNDArray" should "fill" in {
    fromDJLNDArray(ndarrayOps.full(ArraySeq(4), 5)) shouldEqual (ArraySeq(5,5,5,5), ArraySeq(4))
  }
*/
  "DJLNDArray" should "add" in {
    val arr: DJLNDArray[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromDJLNDArray(arr + arr) shouldEqual (ArraySeq(84, 168), ArraySeq(1,2))
  }

  "DJLNDArray" should "subtract" in {
    val arr: DJLNDArray[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromDJLNDArray(arr - arr) shouldEqual (ArraySeq(0, 0), ArraySeq(1,2))
  }

  "DJLNDArray" should "multiply" in {
    val arr: DJLNDArray[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromDJLNDArray(arr * arr) shouldEqual (ArraySeq(1764, 7056), ArraySeq(1,2))
  }

  "DJLNDArray" should "divide" in {
    val arr: DJLNDArray[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromDJLNDArray(arr / arr) shouldEqual (ArraySeq(1, 1), ArraySeq(1,2))
  }

  "DJLNDArray" should "equal" in {
    val arr: DJLNDArray[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromDJLNDArray(arr === arr) shouldEqual (ArraySeq(true, true), ArraySeq(1,2))
  }

  "DJLNDArray" should "not equal" in {
    val arr: DJLNDArray[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromDJLNDArray(arr !== arr) shouldEqual (ArraySeq(false, false), ArraySeq(1,2))
  }

  "DJLNDArray" should "reshape" in {
    val arr: DJLNDArray[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromDJLNDArray(arr reshape ArraySeq(2,1)) shouldEqual (ArraySeq(42, 84), ArraySeq(2,1))
  }

  "DJLNDArray" should "transpose" in {
    val arr: DJLNDArray[Int] = (ArraySeq(1, 2, 3, 4), ArraySeq(2,2))
    fromDJLNDArray(arr.transpose()) shouldEqual (ArraySeq(1, 3, 2, 4), ArraySeq(2,2))
  }

  "DJLNDArray" should "transpose with axes" in {
    val arr: DJLNDArray[Int] = (ArraySeq(1, 2, 3, 4), ArraySeq(2,2))
    fromDJLNDArray(arr transpose ArraySeq(1,0)) shouldEqual (ArraySeq(1, 3, 2, 4), ArraySeq(2,2))
  }

  "DJLNDArray" should "round" in {
    val arr: DJLNDArray[Double] = (ArraySeq(41.7, 84.3), ArraySeq(1,2))
    fromDJLNDArray(arr.round) shouldEqual (ArraySeq(42, 84), ArraySeq(1,2))
  }

  "DJLNDArray" should "slice" in {
    val arr: DJLNDArray[Int] = (ArraySeq(1, 2, 3, 4), ArraySeq(4))
    fromDJLNDArray(ndarrayOps.slice(arr,1,3)) shouldEqual (ArraySeq(2, 3), ArraySeq(2))
  }

  "DJLNDArray" should "squeeze" in {
    val arr: DJLNDArray[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromDJLNDArray(arr squeeze ArraySeq(0)) shouldEqual (ArraySeq(42, 84), ArraySeq(2))
  }

  "DJLNDArray" should "rank" in {
    val arr: DJLNDArray[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    arr.rank shouldEqual 2
  }

  /*
  "DJLNDArray" should "clip" in {
    val arr: DJLNDArray[Double] = (ArraySeq(41.7, 84.5), ArraySeq(1,2))
    fromDJLNDArray(arr.clip(50.0, 90.0)) shouldEqual (ArraySeq(50.0, 84.5), ArraySeq(1,2))
  }
*/
  "DJLNDArray" should "unary subtract" in {
    val arr: DJLNDArray[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromDJLNDArray(-arr) shouldEqual (ArraySeq(-42, -84), ArraySeq(1,2))
  }

  "DJLNDArray" should "abs" in {
    val arr: DJLNDArray[Int] = (ArraySeq(-42, 84), ArraySeq(1,2))
    fromDJLNDArray(arr.abs) shouldEqual (ArraySeq(42, 84), ArraySeq(1,2))
  }

  "DJLNDArray" should "ceil" in {
    val arr: DJLNDArray[Float] = (ArraySeq(-1.5f, 1.2f), ArraySeq(1,2))
    fromDJLNDArray(arr.ceil) shouldEqual (ArraySeq(-1.0f, 2.0f), ArraySeq(1,2))
  }

  "DJLNDArray" should "floor" in {
    val arr: DJLNDArray[Float] = (ArraySeq(-1.5f, 1.2f), ArraySeq(1,2))
    fromDJLNDArray(arr.floor) shouldEqual (ArraySeq(-2.0f, 1.0f), ArraySeq(1,2))
  }

  "DJLNDArray" should "log" in {
    val arr: DJLNDArray[Float] = (ArraySeq(1.0f, 10.0f), ArraySeq(1,2))
    fromDJLNDArray(arr.log) shouldEqual (ArraySeq(0.0f, 2.30258512f), ArraySeq(1,2))
  }

  "DJLNDArray" should "exp" in {
    val arr: DJLNDArray[Double] = (ArraySeq(-1.0, 0.0, 1.0), ArraySeq(1,3))
    fromDJLNDArray(arr.exp) shouldEqual (ArraySeq(0.36787944117144233, 1.0, 2.718281828459045), ArraySeq(1,3))
  }

  "DJLNDArray" should "sqrt" in {
    val arr: DJLNDArray[Double] = (ArraySeq(1.0, 4.0, 9.0), ArraySeq(1,3))
    fromDJLNDArray(arr.sqrt) shouldEqual (ArraySeq(1.0, 2.0, 3.0), ArraySeq(1,3))
  }

  "DJLNDArray" should "cos" in {
    val arr: DJLNDArray[Float] = (ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    fromDJLNDArray(arr.cos) shouldEqual (ArraySeq(0.5403023f, 1.0f, 0.5403023f), ArraySeq(1,3))
  }

  "DJLNDArray" should "sin" in {
    val arr: DJLNDArray[Float] = (ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    fromDJLNDArray(arr.sin) shouldEqual (ArraySeq(-0.84147096f, 0.0f, 0.84147096f), ArraySeq(1,3))
  }

  "DJLNDArray" should "tan" in {
    val arr: DJLNDArray[Float] = (ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    fromDJLNDArray(arr.tan) shouldEqual (ArraySeq(-1.5574077f, 0.0f, 1.5574077f), ArraySeq(1,3))
  }

  "DJLNDArray" should "tanh" in {
    val arr: DJLNDArray[Float] = (ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    fromDJLNDArray(arr.tanh) shouldEqual (ArraySeq(-0.7615942f, 0.0f, 0.7615942f), ArraySeq(1,3))
  }

  "DJLNDArray" should "acos" in {
    val arr: DJLNDArray[Float] = (ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    fromDJLNDArray(arr.acos) shouldEqual (ArraySeq(3.1415927f, 1.5707964f, 0.0f), ArraySeq(1,3))
  }

  "DJLNDArray" should "asin" in {
    val arr: DJLNDArray[Float] = (ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    fromDJLNDArray(arr.asin) shouldEqual (ArraySeq(-1.5707964f, 0.0f, 1.5707964f), ArraySeq(1,3))
  }

  "DJLNDArray" should "atan" in {
    val arr: DJLNDArray[Float] = (ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    fromDJLNDArray(arr.atan) shouldEqual (ArraySeq(-0.7853982f, 0.0f, 0.7853982f), ArraySeq(1,3))
  }

  /*
  "DJLNDArray" should "atanh" in {
    val arr: DJLNDArray[Float] = (ArraySeq(-0.5f, 0.0f, 0.5f), ArraySeq(1,3))
    fromDJLNDArray(arr.atanh) shouldEqual (ArraySeq(-0.54930615f, 0.0f, 0.54930615f), ArraySeq(1,3))
  }
*/
  "DJLNDArray" should "pow" in {
    val arr: DJLNDArray[Double] = (ArraySeq(42.0, 84.0), ArraySeq(1,2))
    fromDJLNDArray(arr ** (ArraySeq(2.0), ArraySeq(1))) shouldEqual (ArraySeq(1764.0, 7056.0), ArraySeq(1,2))
  }

  "DJLNDArray" should "mod" in {
    val arr: DJLNDArray[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromDJLNDArray(arr % (ArraySeq(40), ArraySeq(1))) shouldEqual (ArraySeq(2,4), ArraySeq(1,2))
  }

  "DJLNDArray" should "gt" in {
    val arr: DJLNDArray[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromDJLNDArray(arr > (ArraySeq(42, 80), ArraySeq(1, 2))) shouldEqual (ArraySeq(false, true), ArraySeq(1,2))
  }

  "DJLNDArray" should "gte" in {
    val arr: DJLNDArray[Long] = (ArraySeq(42l, 84l), ArraySeq(1,2))
    fromDJLNDArray(arr >= (ArraySeq(42l, 80l), ArraySeq(1, 2))) shouldEqual (ArraySeq(true, true), ArraySeq(1,2))
  }

  "DJLNDArray" should "lt" in {
    val arr: DJLNDArray[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromDJLNDArray(arr < (ArraySeq(42, 80), ArraySeq(1, 2))) shouldEqual (ArraySeq(false, false), ArraySeq(1,2))
  }

  "DJLNDArray" should "lte" in {
    val arr: DJLNDArray[Long] = (ArraySeq(42l, 84l), ArraySeq(1,2))
    fromDJLNDArray(arr <= (ArraySeq(42l, 80l), ArraySeq(1, 2))) shouldEqual (ArraySeq(true, false), ArraySeq(1,2))
  }

  "DJLNDArray" should "max" in {
    val arr: DJLNDArray[Float] = (ArraySeq(42.0f, 84.0f), ArraySeq(1,2))
    val other: DJLNDArray[Float] = (ArraySeq(50.0f, 80.0f), ArraySeq(1,2))
    fromDJLNDArray(arr maximum other) shouldEqual (ArraySeq(50.0f, 84.0f), ArraySeq(1,2))
  }

  "DJLNDArray" should "min" in {
    val arr: DJLNDArray[Double] = (ArraySeq(42.0, 84.0), ArraySeq(1,2))
    val other: DJLNDArray[Double] = (ArraySeq(50.0, 80.0), ArraySeq(1,2))
    fromDJLNDArray(arr minimum other) shouldEqual (ArraySeq(42.0, 80.0), ArraySeq(1,2))
  }

  "DJLNDArray" should "dot" in {
    val arr: DJLNDArray[Double] = (ArraySeq(42.0, 84.0), ArraySeq(1,2))
    val other: DJLNDArray[Double] = (ArraySeq(42.0, 84.0), ArraySeq(2,1))
    fromDJLNDArray(arr dot other) shouldEqual (ArraySeq(8820.0), ArraySeq(1,1))
  }
}
