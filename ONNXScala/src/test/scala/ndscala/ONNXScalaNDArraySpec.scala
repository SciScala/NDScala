package org.sciscala.ndscala

import scala.collection.immutable.ArraySeq

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ONNXScalaNDArraySpec extends AnyFlatSpec with Matchers {

  implicit val ndarrayOps: NDArrayOps[NDArray] = new ONNXScalaOps()

  import NDArrayOps.ops._
 
  "NDArray" should "zero" in {
    ndarrayOps.zeros[Int](ArraySeq(4)) shouldEqual NDArray(ArraySeq(0,0,0,0), ArraySeq(4))
  }

  "NDArray" should "one" in {
    ndarrayOps.ones[Int](ArraySeq(4)) shouldEqual NDArray(ArraySeq(1,1,1,1), ArraySeq(4))
  }

  "NDArray" should "fill" in {
    ndarrayOps.full(ArraySeq(4), 5) shouldEqual NDArray(ArraySeq(5,5,5,5), ArraySeq(4))
  }

  "NDArray" should "add" in {
    val arr: NDArray[Int] = NDArray(ArraySeq(42, 84), ArraySeq(1,2))
    arr + arr shouldEqual NDArray(ArraySeq(84, 168), ArraySeq(1,2))
  }

  "NDArray" should "subtract" in {
    val arr: NDArray[Int] = NDArray(ArraySeq(42, 84), ArraySeq(1,2))
    arr - arr shouldEqual NDArray(ArraySeq(0, 0), ArraySeq(1,2))
  }

  "NDArray" should "multiply" in {
    val arr: NDArray[Int] = NDArray(ArraySeq(42, 84), ArraySeq(1,2))
    arr * arr shouldEqual NDArray(ArraySeq(1764, 7056), ArraySeq(1,2))
  }

  "NDArray" should "divide" in {
    val arr: NDArray[Int] = NDArray(ArraySeq(42, 84), ArraySeq(1,2))
    arr / arr shouldEqual NDArray(ArraySeq(1, 1), ArraySeq(1,2))
  }

  "NDArray" should "equal" in {
    val arr: NDArray[Int] = NDArray(ArraySeq(42, 84), ArraySeq(1,2))
    arr === arr shouldEqual NDArray(ArraySeq(true, true), ArraySeq(1,2))
  }

  "NDArray" should "not equal" in {
    val arr: NDArray[Int] = NDArray(ArraySeq(42, 84), ArraySeq(1,2))
    (arr !== arr) shouldEqual NDArray(ArraySeq(false, false), ArraySeq(1,2))
  }

  "NDArray" should "reshape" in {
    val arr: NDArray[Int] = NDArray(ArraySeq(42, 84), ArraySeq(1,2))
    (arr reshape ArraySeq(2,1)) shouldEqual NDArray(ArraySeq(42, 84), ArraySeq(2,1))
  }

  "NDArray" should "transpose" in {
    val arr: NDArray[Int] = NDArray(ArraySeq(1, 2, 3, 4), ArraySeq(2,2))
    (arr.transpose) shouldEqual NDArray(ArraySeq(1, 3, 2, 4), ArraySeq(2,2))
  }

  "NDArray" should "transpose with axes" in {
    val arr: NDArray[Int] = NDArray(ArraySeq(1, 2, 3, 4), ArraySeq(2,2))
    (arr transpose ArraySeq(1,0)) shouldEqual NDArray(ArraySeq(1, 3, 2, 4), ArraySeq(2,2))
  }

  "NDArray" should "round" in {
    val arr: NDArray[Double] = NDArray(ArraySeq(41.7, 84.3), ArraySeq(1,2))
    arr.round shouldEqual NDArray(ArraySeq(42, 84), ArraySeq(1,2))
  }

  "NDArray" should "slice" in {
    val arr: NDArray[Int] = NDArray(ArraySeq(1, 2, 3, 4), ArraySeq(4))
    arr.slice(1,3) shouldEqual NDArray(ArraySeq(2, 3), ArraySeq(2))
  }

  "NDArray" should "squeeze" in {
    val arr: NDArray[Int] = NDArray(ArraySeq(42, 84), ArraySeq(1,2))
    (arr squeeze ArraySeq(0)) shouldEqual NDArray(ArraySeq(42, 84), ArraySeq(2))
  }

  "NDArray" should "rank" in {
    val arr: NDArray[Int] = NDArray(ArraySeq(42, 84), ArraySeq(1,2))
    arr.rank shouldEqual 2
  }

  "NDArray" should "clip" in {
    val arr: NDArray[Double] = NDArray(ArraySeq(41.7, 84.5), ArraySeq(1,2))
    arr.clip(50.0, 90.0) shouldEqual NDArray(ArraySeq(50.0, 84.5), ArraySeq(1,2))
  }

  "NDArray" should "unary subtract" in {
    val arr: NDArray[Int] = NDArray(ArraySeq(42, 84), ArraySeq(1,2))
    -arr shouldEqual NDArray(ArraySeq(-42, -84), ArraySeq(1,2))
  }

  "NDArray" should "abs" in {
    val arr: NDArray[Int] = NDArray(ArraySeq(-42, 84), ArraySeq(1,2))
    arr.abs shouldEqual NDArray(ArraySeq(42, 84), ArraySeq(1,2))
  }

  "NDArray" should "ceil" in {
    val arr: NDArray[Float] = NDArray(ArraySeq(-1.5f, 1.2f), ArraySeq(1,2))
    arr.ceil shouldEqual NDArray(ArraySeq(-1.0f, 2.0f), ArraySeq(1,2))
  }

  "NDArray" should "floor" in {
    val arr: NDArray[Float] = NDArray(ArraySeq(-1.5f, 1.2f), ArraySeq(1,2))
    arr.floor shouldEqual NDArray(ArraySeq(-2.0f, 1.0f), ArraySeq(1,2))
  }

  "NDArray" should "log" in {
    val arr: NDArray[Float] = NDArray(ArraySeq(1.0f, 10.0f), ArraySeq(1,2))
    arr.log shouldEqual NDArray(ArraySeq(0.0f, 2.30258512f), ArraySeq(1,2))
  }

  "NDArray" should "exp" in {
    val arr: NDArray[Double] = NDArray(ArraySeq(-1.0, 0.0, 1.0), ArraySeq(1,3))
    arr.exp shouldEqual NDArray(ArraySeq(0.3678794411714423, 1.0, 2.718281828459045), ArraySeq(1,3))
  }

  "NDArray" should "sqrt" in {
    val arr: NDArray[Double] = NDArray(ArraySeq(1.0, 4.0, 9.0), ArraySeq(1,3))
    arr.sqrt shouldEqual NDArray(ArraySeq(1.0, 2.0, 3.0), ArraySeq(1,3))
  }

  "NDArray" should "cos" in {
    val arr: NDArray[Float] = NDArray(ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    arr.cos shouldEqual NDArray(ArraySeq(0.5403023f, 1.0f, 0.5403023f), ArraySeq(1,3))
  }

  "NDArray" should "sin" in {
    val arr: NDArray[Float] = NDArray(ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    arr.sin shouldEqual NDArray(ArraySeq(-0.84147096f, 0.0f, 0.84147096f), ArraySeq(1,3))
  }

  "NDArray" should "tan" in {
    val arr: NDArray[Float] = NDArray(ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    arr.tan shouldEqual NDArray(ArraySeq(-1.5574077f, 0.0f, 1.5574077f), ArraySeq(1,3))
  }

  "NDArray" should "tanh" in {
    val arr: NDArray[Float] = NDArray(ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    arr.tanh shouldEqual NDArray(ArraySeq(-0.7615942f, 0.0f, 0.7615942f), ArraySeq(1,3))
  }

  "NDArray" should "acos" in {
    val arr: NDArray[Float] = NDArray(ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    arr.acos shouldEqual NDArray(ArraySeq(3.1415927f, 1.5707964f, 0.0f), ArraySeq(1,3))
  }

  "NDArray" should "asin" in {
    val arr: NDArray[Float] = NDArray(ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    arr.asin shouldEqual NDArray(ArraySeq(-1.5707964f, 0.0f, 1.5707964f), ArraySeq(1,3))
  }

  "NDArray" should "atan" in {
    val arr: NDArray[Float] = NDArray(ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    arr.atan shouldEqual NDArray(ArraySeq(-0.7853982f, 0.0f, 0.7853982f), ArraySeq(1,3))
  }

  "NDArray" should "atanh" in {
    val arr: NDArray[Float] = NDArray(ArraySeq(-0.5f, 0.0f, 0.5f), ArraySeq(1,3))
    arr.atanh shouldEqual NDArray(ArraySeq(-0.54930615f, 0.0f, 0.54930615f), ArraySeq(1,3))
  }

  "NDArray" should "pow" in {
    val arr: NDArray[Double] = NDArray(ArraySeq(42.0, 84.0), ArraySeq(1,2))
    (arr ** NDArray(ArraySeq(2.0), ArraySeq(1))) shouldEqual NDArray(ArraySeq(1764.0, 7056.0), ArraySeq(1,2))
  }

  "NDArray" should "mod" in {
    val arr: NDArray[Int] = NDArray(ArraySeq(42, 84), ArraySeq(1,2))
    (arr % NDArray(ArraySeq(40), ArraySeq(1))) shouldEqual NDArray(ArraySeq(2,4), ArraySeq(1,2))
  }

  "NDArray" should "gt" in {
    val arr: NDArray[Int] = NDArray(ArraySeq(42, 84), ArraySeq(1,2))
    (arr > NDArray(ArraySeq(42, 80), ArraySeq(1, 2))) shouldEqual NDArray(ArraySeq(false, true), ArraySeq(1,2))
  }

  "NDArray" should "gte" in {
    val arr: NDArray[Long] = NDArray(ArraySeq(42l, 84l), ArraySeq(1,2))
    (arr >= NDArray(ArraySeq(42l, 80l), ArraySeq(1, 2))) shouldEqual NDArray(ArraySeq(true, true), ArraySeq(1,2))
  }

  "NDArray" should "lt" in {
    val arr: NDArray[Int] = NDArray(ArraySeq(42, 84), ArraySeq(1,2))
    (arr < NDArray(ArraySeq(42, 80), ArraySeq(1, 2))) shouldEqual NDArray(ArraySeq(false, false), ArraySeq(1,2))
  }

  "NDArray" should "lte" in {
    val arr: NDArray[Long] = NDArray(ArraySeq(42l, 84l), ArraySeq(1,2))
    (arr <= NDArray(ArraySeq(42l, 80l), ArraySeq(1, 2))) shouldEqual NDArray(ArraySeq(true, false), ArraySeq(1,2))
  }

  "NDArray" should "max" in {
    val arr: NDArray[Float] = NDArray(ArraySeq(42.0f, 84.0f), ArraySeq(1,2))
    val other: NDArray[Float] = NDArray(ArraySeq(50.0f, 80.0f), ArraySeq(1,2))
    (arr max other) shouldEqual NDArray(ArraySeq(50.0f, 84.0f), ArraySeq(1,2))
  }

  "NDArray" should "min" in {
    val arr: NDArray[Double] = NDArray(ArraySeq(42.0, 84.0), ArraySeq(1,2))
    val other: NDArray[Double] = NDArray(ArraySeq(50.0, 80.0), ArraySeq(1,2))
    (arr min other) shouldEqual NDArray(ArraySeq(42.0, 80.0), ArraySeq(1,2))
  }

  "NDArray" should "dot" in {
    val arr: NDArray[Double] = NDArray(ArraySeq(42.0, 84.0), ArraySeq(1,2))
    val other: NDArray[Double] = NDArray(ArraySeq(42.0, 84.0), ArraySeq(2,1))
    (arr dot other) shouldEqual NDArray(ArraySeq(8820.0), ArraySeq(1,1))
  }
}
