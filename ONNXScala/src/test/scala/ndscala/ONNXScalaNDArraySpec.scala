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
