package org.sciscala.ndscala

import scala.collection.immutable.ArraySeq

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.emergentorder.onnx.Tensor

class ONNXScalaTensorSpec extends AnyFlatSpec with Matchers {

  implicit val ndarrayOps: NDArrayOps[Tensor] = new ONNXScalaOps()

  import NDArrayOps.ops._

  import ONNXScalaOps._

  "Tensor" should "zero" in {
    fromTensor(ndarrayOps.zeros[Int](ArraySeq(4))) shouldEqual (ArraySeq(0,0,0,0), ArraySeq(4))
  }

  "Tensor" should "one" in {
    fromTensor(ndarrayOps.ones[Int](ArraySeq(4))) shouldEqual (ArraySeq(1,1,1,1), ArraySeq(4))
  }

  "Tensor" should "fill" in {
    fromTensor(ndarrayOps.full(ArraySeq(4), 5)) shouldEqual (ArraySeq(5,5,5,5), ArraySeq(4))
  }

  "Tensor" should "add" in {
    val arr: Tensor[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromTensor(arr + arr) shouldEqual (ArraySeq(84, 168), ArraySeq(1,2))
  }

  "Tensor" should "subtract" in {
    val arr: Tensor[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromTensor(arr - arr) shouldEqual (ArraySeq(0, 0), ArraySeq(1,2))
  }

  "Tensor" should "multiply" in {
    val arr: Tensor[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromTensor(arr * arr) shouldEqual (ArraySeq(1764, 7056), ArraySeq(1,2))
  }

  "Tensor" should "divide" in {
    val arr: Tensor[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromTensor(arr / arr) shouldEqual (ArraySeq(1, 1), ArraySeq(1,2))
  }

  "Tensor" should "equal" in {
    val arr: Tensor[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromTensor(arr === arr) shouldEqual (ArraySeq(true, true), ArraySeq(1,2))
  }

  "Tensor" should "not equal" in {
    val arr: Tensor[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromTensor(arr !== arr) shouldEqual (ArraySeq(false, false), ArraySeq(1,2))
  }

  "Tensor" should "reshape" in {
    val arr: Tensor[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromTensor(arr reshape ArraySeq(2,1)) shouldEqual (ArraySeq(42, 84), ArraySeq(2,1))
  }

  "Tensor" should "transpose" in {
    val arr: Tensor[Int] = (ArraySeq(1, 2, 3, 4), ArraySeq(2,2))
    fromTensor(arr.transpose) shouldEqual (ArraySeq(1, 3, 2, 4), ArraySeq(2,2))
  }

  "Tensor" should "transpose with axes" in {
    val arr: Tensor[Int] = (ArraySeq(1, 2, 3, 4), ArraySeq(2,2))
    fromTensor(arr transpose ArraySeq(1,0)) shouldEqual (ArraySeq(1, 3, 2, 4), ArraySeq(2,2))
  }

  "Tensor" should "round" in {
    val arr: Tensor[Double] = (ArraySeq(41.7, 84.3), ArraySeq(1,2))
    fromTensor(arr.round) shouldEqual (ArraySeq(42, 84), ArraySeq(1,2))
  }

  "Tensor" should "slice" in {
    val arr: Tensor[Int] = (ArraySeq(1, 2, 3, 4), ArraySeq(4))
    fromTensor(arr.slice(1,3)) shouldEqual (ArraySeq(2, 3), ArraySeq(2))
  }

  "Tensor" should "squeeze" in {
    val arr: Tensor[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromTensor(arr squeeze ArraySeq(0)) shouldEqual (ArraySeq(42, 84), ArraySeq(2))
  }

  "Tensor" should "rank" in {
    val arr: Tensor[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    arr.rank shouldEqual 2
  }

  "Tensor" should "clip" in {
    val arr: Tensor[Double] = (ArraySeq(41.7, 84.5), ArraySeq(1,2))
    fromTensor(arr.clip(50.0, 90.0)) shouldEqual (ArraySeq(50.0, 84.5), ArraySeq(1,2))
  }

  "Tensor" should "unary subtract" in {
    val arr: Tensor[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromTensor(-arr) shouldEqual (ArraySeq(-42, -84), ArraySeq(1,2))
  }

  "Tensor" should "abs" in {
    val arr: Tensor[Int] = (ArraySeq(-42, 84), ArraySeq(1,2))
    fromTensor(arr.abs) shouldEqual (ArraySeq(42, 84), ArraySeq(1,2))
  }

  "Tensor" should "ceil" in {
    val arr: Tensor[Float] = (ArraySeq(-1.5f, 1.2f), ArraySeq(1,2))
    fromTensor(arr.ceil) shouldEqual (ArraySeq(-1.0f, 2.0f), ArraySeq(1,2))
  }

  "Tensor" should "floor" in {
    val arr: Tensor[Float] = (ArraySeq(-1.5f, 1.2f), ArraySeq(1,2))
    fromTensor(arr.floor) shouldEqual (ArraySeq(-2.0f, 1.0f), ArraySeq(1,2))
  }

  "Tensor" should "log" in {
    val arr: Tensor[Float] = (ArraySeq(1.0f, 10.0f), ArraySeq(1,2))
    fromTensor(arr.log) shouldEqual (ArraySeq(0.0f, 2.30258512f), ArraySeq(1,2))
  }

  "Tensor" should "exp" in {
    val arr: Tensor[Double] = (ArraySeq(-1.0, 0.0, 1.0), ArraySeq(1,3))
    fromTensor(arr.exp) shouldEqual (ArraySeq(0.3678794411714423, 1.0, 2.718281828459045), ArraySeq(1,3))
  }

  "Tensor" should "sqrt" in {
    val arr: Tensor[Double] = (ArraySeq(1.0, 4.0, 9.0), ArraySeq(1,3))
    fromTensor(arr.sqrt) shouldEqual (ArraySeq(1.0, 2.0, 3.0), ArraySeq(1,3))
  }

  "Tensor" should "cos" in {
    val arr: Tensor[Float] = (ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    fromTensor(arr.cos) shouldEqual (ArraySeq(0.5403023f, 1.0f, 0.5403023f), ArraySeq(1,3))
  }

  "Tensor" should "sin" in {
    val arr: Tensor[Float] = (ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    fromTensor(arr.sin) shouldEqual (ArraySeq(-0.84147096f, 0.0f, 0.84147096f), ArraySeq(1,3))
  }

  "Tensor" should "tan" in {
    val arr: Tensor[Float] = (ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    fromTensor(arr.tan) shouldEqual (ArraySeq(-1.5574077f, 0.0f, 1.5574077f), ArraySeq(1,3))
  }

  "Tensor" should "tanh" in {
    val arr: Tensor[Float] = (ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    fromTensor(arr.tanh) shouldEqual (ArraySeq(-0.7615942f, 0.0f, 0.7615942f), ArraySeq(1,3))
  }

  "Tensor" should "acos" in {
    val arr: Tensor[Float] = (ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    fromTensor(arr.acos) shouldEqual (ArraySeq(3.1415927f, 1.5707964f, 0.0f), ArraySeq(1,3))
  }

  "Tensor" should "asin" in {
    val arr: Tensor[Float] = (ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    fromTensor(arr.asin) shouldEqual (ArraySeq(-1.5707964f, 0.0f, 1.5707964f), ArraySeq(1,3))
  }

  "Tensor" should "atan" in {
    val arr: Tensor[Float] = (ArraySeq(-1.0f, 0.0f, 1.0f), ArraySeq(1,3))
    fromTensor(arr.atan) shouldEqual (ArraySeq(-0.7853982f, 0.0f, 0.7853982f), ArraySeq(1,3))
  }

  /*
  "Tensor" should "atanh" in {
    val arr: Tensor[Float] = (ArraySeq(-0.5f, 0.0f, 0.5f), ArraySeq(1,3))
    fromTensor(arr.atanh) shouldEqual (ArraySeq(-0.54930615f, 0.0f, 0.54930615f), ArraySeq(1,3))
  }
*/
  "Tensor" should "pow" in {
    val arr: Tensor[Double] = (ArraySeq(42.0, 84.0), ArraySeq(1,2))
    fromTensor(arr ** (ArraySeq(2.0), ArraySeq(1))) shouldEqual (ArraySeq(1764.0, 7056.0), ArraySeq(1,2))
  }

  "Tensor" should "mod" in {
    val arr: Tensor[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromTensor(arr % (ArraySeq(40), ArraySeq(1))) shouldEqual (ArraySeq(2,4), ArraySeq(1,2))
  }

  "Tensor" should "gt" in {
    val arr: Tensor[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromTensor(arr > (ArraySeq(42, 80), ArraySeq(1, 2))) shouldEqual (ArraySeq(false, true), ArraySeq(1,2))
  }

  "Tensor" should "gte" in {
    val arr: Tensor[Long] = (ArraySeq(42l, 84l), ArraySeq(1,2))
    fromTensor(arr >= (ArraySeq(42l, 80l), ArraySeq(1, 2))) shouldEqual (ArraySeq(true, true), ArraySeq(1,2))
  }

  "Tensor" should "lt" in {
    val arr: Tensor[Int] = (ArraySeq(42, 84), ArraySeq(1,2))
    fromTensor(arr < (ArraySeq(42, 80), ArraySeq(1, 2))) shouldEqual (ArraySeq(false, false), ArraySeq(1,2))
  }

  "Tensor" should "lte" in {
    val arr: Tensor[Long] = (ArraySeq(42l, 84l), ArraySeq(1,2))
    fromTensor(arr <= (ArraySeq(42l, 80l), ArraySeq(1, 2))) shouldEqual (ArraySeq(true, false), ArraySeq(1,2))
  }

  "Tensor" should "max" in {
    val arr: Tensor[Float] = (ArraySeq(42.0f, 84.0f), ArraySeq(1,2))
    val other: Tensor[Float] = (ArraySeq(50.0f, 80.0f), ArraySeq(1,2))
    fromTensor(arr max other) shouldEqual (ArraySeq(50.0f, 84.0f), ArraySeq(1,2))
  }

  "Tensor" should "min" in {
    val arr: Tensor[Double] = (ArraySeq(42.0, 84.0), ArraySeq(1,2))
    val other: Tensor[Double] = (ArraySeq(50.0, 80.0), ArraySeq(1,2))
    fromTensor(arr min other) shouldEqual (ArraySeq(42.0, 80.0), ArraySeq(1,2))
  }

  "Tensor" should "dot" in {
    val arr: Tensor[Double] = (ArraySeq(42.0, 84.0), ArraySeq(1,2))
    val other: Tensor[Double] = (ArraySeq(42.0, 84.0), ArraySeq(2,1))
    fromTensor(arr dot other) shouldEqual (ArraySeq(8820.0), ArraySeq(1,1))
  }
}
