package org.sciscala.ndscala


import scala.language.implicitConversions
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import ai.djl._
import ai.djl.ndarray._
//import scala.reflect.ClassTag

class DJLNDArraySpec extends AnyFlatSpec {

import DJLOps._

  //  type Supported = Int | Long | Float | Double //Union[Int]#or[Long]#or[Float]#or[Double]#create
//  type FloatSupported = Float | Double //Union[Float]#or[Double]#create

//  given ops1 as NDArrayOps[DJLNDArray]
 // implicit val ndarrayOps: NDArrayOps[DJLNDArray] = new ONNXScalaOps()

//  import NDArrayOps.ops._

//  implicit def convert[DType : ClassTag: Numeric](d: DType): DJLNDArray[DType] = DJLNDArrayFactory.getDJLNDArray(Array(d).toArray, Array(1).toArray)
//  implicit def toDJLNDArray[DType : ClassTag: Numeric](t: (Array[DType], Array[Int])): DJLNDArray[DType] = DJLNDArrayFactory.getDJLNDArray(t._1.toArray, t._2.toArray)
//  implicit def fromDJLNDArray[DType : ClassTag](t: DJLNDArray[DType]): (Array[DType], Array[Int]) = (t._1, t._2)



  /*
  "DJLNDArray" should "zero" in {
    (ndarrayOps.zeros[Int](Array(4))) shouldEqual (Array(0,0,0,0), Array(4))
  }

  "DJLNDArray" should "one" in {
    (ndarrayOps.ones[Int](Array(4))) shouldEqual (Array(1,1,1,1), Array(4))
  }

  "DJLNDArray" should "fill" in {
    (ndarrayOps.full(Array(4), 5)) shouldEqual (Array(5,5,5,5), Array(4))
  }
*/

//TODO: fix issue with type erasure on DJLNDArray, shouldn't need to cast here

// TODO: Don't do this, it silences match errors
  def doAssert(t: DJLNDArray[Boolean]) = {
    assert(t._1(0).asInstanceOf[Boolean])
  }

  "DJLNDArray" should "add" in {
    val arr: DJLNDArray[Int] = (Array(42, 84), Array(1,2))
    doAssert(((arr:DJLNDArray[Int]) + (arr:DJLNDArray[Int])) ==== (Array(84, 168), Array(1,2)))
  }

  "DJLNDArray" should "subtract" in {
    val arr: DJLNDArray[Int] = (Array(42, 84), Array(1,2))
    doAssert(((arr: DJLNDArray[Int]) - (arr: DJLNDArray[Int])) ==== (Array(0, 0), Array(1,2)))
  }

  "DJLNDArray" should "multiply" in {
    val arr: DJLNDArray[Int] = (Array(42, 84), Array(1,2))
    doAssert(((arr: DJLNDArray[Int]) * (arr: DJLNDArray[Int])) ==== (Array(1764, 7056), Array(1,2)))
  }

  //TODO: weirdness here with Int
  "DJLNDArray" should "divide" in {
    val arr: DJLNDArray[Float] = (Array(42.0f, 84.0f), Array(1,2))
    doAssert(((arr: DJLNDArray[Float]) / (arr: DJLNDArray[Float])) ==== (Array(1.0f, 1.0f), Array(1,2)))
  }

  "DJLNDArray" should "equal" in {
    val arr: DJLNDArray[Int] = (Array(42, 84), Array(1,2))
    doAssert(arr ==== (Array(42, 84), Array(1, 2))) 
  }

  "DJLNDArray" should "not equal" in {
    val arr: DJLNDArray[Int] = (Array(42, 84), Array(1,2)) 
    val result = (arr !=== (Array(42, 84), Array(1, 2)))
    assert(result._1(0) == false)
  }

  "DJLNDArray" should "reshape" in {
    val arr: DJLNDArray[Int] = (Array(42, 84), Array(1,2))
    doAssert((arr reShape Array(2,1)) ==== (Array(42, 84), Array(2,1)))
  }

  "DJLNDArray" should "transpose" in {
    val arr: DJLNDArray[Int] = (Array(1, 2, 3, 4), Array(2,2))
    doAssert(arr.transpose ==== (Array(1, 3, 2, 4), Array(2,2)))
  }

  "DJLNDArray" should "transpose with axes" in {
    val arr: DJLNDArray[Int] = (Array(1, 2, 3, 4), Array(2,2))
    doAssert((arr.transpose(Array(1,0), None)) ==== (Array(1, 3, 2, 4), Array(2,2)))
  }

  "DJLNDArray" should "round" in {
    val arr: DJLNDArray[Double] = (Array(41.7, 84.3), Array(1,2))
    doAssert((arr.round()) ==== (Array(42.0, 84.0), Array(1,2)))
  }

  "DJLNDArray" should "slice" in {
    val arr: DJLNDArray[Int] = (Array(1, 2, 3, 4), Array(4))
    doAssert((arr.slice(1,3, None)) ==== (Array(2, 3), Array(2)))
  }

  "DJLNDArray" should "squeeze" in {
    val arr: DJLNDArray[Int] = (Array(42, 84), Array(1,2))
    doAssert((arr squeeze (Array(0), None)) ==== (Array(42, 84), Array(2)))
  }

  "DJLNDArray" should "rank" in {
    val arr: DJLNDArray[Int] = (Array(42, 84), Array(1,2))
    arr.rank == 2
  }

  /*
  "DJLNDArray" should "clip" in {
    val arr: DJLNDArray[Double] = (Array(41.7, 84.5), Array(1,2))
    (arr.clip(50.0, 90.0)) shouldEqual (Array(50.0, 84.5), Array(1,2))
  }
*/
  "DJLNDArray" should "unary subtract" in {
    val arr: DJLNDArray[Int] = (Array(42, 84), Array(1,2))
    doAssert((- (arr:DJLNDArray[Int])) ==== (Array(-42, -84), Array(1,2)))
  }

  "DJLNDArray" should "abs" in {
    val arr: DJLNDArray[Int] = (Array(-42, 84), Array(1,2))
    doAssert((arr.abs()) ==== (Array(42, 84), Array(1,2)))
  }

  "DJLNDArray" should "ceil" in {
    val arr: DJLNDArray[Float] = (Array(-1.5f, 1.2f), Array(1,2))
    doAssert((arr.ceil()) ==== (Array(-1.0f, 2.0f), Array(1,2)))
  }

  "DJLNDArray" should "floor" in {
    val arr: DJLNDArray[Float] = (Array(-1.5f, 1.2f), Array(1,2))
    doAssert((arr.floor()) ==== (Array(-2.0f, 1.0f), Array(1,2)))
  }

  "DJLNDArray" should "log" in {
    val arr: DJLNDArray[Float] = (Array(1.0f, 10.0f), Array(1,2))
    doAssert((arr.log()) ==== (Array(0.0f, 2.30258512f), Array(1,2)))
  }

  "DJLNDArray" should "exp" in {
    val arr: DJLNDArray[Double] = (Array(-1.0, 0.0, 1.0), Array(1,3))
    val exp = arr.exp()
    //Tiny difference between CUDA and cpu - maybe fixed in latest CUDA
    doAssert((exp) ==== (Array(0.36787944117144233, 1.0, 2.718281828459045), Array(1,3)))
  }

  /*
  "DJLNDArray" should "concat" in {
    val arr: DJLNDArray[Double] = (Array(1.0, 4.0, 9.0), Array(1,3))
    val arrB: DJLNDArray[Double] = (Array(2.0, 3.0, 4.0, 5.0, 6.0, 7.0), Array(2,3))
    doAssert((Seq(arr, arrB) concat(0)) ==== (Array(1.0, 4.0, 9.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0), Array(3,3)))
  }

  "DJLNDArray" should "mean" in {
    val arr: DJLNDArray[Float] = (Array(1.0f, 4.0f, 9.0f), Array(1,3))
    val arrB: DJLNDArray[Float] = (Array(3.0f, 2.0f, 3.0f), Array(1,3))
    doAssert((mean(Seq(arr, arrB))) ==== (Array(2.0f, 3.0f, 6.0f), Array(1,3)))
  }
*/
  "DJLNDArray" should "sqrt" in {
    val arr: DJLNDArray[Double] = (Array(1.0, 4.0, 9.0), Array(1,3))
    doAssert((arr.sqrt()) ==== (Array(1.0, 2.0, 3.0), Array(1,3)))
  }

  "DJLNDArray" should "cos" in {
    val arr: DJLNDArray[Float] = (Array(-1.0f, 0.0f, 1.0f), Array(1,3))
    val result = arr.cos()
    println(result._1(0))
    println(result._1(1))
    println(result._1(2))
    doAssert((result:DJLNDArray[Float]) ==== (Array(0.5403023f, 1.0f, 0.5403023f), Array(1,3)))
  }

  "DJLNDArray" should "sin" in {
    val arr: DJLNDArray[Float] = (Array(-1.0f, 0.0f, 1.0f), Array(1,3))
    doAssert((arr.sin()) ==== (Array(-0.84147096f, 0.0f, 0.84147096f), Array(1,3)))
  }

  "DJLNDArray" should "tan" in {
    val arr: DJLNDArray[Float] = (Array(-1.0f, 0.0f, 1.0f), Array(1,3))
    doAssert((arr.tan()) ==== (Array(-1.5574077f, 0.0f, 1.5574077f), Array(1,3)))
  }

  "DJLNDArray" should "tanh" in {
    val arr: DJLNDArray[Float] = (Array(-1.0f, 0.0f, 1.0f), Array(1,3))
    doAssert((arr.tanh()) ==== (Array(-0.7615942f, 0.0f, 0.7615942f), Array(1,3)))
  }

  "DJLNDArray" should "acos" in {
    val arr: DJLNDArray[Float] = (Array(-1.0f, 0.0f, 1.0f), Array(1,3))
    doAssert((arr.acos()) ==== (Array(3.1415927f, 1.5707964f, 0.0f), Array(1,3)))
  }

  "DJLNDArray" should "asin" in {
    val arr: DJLNDArray[Float] = (Array(-1.0f, 0.0f, 1.0f), Array(1,3))
    doAssert((arr.asin()) ==== (Array(-1.5707964f, 0.0f, 1.5707964f), Array(1,3)))
  }

  "DJLNDArray" should "atan" in {
    val arr: DJLNDArray[Float] = (Array(-1.0f, 0.0f, 1.0f), Array(1,3))
    doAssert((arr.atan()) ==== (Array(-0.7853982f, 0.0f, 0.7853982f), Array(1,3)))
  }

  "DJLNDArray" should "atanh" in {
    val arr: DJLNDArray[Float] = (Array(-0.5f, 0.0f, 0.5f), Array(1,3))
    doAssert((arr.atanh()) ==== (Array(-0.54930615f, 0.0f, 0.54930615f), Array(1,3)))
  }

  //TODO: test sigmoid, relu

  "DJLNDArray" should "pow" in {
    val arr: DJLNDArray[Double] = (Array(42.0, 84.0), Array(1,2))
    doAssert((arr ** (Array(2.0), Array(1))) ==== (Array(1764.0, 7056.0), Array(1,2)))
  }

  "DJLNDArray" should "mod" in {
    val arr: DJLNDArray[Int] = (Array(42, 84), Array(1,2))
    doAssert((arr % (Array(40), Array(1))) ==== (Array(2,4), Array(1,2)))
  }

  "DJLNDArray" should "gt" in {
    val arr: DJLNDArray[Int] = (Array(42, 84), Array(1,2))
    val result = (arr > (Array(42, 80), Array(1,2)))
    assert(result._1(0) == false)
    assert(result._1(1) == true)
  }

  "DJLNDArray" should "gte" in {
    val arr: DJLNDArray[Long] = (Array(42l, 84l), Array(1,2))
    val result = (arr >= (Array(42l, 80l), Array(1, 2)))
    assert(result._1(0) == true)
    assert(result._1(1) == true) 
  }

  "DJLNDArray" should "lt" in {
    val arr: DJLNDArray[Int] = (Array(42, 84), Array(1,2))
    val result = (arr < (Array(42, 80), Array(1, 2)))
    assert(result._1(0) == false)
    assert(result._1(1) == false) 
  }

  "DJLNDArray" should "lte" in {
    val arr: DJLNDArray[Long] = (Array(42l, 84l), Array(1,2))
    val result = (arr <= (Array(42l, 80l), Array(1, 2)))
    assert(result._1(0) == true)
    assert(result._1(1) == false) 
  }

  "DJLNDArray" should "max" in {
    val arr: DJLNDArray[Float] = (Array(42.0f, 84.0f), Array(1,2))

    doAssert((arr max (Array(50.0f, 80.0f), Array(1,2))) ==== (Array(50.0f, 84.0f), Array(1,2)))
  }

  "DJLNDArray" should "min" in {
    val arr: DJLNDArray[Double] = (Array(42.0, 84.0), Array(1,2))
    doAssert((arr min (Array(50.0, 80.0), Array(1,2))) ==== (Array(42.0, 80.0), Array(1,2)))
  }

  "DJLNDArray" should "matmul" in {
    val arr: DJLNDArray[Double] = (Array(42.0, 84.0), Array(1,2))
    val other: DJLNDArray[Double] = (Array(42.0, 84.0), Array(2,1))
    val result = (((arr: DJLNDArray[Double]) matmul (other: DJLNDArray[Double])) ==== (Array(8820.0), Array(1,1)))
    doAssert(result)
  }
}
