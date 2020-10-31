package org.sciscala.ndscala

//import scala.collection.immutable.ArraySeq
import scala.language.implicitConversions
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.platanios.tensorflow.api._
//import scala.reflect.ClassTag
import TensorFlowOps._


class TensorFlowTensorSpec extends AnyFlatSpec {
//  type Supported = Int | Long | Float | Double //Union[Int]#or[Long]#or[Float]#or[Double]#create
//  type FloatSupported = Float | Double //Union[Float]#or[Double]#create

//  given ops1 as NDArrayOps[Tensor]
 // implicit val ndarrayOps: NDArrayOps[Tensor] = new ONNXScalaOps()

//  import NDArrayOps.ops._

//  implicit def convert[DType : ClassTag: Numeric](d: DType): Tensor[DType] = TensorFactory.getTensor(Array(d).toArray, Array(1).toArray)
//  implicit def toTensor[DType : ClassTag: Numeric](t: (Array[DType], Array[Int])): Tensor[DType] = TensorFactory.getTensor(t._1.toArray, t._2.toArray)
//  implicit def fromTensor[DType : ClassTag](t: Tensor[DType]): (Array[DType], Array[Int]) = (t._1, t._2)



  /*
  "Tensor" should "zero" in {
    (ndarrayOps.zeros[Int](Array(4))) shouldEqual (Array(0,0,0,0), Array(4))
  }

  "Tensor" should "one" in {
    (ndarrayOps.ones[Int](Array(4))) shouldEqual (Array(1,1,1,1), Array(4))
  }

  "Tensor" should "fill" in {
    (ndarrayOps.full(Array(4), 5)) shouldEqual (Array(5,5,5,5), Array(4))
  }
*/

// TODO: Don't do this, it silences match errors
  def doAssert(t: Tensor[Boolean]) = {
    assert(t._1(0))
  }

  "Tensor" should "add" in {
    val arr: Tensor[Int] = (Array(42, 84), Array(1,2))
    doAssert(((arr + arr) ====(Array(84, 168), Array(1,2))))
  }

  "Tensor" should "subtract" in {
    val arr: Tensor[Int] = (Array(42, 84), Array(1,2))
    doAssert((arr - arr) ==== (Array(0, 0), Array(1,2)))
  }

  "Tensor" should "multiply" in {
    val arr: Tensor[Int] = (Array(42, 84), Array(1,2))
    doAssert((arr * arr) ==== (Array(1764, 7056), Array(1,2)))
  }

  "Tensor" should "divide" in {
    val arr: Tensor[Int] = (Array(42, 84), Array(1,2))
    doAssert((arr / arr) ==== (Array(1, 1), Array(1,2)))
  }

  "Tensor" should "equal" in {
    val arr: Tensor[Int] = (Array(42, 84), Array(1,2))
    doAssert(arr ==== (Array(42, 84), Array(1, 2))) 
  }

  "Tensor" should "not equal" in {
    val arr: Tensor[Int] = (Array(42, 84), Array(1,2)) 
    val result = (arr !=== (Array(42, 84), Array(1, 2)))
    assert(result._1(0) == false)
  }

  "Tensor" should "reshape" in {
    val arr: Tensor[Int] = (Array(42, 84), Array(1,2))
    doAssert((arr reShape Array(2,1)) ==== (Array(42, 84), Array(2,1)))
  }

  "Tensor" should "transpose" in {
    val arr: Tensor[Int] = (Array(1, 2, 3, 4), Array(2,2))
    doAssert(arr.transpose ==== (Array(1, 3, 2, 4), Array(2,2)))
  }

  "Tensor" should "transpose with axes" in {
    val arr: Tensor[Int] = (Array(1, 2, 3, 4), Array(2,2))
    doAssert((arr.transpose(Array(1,0), None)) ==== (Array(1, 3, 2, 4), Array(2,2)))
  }

  "Tensor" should "round" in {
    val arr: Tensor[Double] = (Array(41.7, 84.3), Array(1,2))
    doAssert((arr.round()) ==== (Array(42.0, 84.0), Array(1,2)))
  }

  "Tensor" should "slice" in {
    val arr: Tensor[Int] = (Array(1, 2, 3, 4), Array(4))
    doAssert((arr.slice(1,3, None)) ==== (Array(2, 3), Array(2)))
  }

  "Tensor" should "squeeze" in {
    val arr: Tensor[Int] = (Array(42, 84), Array(1,2))
    doAssert((arr.squeeze(Array(0), None)) ==== (Array(42, 84), Array(2)))
  }

  "Tensor" should "rank" in {
    val arr: Tensor[Int] = (Array(42, 84), Array(1,2))
    arr.rank == 2
  }

  /*
  "Tensor" should "clip" in {
    val arr: Tensor[Double] = (Array(41.7, 84.5), Array(1,2))
    (arr.clip(50.0, 90.0)) shouldEqual (Array(50.0, 84.5), Array(1,2))
  }
*/
  "Tensor" should "unary subtract" in {
    val arr: Tensor[Int] = (Array(42, 84), Array(1,2))
    doAssert((-arr) ==== (Array(-42, -84), Array(1,2)))
  }

  "Tensor" should "abs" in {
    val arr: Tensor[Int] = (Array(-42, 84), Array(1,2))
    doAssert((arr.abs()) ==== (Array(42, 84), Array(1,2)))
  }

  "Tensor" should "ceil" in {
    val arr: Tensor[Float] = (Array(-1.5f, 1.2f), Array(1,2))
    doAssert((arr.ceil()) ==== (Array(-1.0f, 2.0f), Array(1,2)))
  }

  "Tensor" should "floor" in {
    val arr: Tensor[Float] = (Array(-1.5f, 1.2f), Array(1,2))
    doAssert((arr.floor()) ==== (Array(-2.0f, 1.0f), Array(1,2)))
  }

  "Tensor" should "log" in {
    val arr: Tensor[Float] = (Array(1.0f, 10.0f), Array(1,2))
    doAssert((arr.log()) ==== (Array(0.0f, 2.30258512f), Array(1,2)))
  }

  "Tensor" should "exp" in {
    val arr: Tensor[Double] = (Array(-1.0, 0.0, 1.0), Array(1,3))
    val exp = arr.exp()
    //Tiny difference between CUDA and cpu - maybe fixed in latest CUDA
    doAssert((exp) ==== (Array(0.36787944117144233, 1.0, 2.718281828459045), Array(1,3)))
  }

  /*
  "Tensor" should "concat" in {
    val arr: Tensor[Double] = (Array(1.0, 4.0, 9.0), Array(1,3))
    val arrB: Tensor[Double] = (Array(2.0, 3.0, 4.0, 5.0, 6.0, 7.0), Array(2,3))
    doAssert((Seq(arr, arrB) concat(0)) ==== (Array(1.0, 4.0, 9.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0), Array(3,3)))
  }

  "Tensor" should "mean" in {
    val arr: Tensor[Float] = (Array(1.0f, 4.0f, 9.0f), Array(1,3))
    val arrB: Tensor[Float] = (Array(3.0f, 2.0f, 3.0f), Array(1,3))
    doAssert((mean(Seq(arr, arrB))) ==== (Array(2.0f, 3.0f, 6.0f), Array(1,3)))
  }
*/
  "Tensor" should "sqrt" in {
    val arr: Tensor[Double] = (Array(1.0, 4.0, 9.0), Array(1,3))
    doAssert((arr.sqrt()) ==== (Array(1.0, 2.0, 3.0), Array(1,3)))
  }

  "Tensor" should "cos" in {
    val arr: Tensor[Float] = (Array(-1.0f, 0.0f, 1.0f), Array(1,3))
    doAssert((arr.cos()) ==== (Array(0.5403023f, 1.0f, 0.5403023f), Array(1,3)))
  }

  "Tensor" should "sin" in {
    val arr: Tensor[Float] = (Array(-1.0f, 0.0f, 1.0f), Array(1,3))
    doAssert((arr.sin()) ==== (Array(-0.84147096f, 0.0f, 0.84147096f), Array(1,3)))
  }

  "Tensor" should "tan" in {
    val arr: Tensor[Float] = (Array(-1.0f, 0.0f, 1.0f), Array(1,3))
    doAssert((arr.tan()) ==== (Array(-1.5574077f, 0.0f, 1.5574077f), Array(1,3)))
  }

  "Tensor" should "tanh" in {
    val arr: Tensor[Float] = (Array(-1.0f, 0.0f, 1.0f), Array(1,3))
    doAssert((arr.tanh()) ==== (Array(-0.7615942f, 0.0f, 0.7615942f), Array(1,3)))
  }

  "Tensor" should "acos" in {
    val arr: Tensor[Float] = (Array(-1.0f, 0.0f, 1.0f), Array(1,3))
    doAssert((arr.acos()) ==== (Array(3.1415927f, 1.5707964f, 0.0f), Array(1,3)))
  }

  "Tensor" should "asin" in {
    val arr: Tensor[Float] = (Array(-1.0f, 0.0f, 1.0f), Array(1,3))
    doAssert((arr.asin()) ==== (Array(-1.5707964f, 0.0f, 1.5707964f), Array(1,3)))
  }

  "Tensor" should "atan" in {
    val arr: Tensor[Float] = (Array(-1.0f, 0.0f, 1.0f), Array(1,3))
    doAssert((arr.atan()) ==== (Array(-0.7853982f, 0.0f, 0.7853982f), Array(1,3)))
  }

  "Tensor" should "atanh" in {
    val arr: Tensor[Float] = (Array(-0.5f, 0.0f, 0.5f), Array(1,3))
    doAssert((arr.atanh()) ==== (Array(-0.54930615f, 0.0f, 0.54930615f), Array(1,3)))
  }

  //TODO: test sigmoid, relu

  "Tensor" should "pow" in {
    val arr: Tensor[Double] = (Array(42.0, 84.0), Array(1,2))
    doAssert((arr ** (Array(2.0), Array(1))) ==== (Array(1764.0, 7056.0), Array(1,2)))
  }

  "Tensor" should "mod" in {
    val arr: Tensor[Int] = (Array(42, 84), Array(1,2))
    doAssert((arr % (Array(40), Array(1))) ==== (Array(2,4), Array(1,2)))
  }

  "Tensor" should "gt" in {
    val arr: Tensor[Int] = (Array(42, 84), Array(1,2))
    val result = (arr > (Array(42, 80), Array(1,2)))
    assert(result._1(0) == false)
    assert(result._1(1) == true)
  }

  "Tensor" should "gte" in {
    val arr: Tensor[Long] = (Array(42l, 84l), Array(1,2))
    val result = (arr >= (Array(42l, 80l), Array(1, 2)))
    assert(result._1(0) == true)
    assert(result._1(1) == true) 
  }

  "Tensor" should "lt" in {
    val arr: Tensor[Int] = (Array(42, 84), Array(1,2))
    val result = (arr < (Array(42, 80), Array(1, 2)))
    assert(result._1(0) == false)
    assert(result._1(1) == false) 
  }

  "Tensor" should "lte" in {
    val arr: Tensor[Long] = (Array(42l, 84l), Array(1,2))
    val result = (arr <= (Array(42l, 80l), Array(1, 2)))
    assert(result._1(0) == true)
    assert(result._1(1) == false) 
  }

  "Tensor" should "max" in {
    val arr: Tensor[Float] = (Array(42.0f, 84.0f), Array(1,2))
    val other: Tensor[Float] = (Array(50.0f, 80.0f), Array(1,2))
    doAssert((arr max other) ==== (Array(50.0f, 84.0f), Array(1,2)))
  }

  "Tensor" should "min" in {
    val arr: Tensor[Double] = (Array(42.0, 84.0), Array(1,2))
    val other: Tensor[Double] = (Array(50.0, 80.0), Array(1,2))
    doAssert((arr min other) ==== (Array(42.0, 80.0), Array(1,2)))
  }

  "Tensor" should "matmul" in {
    val arr: Tensor[Double] = (Array(42.0, 84.0), Array(1,2))
    val other: Tensor[Double] = (Array(42.0, 84.0), Array(2,1))
    val result = (arr matmul other) ==== (Array(8820.0), Array(1,1))
    doAssert(result)
  }
}
