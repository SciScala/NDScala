package org.sciscala.ndscala

//import scala.collection.immutable.ArraySeq
import scala.language.implicitConversions
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import org.emergentorder.onnx.Tensors._
import org.emergentorder.compiletime._
import io.kjaer.compiletime._
//import scala.reflect.ClassTag
import DJLOps._


//TODO: tests for higher rank tensors
class DJLNDArraySpec extends AnyFlatSpec {
//  type Supported = Int | Long | Float | Double //Union[Int]#or[Long]#or[Float]#or[Double]#create
//  type FloatSupported = Float | Double //Union[Float]#or[Double]#create

//  given ops1 as NDArrayOps[Tensor]
 // implicit val ndarrayOps: NDArrayOps[Tensor] = new ONNXScalaOps()

//  import NDArrayOps.ops._

//  implicit def convert[DType : ClassTag: Numeric](d: DType): DJLNDArray[DType] = TensorFactory.getTensor(Array(d).toArray, Mat(1).toArray)
//  implicit def toDJLNDArray[DType : ClassTag: Numeric](t: (Array[DType], Mat[?,?,?,Int])): DJLNDArray[DType] = TensorFactory.getTensor(t._1.toArray, t._2.toArray)
//  implicit def fromDJLNDArray[DType : ClassTag](t: DJLNDArray[DType]): (Array[DType], Mat[?,?,?,Int]) = (t._1, t._2)



  /*
  "Tensor" should "zero" in {
    (ndarrayOps.zeros[Int](Array(4))) shouldEqual (Array(0,0,0,0), Mat(4))
  }

  "Tensor" should "one" in {
    (ndarrayOps.ones[Int](Array(4))) shouldEqual (Array(1,1,1,1), Mat(4))
  }

  "Tensor" should "fill" in {
    (ndarrayOps.full(Array(4), 5)) shouldEqual (Array(5,5,5,5), Mat(4))
  }
*/

  type TT = "TensorTypeDenotation"
  type TD = "TensorShapeDenotation" ##: TSNil


// TODO: Don't do this, it silences match errors
  def doAssert(t: DJLNDArray[Boolean, Axes]) = {
    assert((t.toBooleanArray())(0))
  }
  
  "Tensor" should "add" in {
    val arr: DJLNDArray[Int,(TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr + arr) ====Tensor(Array(84, 168),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor" should "subtract" in {
    val arr: DJLNDArray[Int, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr - arr) ==== Tensor(Array(0, 0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor" should "multiply" in {
    val arr: DJLNDArray[Int, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr * arr) ==== Tensor(Array(1764, 7056),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor" should "divide" in {
    val arr: DJLNDArray[Int, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr / arr) ==== Tensor(Array(1, 1),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor" should "equal" in {
    val arr: DJLNDArray[Int, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert(arr ==== Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)) 
  }

  "Tensor" should "not equal" in {
    val arr: DJLNDArray[Int, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil) 
    val result = (arr !=== Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
    assert(result.toBooleanArray()(0) == false)
  }
 
  "Tensor" should "reshape" in {
    val arr: DJLNDArray[Int, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val expectedResult: DJLNDArray[Int, (TT, TD, 2 #: 1 #: SNil)] = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 1 #: SNil)
    doAssert((arr.reshape[TT, TD, 2 #: 1 #: SNil]) ==== expectedResult)
  }

  "Tensor" should "transpose" in {
    val arr: DJLNDArray[Int, (TT, TD, 2 #: 2 #: SNil)] = Tensor(Array(1, 2, 3, 4),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 2 #: SNil)
    doAssert(arr.transpose ==== Tensor(Array(1, 3, 2, 4),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 2 #: SNil))
  }

  //Broken
  /*
  "Tensor" should "transpose with axes" in {
    val arr: DJLNDArray[Int, (TT, TD, 2 #: 2 #: SNil)] = Tensor(Array(1, 2, 3, 4),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 2 #: SNil)
    doAssert((arr.transpose(Array(1,0), None)) ==== Tensor(Array(1, 3, 2, 4),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 2 #: SNil))
  }
*/
  "Tensor" should "round" in {
    val arr: DJLNDArray[Double, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(41.7, 84.3),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr.round()) ==== Tensor(Array(42.0, 84.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  /*
  "Tensor" should "slice" in {
    val arr: DJLNDArray[Int, (TT, TD, 4 #: SNil)] = Tensor(Array(1, 2, 3, 4),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 4 #: SNil )
    val expectedResult = Tensor(Array(2, 3),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: SNil)
    doAssert((arr.slice[TT, TD, 2 #: SNil](1,3)) ==== expectedResult)
  }

  */

  "Tensor" should "squeeze" in {
    val arr: DJLNDArray[Int, (TT, "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)] = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val expectedResult = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: SNil)
    doAssert((arr.squeeze[TT, 0 ::: INil]) ==== expectedResult)
  }

  "Tensor" should "rank" in {
    val arr: DJLNDArray[Int, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    arr.rank == 2
  }

  /*
  "Tensor" should "clip" in {
    val arr: DJLNDArray[Double] = (Array(41.7, 84.5), Mat(1,2))
    (arr.clip(50.0, 90.0)) shouldEqual (Array(50.0, 84.5), Mat(1,2))
  }
*/
  "Tensor" should "unary subtract" in {
    val arr: DJLNDArray[Int, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((-arr) ==== Tensor(Array(-42, -84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor" should "abs" in {
    val arr: DJLNDArray[Int, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(-42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr.abs()) ==== Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor" should "ceil" in {
    val arr: DJLNDArray[Float, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(-1.5f, 1.2f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr.ceil()) ==== Tensor(Array(-1.0f, 2.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor" should "floor" in {
    val arr: DJLNDArray[Float, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(-1.5f, 1.2f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr.floor()) ==== Tensor(Array(-2.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor" should "log" in {
    val arr: DJLNDArray[Float, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(1.0f, 10.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr.log()) ==== Tensor(Array(0.0f, 2.30258512f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor" should "exp" in {
    val arr: DJLNDArray[Double, (TT, TD, 1 #: 3 #: SNil)] = Tensor(Array(-1.0, 0.0, 1.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)

    val exp = arr.exp()
    //Tiny difference between CUDA and cpu - maybe fixed in latest CUDA
    doAssert((exp) ==== Tensor(Array(0.36787944117144233, 1.0, 2.718281828459045),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  /*
  "Tensor" should "concat" in {
    val arr: DJLNDArray[Double] = (Array(1.0, 4.0, 9.0), Mat(1,3))
    val arrB: DJLNDArray[Double] = (Array(2.0, 3.0, 4.0, 5.0, 6.0, 7.0), Mat(2,3))
    doAssert((Seq(arr, arrB) concat(0)) ==== (Array(1.0, 4.0, 9.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0), Mat(3,3)))
  }

  "Tensor" should "mean" in {
    val arr: DJLNDArray[Float] = (Array(1.0f, 4.0f, 9.0f), Mat(1,3))
    val arrB: DJLNDArray[Float] = (Array(3.0f, 2.0f, 3.0f), Mat(1,3))
    doAssert((mean(Seq(arr, arrB))) ==== (Array(2.0f, 3.0f, 6.0f), Mat(1,3)))
  }
*/
  "Tensor" should "sqrt" in {
    val arr: DJLNDArray[Double, (TT, TD, 1 #: 3 #: SNil)] = Tensor(Array(1.0, 4.0, 9.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.sqrt()) ==== Tensor(Array(1.0, 2.0, 3.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor" should "cos" in {
    val arr: DJLNDArray[Float, (TT, TD, 1 #: 3 #: SNil)] = Tensor(Array(-1.0f, 0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.cos()) ==== Tensor(Array(0.54030234f, 1.0f, 0.54030234f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor" should "sin" in {
    val arr: DJLNDArray[Float, (TT, TD, 1 #: 3 #: SNil)] = Tensor(Array(-1.0f, 0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.sin()) ==== Tensor(Array(-0.84147096f, 0.0f, 0.84147096f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor" should "tan" in {
    val arr: DJLNDArray[Float, (TT, TD, 1 #: 3 #: SNil)] = Tensor(Array(-1.0f, 0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.tan()) ==== Tensor(Array(-1.5574077f, 0.0f, 1.5574077f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor" should "tanh" in {
    val arr: DJLNDArray[Float, (TT, TD, 1 #: 3 #: SNil)] = Tensor(Array(-1.0f, 0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.tanh()) ==== Tensor(Array(-0.7615942f, 0.0f, 0.7615942f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor" should "acos" in {
    val arr: DJLNDArray[Float, (TT, TD, 1 #: 3 #: SNil)] = Tensor(Array(-1.0f, 0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.acos()) ==== Tensor(Array(3.1415927f, 1.5707964f, 0.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor" should "asin" in {
    val arr: DJLNDArray[Float, (TT, TD, 1 #: 3 #: SNil)] = Tensor(Array(-1.0f, 0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.asin()) ==== Tensor(Array(-1.5707964f, 0.0f, 1.5707964f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor" should "atan" in {
    val arr: DJLNDArray[Float, (TT, TD, 1 #: 3 #: SNil)] = Tensor(Array(-1.0f, 0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.atan()) ==== Tensor(Array(-0.7853982f, 0.0f, 0.7853982f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  //Not implemented
  /*
  "Tensor" should "atanh" in {
    val arr: DJLNDArray[Float, (TT, TD, 1 #: 3 #: SNil)] = Tensor(Array(-0.5f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.atanh()) ==== Tensor(Array(-0.54930615f, 0.0f, 0.54930615f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }
  */
  //TODO: test sigmoid, relu

  "Tensor" should "pow" in {
    val arr: DJLNDArray[Double, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(42.0, 84.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr ** Tensor(Array(2.0, 2.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)) ==== Tensor(Array(1764.0, 7056.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor" should "mod" in {
   val arr: DJLNDArray[Int, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr % Tensor(Array(40, 40),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)) ==== Tensor(Array(2,4),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor" should "gt" in {
    val arr: DJLNDArray[Int, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val result = (arr > Tensor(Array(42, 80),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
    assert(result.toBooleanArray()(0) == false)
    assert(result.toBooleanArray()(1) == true)
  }

  "Tensor" should "gte" in {
    val arr: DJLNDArray[Long, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(42l, 84l),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val result = (arr >= Tensor(Array(42l, 80l),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
    assert(result.toBooleanArray()(0) == true)
    assert(result.toBooleanArray()(1) == true) 
  }

  "Tensor" should "lt" in {
    val arr: DJLNDArray[Int, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val result = (arr < Tensor(Array(42, 80),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
    assert(result.toBooleanArray()(0) == false)
    assert(result.toBooleanArray()(1) == false) 
  }

  "Tensor" should "lte" in {
    val arr: DJLNDArray[Long, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(42l, 84l),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val result = (arr <= Tensor(Array(42l, 80l),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
    assert(result.toBooleanArray()(0) == true)
    assert(result.toBooleanArray()(1) == false) 
  }

  "Tensor" should "max" in {
    val arr: DJLNDArray[Float, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(42.0f, 84.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val other: DJLNDArray[Float, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(50.0f, 80.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr max other) ==== Tensor(Array(50.0f, 84.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor" should "min" in {
    val arr: DJLNDArray[Double, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(42.0, 84.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val other: DJLNDArray[Double, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(50.0, 80.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr min other) ==== Tensor(Array(42.0, 80.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor" should "matmul" in {
    val arr: DJLNDArray[Double, (TT, TD, 1 #: 2 #: SNil)] = Tensor(Array(42.0, 84.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val other: DJLNDArray[Double, (TT, TD, 2 #: 1 #: SNil)] = Tensor(Array(42.0, 84.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 1 #: SNil)
    val expectedResult: DJLNDArray[Double, (TT, TD, 1 #: 1 #: SNil)] = Tensor(Array(8820.0), valueOf[TT], "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: SNil)
    val result = (arr.matmul(other)) ==== expectedResult
    doAssert(result)
  }
}
