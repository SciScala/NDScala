package org.sciscala.ndscala

//import scala.collection.immutable.ArraySeq
import scala.language.implicitConversions
import org.scalatest.freespec.AsyncFreeSpec
import org.scalatest.matchers.should.Matchers
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.Tensors.Tensor._

import org.emergentorder.compiletime._
import org.emergentorder.io.kjaer.compiletime._
//import scala.reflect.ClassTag
import ONNXScalaOps._

import cats.effect.testing.scalatest.AsyncIOSpec

//TODO: tests for higher rank tensors
class ONNXScalaTensorSpec extends AsyncFreeSpec with AsyncIOSpec with Matchers{

//Uncomment the following line to run tests in Scala.js
//  implicit override def executionContext = scala.scalajs.concurrent.JSExecutionContext.Implicits.queue


//  type Supported = Int | Long | Float | Double //Union[Int]#or[Long]#or[Float]#or[Double]#create
//  type FloatSupported = Float | Double //Union[Float]#or[Double]#create

//  given ops1 as NDArrayOps[Tensor]
 // implicit val ndarrayOps: NDArrayOps[Tensor] = new ONNXScalaOps()

//  import NDArrayOps.ops._

//  implicit def convert[DType : ClassTag: Numeric](d: DType): Tensor[DType] = TensorFactory.getTensor(Array(d).toArray, Mat(1).toArray)
//  implicit def toTensor[DType : ClassTag: Numeric](t: (Array[DType], Mat[?,?,?,Int])): Tensor[DType] = TensorFactory.getTensor(t._1.toArray, t._2.toArray)
//  implicit def fromTensor[DType : ClassTag](t: Tensor[DType]): (Array[DType], Mat[?,?,?,Int]) = (t._1, t._2)



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

//TODO: more negative tests

// TODO: Don't do this, it silences match errors
  def doAssert(t: Tensor[Boolean, Axes]) = {
    //TODO: try to do memoization here
    t.data.asserting(_(0) shouldBe true)
  }

  "Tensor should add" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert(((arr + arr) ====Tensor(Array(84, 168),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)))
  }

  "Tensor fail to compile add with a different shape" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)  
    val arrB = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 1 #: SNil) 
    assertTypeError("arr + arrB")
  }

  "Tensor should subtract" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr - arr) ==== Tensor(Array(0, 0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor should fail to compile subtract with a different shape" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val arrB = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 1 #: SNil)
    assertTypeError("arr - arrB")
  }

  "Tensor should multiply" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr * arr) ==== Tensor(Array(1764, 7056),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor should fail to compile multiply with a different shape" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val arrB = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 1 #: SNil)
    assertTypeError("arr * arrB")
  }

  "Tensor should divide" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr / arr) ==== Tensor(Array(1, 1),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor should fail to compile divide with a different shape" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val arrB = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 1 #: SNil)
    assertTypeError("arr / arrB")
  }

  "Tensor should equal" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert(arr ==== Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)) 
  }

  "Tensor should fail to compile equal with a different shape" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val arrB = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 1 #: SNil)
    assertTypeError("arr ==== arrB")
  }

  "Tensor should not equal" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil) 
    val result = (arr !=== Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))

    result.data.asserting(_(0) shouldBe false)
  }
 
  "Tensor should fail to compile not equal with a different shape" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val arrB = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 1 #: SNil)
    assertTypeError("arr !=== arrB")
  }

  "Tensor should reshape" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val expectedResult = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 1 #: SNil)
    doAssert((arr.reshape[TT, TD, 2 #: 1 #: SNil](Some(0))) ==== expectedResult)
  }

  "Tensor should expand" in {
    val arr = Tensor(Array(42),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: SNil)
    val expectedResult = Tensor(Array(42, 42, 42, 42),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 2 #: SNil)
    doAssert((arr.expand[TT, TD, 2 #: 2 #: SNil]) ==== expectedResult)
  }

  "Tensor should fail to compile reshape where the new shape has wrong size" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    assertTypeError("arr.reshape[TT, TD, 3 #: 1 #: SNil]()")
  }

  "Tensor should reduceSum with keepdims on" in {
    val arr = Tensor(Array(42f, 84f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val expectedResult = Tensor(Array(126f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: SNil)
    doAssert((arr.reduceSum[TT, 1 ::: INil, true]) ==== expectedResult)
  }

  "Tensor should reduceSum with keepdims off" in {
    val arr = Tensor(Array(42f, 84f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val expectedResult = Tensor(Array(126f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: SNil)
    doAssert((arr.reduceSum[TT, 1 ::: INil, false]) ==== expectedResult)
  }

  "Tensor should reduceLogSum with keepdims on" in {
    val arr = Tensor(Array(42f, 84f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val expectedResult = Tensor(Array(4.836282f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: SNil)
    doAssert((arr.reduceLogSum[TT, 1 ::: INil, true]) ==== expectedResult)
  }

  "Tensor should reduceMax with keepdims on" in {
    val arr = Tensor(Array(42f, 84f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val expectedResult = Tensor(Array(84.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: SNil)
    doAssert((arr.reduceMax[TT, 1 ::: INil, true]) ==== expectedResult)
  }

  "Tensor should reduceMin with keepdims on" in {
    val arr = Tensor(Array(42f, 84f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val expectedResult = Tensor(Array(42.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: SNil)
    doAssert((arr.reduceMin[TT, 1 ::: INil, true]) ==== expectedResult)
  }

  "Tensor should reduceProd with keepdims on" in {
    val arr = Tensor(Array(42f, 84f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val expectedResult = Tensor(Array(3528.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: SNil)
    doAssert((arr.reduceProd[TT, 1 ::: INil, true]) ==== expectedResult)
  }

  "Tensor should reduceSumSquare with keepdims on" in {
    val arr = Tensor(Array(42f, 84f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val expectedResult = Tensor(Array(8820.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: SNil)
    doAssert((arr.reduceSumSquare[TT, 1 ::: INil, true]) ==== expectedResult)
  }

  "Tensor should argmax" in {
    val arr: Tensor[Float, (TT, "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)] = Tensor(Array(42f, 84f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val expectedResult: Tensor[Long, (TT, "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: SNil)] = Tensor(Array(1),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: SNil)
    doAssert(arr.argMax[TT, 1 ::: INil, true] ==== expectedResult)
  }

  "Tensor should argmin" in {
    val arr: Tensor[Float, (TT, "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)] = Tensor(Array(42f, 84f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val expectedResult: Tensor[Long, (TT, "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: SNil)] = Tensor(Array(0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: SNil)
    doAssert(arr.argMin[TT, 1 ::: INil, true] ==== expectedResult)
  }

  "Tensor should transpose" in {
    val arr = Tensor(Array(1, 2, 3, 4),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 2 #: SNil)
    doAssert(arr.transpose ==== Tensor(Array(1, 3, 2, 4),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 2 #: SNil))
  }

  "Tensor should transpose with axes" in {
    val arr = Tensor(Array(1, 2, 3, 4),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 2 #: SNil)
    doAssert((arr.transpose(Array(1,0))) ==== Tensor(Array(1, 3, 2, 4),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 2 #: SNil))
  }

  "Tensor should round" in {
    val arr = Tensor(Array(41.7, 84.3),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr.round()) ==== Tensor(Array(42.0, 84.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor should slice" in {
    val arr = Tensor(Array(1, 2, 3, 4),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 4 #: SNil )
    val expectedResult = Tensor(Array(2, 3),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: SNil)

    doAssert((arr.slice[TT, 1 ::: INil, 3 ::: INil]) ==== expectedResult)
  }

    "Tensor should slice 2d" in {
    val arr = Tensor(Array(1, 2, 3, 4),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 2 #: 2 #: SNil )
    val expectedResult = Tensor(Array(1,2),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 2 #: 1 #: SNil)
    doAssert((arr.slice[TT, 0 ::: 0 ::: INil, 2 ::: 1 ::: INil]) ==== expectedResult)
  }

  //FIXME: Slice fails at runtime if indices are wrong sizes
  //TODO
  //Will work when Slice is fixed
  //Still broken, but bounds on squeeze below are working
 /* 
  "Tensor should fail to compile slice when indices out of range" in {
    val arr = Tensor(Array(1, 2, 3, 4),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 4 #: SNil )
    assertTypeError("arr.slice[TT, 0 ::: INil, 11 ::: INil]")
  }

  "Tensor should fail to compile slice when indices negative" in {
    val arr = Tensor(Array(1, 2, 3, 4),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 4 #: SNil )
    assertTypeError("arr.slice[TT, -11 ::: INil, 3 ::: INil]")
  }
*/
  "Tensor should squeeze" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val expectedResult = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: SNil)
    doAssert((arr.squeeze[TT, 0 ::: INil]) ==== expectedResult)
  }
 
  "Tensor should fail to compile squeeze when indices out of range" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    assertTypeError("arr.squeeze[TT, 8 ::: INil]")
  }

  "Tensor should unsqueeze" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 2 #: SNil)
    val expectedResult = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val result = arr.unsqueeze[TT, 0 ::: INil]
    doAssert((result) ==== expectedResult)
  }

  "Tensor should pad" in {
    val arr = Tensor(Array(1, 2, 3, 4),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 4 #: SNil )
    val padValue = 42
    val expectedResult = Tensor(Array(42, 1, 2, 3, 4, 42),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 6 #: SNil)

    doAssert((arr.pad[TT, 1 #: SNil, 1 #: SNil](42)) ==== expectedResult)
  }

  "Tensor should gather" in {
    val arr = Tensor(Array(1, 2, 3, 4),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 4 #: SNil )
    val expectedResult = Tensor(Array(2, 4),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: SNil)

    doAssert((arr.gather[TT, TD, 0 ::: INil, 1 ::: 3 ::: INil]) ==== expectedResult)
  }

  "Tensor should flatten" in {
    val arr = Tensor(Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 2 #: 2 #: 2 #: SNil )
    val expectedResult = Tensor(Array(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 8 #: SNil)
    val res: Tensor[Int, Tuple3[TT, TD, 2 #: 8 #: SNil]] = arr.flatten_[TT, 1 ::: INil]

    doAssert((res) ==== expectedResult)
  }

  "Tensor should tile" in {
    val arr = Tensor(Array(1, 2, 3, 4),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 4 #: SNil )
    val expectedResult = Tensor(Array(1, 2, 3, 4, 1, 2, 3, 4),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 8 #: SNil)
    doAssert((arr.tile[TT, 2 ::: INil]) ==== expectedResult)
  }

  "Tensor should rank" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)

    arr.rank.asserting(_ shouldBe 2)
  }

  "Tensor should gemm" in {
    val arr = Tensor(Array(42.0, 84.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val other = Tensor(Array(42.0, 84.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 1 #: SNil)
    val bias = Tensor(Array(42.0), valueOf[TT], "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: SNil)
    val expectedResult = Tensor(Array(8862.886347055435), valueOf[TT], "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: SNil)
  
    val in = arr.gemm(other, Some(bias))

    val result = in ==== expectedResult
    doAssert(result)
  }

  "Tensor should lrn" in {
    //NCHW tensor, 3 channels, 1 pixel
    val arr = Tensor(Array(-1.0f, 0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: 1 #: 1 #: SNil)
    val expectedResult = Tensor(Array(-0.9999f, 0.0f, -0.9999f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: 1 #: 1 #: SNil)

    val in = arr.lrn(size=3)

    doAssert(in ==== expectedResult)
  }

  "Tensor should conv" in {
    //NCHW tensor, 1 channel, 4 pixels
    val arr = Tensor(Array(-1.0f, 0.0f, 1.0f, 2.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 4 #: SNil)
    val kernelShape = 1 #: 2 #: SNil
    val w = Tensor(Array(1.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 2 #: SNil)
    val expectedResult = Tensor(Array(-1.0f, 1.0f, 3.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 3 #: SNil)
    val result: Tensor[Float, Tuple3["TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 3 #: SNil]] = arr.conv(kernelShape, w)

    doAssert((result) ==== expectedResult)
  }

  "Tensor should global average pool" in {
    //NCHW tensor, 1 channel, 4 pixels
    val arr = Tensor(Array(-1.0f, 0.0f, 1.0f, 2.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 4 #: SNil)
    val expectedResult = Tensor(Array(0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 1 #: SNil)
    val result: Tensor[Float, Tuple3["TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 1 #: SNil]] = arr.globalAveragePool()
    doAssert((result) ==== expectedResult)
  }

  "Tensor should average pool" in {
    //NCHW tensor, 1 channel, 4 pixels
    val arr = Tensor(Array(-1.0f, 0.0f, 1.0f, 2.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 4 #: SNil)
    val kernelShape = 1 #: 2 #: SNil
    val expectedResult = Tensor(Array(-0.5f,0.5f,1.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 3 #: SNil)
    val result: Tensor[Float, Tuple3["TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 3 #: SNil]] = arr.averagePool(kernelShape)

    doAssert((result) ==== expectedResult)
  }

  "Tensor should average pool w/ pads" in {
    //NCHW tensor, 1 channel, 4 pixels
    val arr = Tensor(Array(-1.0f, 0.0f, 1.0f, 2.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 4 #: SNil)
    val kernelShape = 1 #: 2 #: SNil
    val expectedResult = Tensor(Array(-1.0f,-0.5f,0.5f,1.5f,2.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 5 #: SNil)
    val result: Tensor[Float, Tuple3["TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 5 #: SNil]] = arr.averagePool(kernelShape, 0 #: 1 #: SNil, 0 #: 1 #: SNil)
    doAssert((result) ==== expectedResult)
  }

  "Tensor should global max pool" in {
    //NCHW tensor, 1 channel, 4 pixels
    val arr = Tensor(Array(-1.0f, 0.0f, 1.0f, 2.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 4 #: SNil)
    val expectedResult = Tensor(Array(2.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 1 #: SNil)
    val result: Tensor[Float, Tuple3["TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 1 #: SNil]] = arr.globalMaxPool()
    doAssert((result) ==== expectedResult)
  }

  "Tensor should max pool" in {
    //NCHW tensor, 1 channels, 4 pixels
    val arr = Tensor(Array(-1.0f, 0.0f, 1.0f, 2.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 4 #: SNil)
    val kernelShape = 1 #: 2 #: SNil
    val expectedResult = Tensor(Array(0.0f,1.0f,2.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 3 #: SNil)
    val result: Tensor[Float, Tuple3["TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 3 #: SNil]] = arr.maxPool(kernelShape)

    doAssert((result) ==== expectedResult)
  }

  "Tensor should max pool w/ pads" in {
    //NCHW tensor, 1 channel, 4 pixels
    val arr = Tensor(Array(-1.0f, 0.0f, 1.0f, 2.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 4 #: SNil)
    val kernelShape = 1 #: 2 #: SNil
    val expectedResult = Tensor(Array(-1.0f,0.0f,1.0f,2.0f,2.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 5 #: SNil)
    val result: Tensor[Float, Tuple3["TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: 1 #: 5 #: SNil]] = arr.maxPool(kernelShape, 0 #: 1 #: SNil, 0 #: 1 #: SNil)
    doAssert((result) ==== expectedResult)
  }

  "Tensor should instance normalization" in {
    //NCHW tensor, 2 channels, height 1, width 3
    val arr = Tensor(Array(-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: 1 #: 3 #: SNil)
    val scale = Tensor(Array(1.0f, 1.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 2 #: SNil)
    val bias = Tensor(Array(0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 2 #: SNil)
    val expectedResult = Tensor(Array(-1.2247356f, 0.0f, 1.2247356f, -0.8371035f, 1.0f, 2.8371034f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: 1 #: 3 #: SNil)
    doAssert((arr.instanceNormalization(scale=scale, b=bias)) ==== expectedResult)
  }

  "Tensor should batch normalization" in {
    //NCHW tensor, batch 2, channels 2, height 1, width 3
    val arr = Tensor(Array(-1.0f, 0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 1.0f, 0.0f, -1.0f, -2.0f, -3.0f, -4.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 2 #: 2 #: 1 #: 3 #: SNil)
    val scale = Tensor(Array(1.0f, 1.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 2 #: SNil)
    val bias = Tensor(Array(0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 2 #: SNil)
    val mean = Tensor(Array(0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 2 #: SNil)
    val variance = Tensor(Array(0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 2 #: SNil)

    val expectedResult = Tensor(Array(-316.22778f, 0.0f, 316.22778f, 2.4999924f, 3.9999852f, 5.4999776f, 316.22778f, 0.0f, -316.22778f, -3.4999776f, -4.99997f, -6.4999623f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 2 #: 2 #: 1 #: 3 #: SNil)
    doAssert((arr.batchnorm(scale=scale, b=bias, mean = mean, someVar = variance)) ==== expectedResult)
  }

  "Tensor should softmax" in {
    val arr = Tensor(Array(-1.0f, 0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    val expectedResult = Tensor(Array(0.09003057f, 0.24472848f, 0.66524094f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.softmax()) ==== expectedResult)
  }

  //This is a no-op without training mode
  "Tensor should dropout" in {
    val arr = Tensor(Array(42.0f, 84.0f, 168.0f, 336.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 4 #: SNil)
    val expectedResult = Tensor(Array(42.0f, 84.0f, 168.0f, 336.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 4 #: SNil)
    doAssert((arr.dropout(42, 0.5f)) ==== expectedResult)
  }

  "Tensor should dropout with training mode enabled" in {
    val arr = Tensor(Array(42.0f, 84.0f, 168.0f, 336.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 4 #: SNil)
    val expectedResult = Tensor(Array(0.0f, 84.0f, 168.0f, 0.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 4 #: SNil)
    doAssert((arr.dropout(42, 0.5f, true)) ==== expectedResult)
  }

  "Tensor should clip" in {
    val arr = Tensor(Array(42.0f, 84.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val expectedResult = Tensor(Array(50.0f, 80.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr.clip(50.0f, 80.0f)) ==== expectedResult)
  }

  "Tensor should reciprocal" in {
    val arr = Tensor(Array(42.0f, 84.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val expectedResult = Tensor(Array(0.023809524f, 0.011904762f),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr.reciprocal()) ==== expectedResult)
  }

  "Tensor should unary subtract" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((-arr) ==== Tensor(Array(-42, -84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor should abs" in {
    val arr = Tensor(Array(-42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr.abs()) ==== Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor should ceil" in {
    val arr = Tensor(Array(-1.5f, 1.2f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr.ceil()) ==== Tensor(Array(-1.0f, 2.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor should floor" in {
    val arr = Tensor(Array(-1.5f, 1.2f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr.floor()) ==== Tensor(Array(-2.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor should log" in {
    val arr = Tensor(Array(1.0f, 10.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr.log()) ==== Tensor(Array(0.0f, 2.30258512f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor should exp" in {
    val arr = Tensor(Array(-1.0, 0.0, 1.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    val exp = arr.exp()
    //Tiny difference between CUDA and cpu - maybe fixed in latest CUDA
    doAssert((exp) ==== Tensor(Array(0.3678794411714423, 1.0, 2.718281828459045),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should concat" in {
    val arr = Tensor(Array(1.0, 4.0, 9.0, 8.0, 6.0, 7.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 2 #: 3 #: SNil)
    val arrB = Tensor(Array(2.0, 3.0, 4.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)

    val tensors = (arr, arrB)
    val result = tensors.concat[0 ::: INil]
    doAssert(result ==== Tensor(Array(1.0, 4.0, 9.0, 8.0, 6.0, 7.0, 2.0, 3.0, 4.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: "TensorShapeDenotation" ##: TSNil, 3 #: 3 #: SNil))
  }

  "Tensor should sum" in {
    val arr = Tensor(Array(1.0f, 4.0f, 9.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    val arrB = Tensor(Array(3.0f, 2.0f, 3.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil) 
    val arrSeq: Seq[Tensor[Float, ("TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)]] = Seq(arr,arrB)
    doAssert(Seq(arr, arrB).ndsum() ==== Tensor(Array(4.0f, 6.0f, 12.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)) 
  }

  "Tensor should mean" in {
    val arr = Tensor(Array(1.0f, 4.0f, 9.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    val arrB = Tensor(Array(3.0f, 2.0f, 3.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    val arrSeq: Seq[Tensor[Float, ("TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)]] = Seq(arr,arrB)
    doAssert(Seq(arr, arrB).mean() ==== Tensor(Array(2.0f, 3.0f, 6.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should sign" in {
    val arr = Tensor(Array(-0.5f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert(arr.sign() ==== Tensor(Array(-1.0f, 0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should isNaN" in {
    val arr = Tensor(Array(-0.5f, 0.0f, java.lang.Float.NaN),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert(arr.isNaN() ==== Tensor(Array(false, false, true),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should not" in {
    val arr = Tensor(Array(true, false, true),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert(!arr ==== Tensor(Array(false, true, false),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should and" in {
    val arr = Tensor(Array(true, false, true),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    val arrB = Tensor(Array(true, false, false),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert(arr && arrB ==== Tensor(Array(true, false, false),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should or" in {
    val arr = Tensor(Array(true, false, true),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    val arrB = Tensor(Array(true, false, false),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert(arr || arrB ==== Tensor(Array(true, false, true),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should xor" in {
    val arr = Tensor(Array(true, false, true),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    val arrB = Tensor(Array(true, false, false),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert(arr ^ arrB ==== Tensor(Array(false, false, true),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should sqrt" in {
    val arr = Tensor(Array(1.0, 4.0, 9.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.sqrt()) ==== Tensor(Array(1.0, 2.0, 3.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should cos" in {
    val arr = Tensor(Array(-1.0f, 0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.cos()) ==== Tensor(Array(0.5403023f, 1.0f, 0.5403023f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should sin" in {
    val arr = Tensor(Array(-1.0f, 0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.sin()) ==== Tensor(Array(-0.84147096f, 0.0f, 0.84147096f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should tan" in {
    val arr = Tensor(Array(-1.0f, 0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.tan()) ==== Tensor(Array(-1.5574077f, 0.0f, 1.5574077f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should tanh" in {
    val arr = Tensor(Array(-1.0f, 0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.tanh()) ==== Tensor(Array(-0.7615942f, 0.0f, 0.7615942f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should acos" in {
    val arr = Tensor(Array(-1.0f, 0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.acos()) ==== Tensor(Array(3.1415927f, 1.5707964f, 0.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should asin" in {
    val arr = Tensor(Array(-1.0f, 0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.asin()) ==== Tensor(Array(-1.5707964f, 0.0f, 1.5707964f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should atan" in {
    val arr = Tensor(Array(-1.0f, 0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.atan()) ==== Tensor(Array(-0.7853982f, 0.0f, 0.7853982f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should atanh" in {
    val arr = Tensor(Array(-0.5f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.atanh()) ==== Tensor(Array(-0.54930615f, 0.0f, 0.54930615f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should sigmoid" in {
    val arr = Tensor(Array(-1.0f, 0.0f, 1.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.sigmoid()) ==== Tensor(Array(0.2689414213699951f, 0.5f, 0.7310585786300049f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should relu" in {
    val arr = Tensor(Array(-0.5f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.relu()) ==== Tensor(Array(0.0f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should leaky relu" in {
    val arr = Tensor(Array(-0.5f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.leakyRelu()) ==== Tensor(Array(-0.005f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should celu" in {
    val arr = Tensor(Array(-0.5f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)

    val in = arr.celu()
    doAssert(in ==== Tensor(Array(-0.39347833f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should celu with alpha" in {
    val arr = Tensor(Array(-0.5f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)

    val in = arr.celu(2.1f)
    doAssert(in ==== Tensor(Array(-0.4449320137500763f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should elu" in {
    val arr = Tensor(Array(-0.5f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)

    val in = arr.elu()

    doAssert(in ==== Tensor(Array(-0.39350867f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should elu with alpha" in {
    val arr = Tensor(Array(-0.5f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    val in = arr.elu(2.1f)
    doAssert(in ==== Tensor(Array(-0.8262855410575867f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should selu" in {
    val arr = Tensor(Array(-0.5f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.selu()) ==== Tensor(Array(-0.6917561f, 0.0f, 0.52535f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should selu with alpha and gamma" in {
    val arr = Tensor(Array(-0.5f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    val in = arr.selu(2.1f, 3.1f)
    doAssert(in ==== Tensor(Array(-2.5614850521087646f, 0.0f, 1.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should prelu" in {
    val arr = Tensor(Array(-0.5f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    val slopeArr = Tensor(Array(-0.5f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil)
    doAssert((arr.prelu(slopeArr)) ==== Tensor(Array(0.25f, 0.0f, 0.5f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 3 #: SNil))
  }

  "Tensor should pow" in {
    val arr = Tensor(Array(42.0, 84.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr ** Tensor(Array(2.0, 2.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)) ==== Tensor(Array(1764.0, 7056.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor should mod" in {
   val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr % Tensor(Array(40, 40),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)) ==== Tensor(Array(2,4),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  //TODO: Fix: Multiple asserts need to be chained together in IO
  "Tensor should gt" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val result = (arr > Tensor(Array(42, 80),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))


    result.data.asserting(_(0) shouldBe false)
    result.data.asserting(_(1) shouldBe true)
  }

  "Tensor should gte" in {
    val arr = Tensor(Array(42l, 84l),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val result = (arr >= Tensor(Array(42l, 80l),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))

    result.data.asserting(_(0) shouldBe true)
    result.data.asserting(_(1) shouldBe true)
  }

  "Tensor should lt" in {
    val arr = Tensor(Array(42, 84),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val result = (arr < Tensor(Array(42, 80),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  
    result.data.asserting(_(0) shouldBe false)
    result.data.asserting(_(1) shouldBe false)
  }

  "Tensor should lte" in {
    val arr = Tensor(Array(42l, 84l),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val result = (arr <= Tensor(Array(42l, 80l),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))

    result.data.asserting(_(0) shouldBe true)
    result.data.asserting(_(1) shouldBe false)
  }

  "Tensor should max" in {
    val arr = Tensor(Array(42.0f, 84.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val other = Tensor(Array(50.0f, 80.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr max other) ==== Tensor(Array(50.0f, 84.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor should fail to compile max if shapes don't match" in {
    val arr = Tensor(Array(42.0f, 84.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val other = Tensor(Array(50.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: SNil)
    assertTypeError("arr max other")
  }

  "Tensor should min" in {
    val arr = Tensor(Array(42.0, 84.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val other = Tensor(Array(50.0, 80.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    doAssert((arr min other) ==== Tensor(Array(42.0, 80.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil))
  }

  "Tensor should fail to compile min if shapes don't match" in {
    val arr = Tensor(Array(42.0f, 84.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val other = Tensor(Array(50.0f),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: SNil)
    assertTypeError("arr min other")
  }

  "Tensor should matmul" in {
    val arr = Tensor(Array(42.0, 84.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    val other = Tensor(Array(42.0, 84.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 2 #: 1 #: SNil)
    val expectedResult = Tensor(Array(8820.0), valueOf[TT], "TensorShapeDenotation" ##: TSNil, 1 #: 1 #: SNil)

    val res = arr.matmul(other)

    val result = res ==== expectedResult
    doAssert(result)
  }

  "Tensor should not matmul if dimensions don't match" in {
    val arr = Tensor(Array(42.0, 84.0),"TensorTypeDenotation", "TensorShapeDenotation" ##: TSNil, 1 #: 2 #: SNil)
    assertTypeError("arr.matmul(other)")
  }

}
