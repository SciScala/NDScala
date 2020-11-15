package org.sciscala.ndscala 

import scala.util.Random

import scala.reflect.ClassTag
import spire.random.Dist
import spire.math.Numeric
import spire.implicits._
import scala.language.implicitConversions
import org.sciscala.ndscala.union._

import ai.djl._
import ai.djl.ndarray._
import ai.djl.ndarray.types.{Shape => DJLShape}
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.index.dim.NDIndexSlice
import ai.djl.ndarray.types.DataType
//import ai.djl.nn._
//import ai.djl.nn.core._
import org.emergentorder.onnx.Tensors.Axes
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.Tensors.Tensor._
import org.emergentorder.compiletime._
import io.kjaer.compiletime._
import io.kjaer.compiletime.Shape._

object DJLOps {

  //possible leak
  val manager = ai.djl.ndarray.NDManager.newBaseManager()

  //bug where toArray only returns Number in DJL
  //No enforcement of DType
  type DJLNDArray[DType <: AllSupported, Ax <: Axes] =NDArray
  //  implicit def convert[DType: ClassTag](d: DType): DJLNDArray[DType] = DJLNDArray(d)
  
  implicit def toDJLNDArray[DType <: AllSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](t: Tensor[DType, (Tt,Td,S)]): DJLNDArray[DType, (Tt,Td,S)] = t.data match {
     case arr: Array[DType] => arr match {
      case i: Array[Int] => manager.create(arr, new DJLShape(t.shape.map(_.toLong): _*))
      case l: Array[Long] => manager.create(arr, new DJLShape(t.shape.map(_.toLong): _*))
      case f: Array[Float] => manager.create(arr, new DJLShape(t.shape.map(_.toLong): _*))
      case d: Array[Double] => manager.create(arr, new DJLShape(t.shape.map(_.toLong): _*))
      case b: Array[Boolean]=> manager.create(arr, new DJLShape(t.shape.map(_.toLong): _*))
     }
  }
} 
  //returns array of numbers
  //TODO: same thing, match on first element type
  //for now just use Axesi
  /*
  implicit def fromDJLNDArray[DType <: AllSupported : ClassTag, Ax <: Axes](t: DJLNDArray[DType, Ax]): (Array[DType], Ax) = {
    val shape = Array(t.getShape.getShape.toArray: _*).map(x => x.toInt)

    t.getDataType.ordinal match{
    case 4 => {
      val tens = (Array(t.toIntArray: _*).asInstanceOf[Array[DType]] , A)
      (tens.data, tens._2)
    }
    case 6 => {
      val tens = (Array(t.toLongArray: _*).asInstanceOf[Array[DType]] , shape)
      (tens.data, tens._2)
    } 
    case 0 => {
      val tens = (Array(t.toFloatArray: _*).asInstanceOf[Array[DType]] , shape)
      (tens.data, tens._2)
    }
    case 1 => {
      val tens = (Array(t.toDoubleArray: _*).asInstanceOf[Array[DType]] , shape)
      (tens.data, tens._2)
    }
    case 7 => {
      val tens = (Array(t.toBooleanArray: _*).asInstanceOf[Array[DType]] , shape)
      (tens.data, tens._2)
    }
    }

  }
}
*/
//class DJLNDArray[DType](ndarray: DJLNDArray) {
//  require(ndarray.getDataType() =
//}

given NDArrayOps[DJLOps.DJLNDArray]{
import DJLOps._ 
  //Nullary / factory ops
def zeros[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] =  manager.zeros(new DJLShape(s.value.toSeq.map(_.toLong): _*))
  def ones[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](shape: Array[Int])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = manager.ones(new DJLShape(shape.map(_.toLong): _*))
  def full[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](shape: Array[Int], value: DType)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)]  = ??? // manager.full(new DJLShape(shape: _*), value)

  //TOFIX
//  def rand[DType <: Supported : ClassTag: Numeric](shape: Array[Int]): DJLNDArray[DType]

  //Unary ops
  //def reshape[DType <: Supported : ClassTag: Numeric](arr: DJLNDArray[DType], newShape: Array[Int]): DJLNDArray[DType]
    //Unary ops
//  def reshape[DType <: Supported : ClassTag: Numeric](arr: DJLNDArray[DType], newShape: Array[Int])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType] 
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def reShape(newShape: Array[Int])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S],tt1: ValueOf[Tt1], td1: TensorShapeDenotationOf[Td1], s1: ShapeOf[S1], sizeSeq: NumElements[S] =:= NumElements[S1]): DJLNDArray[DType, (Tt1,Td1,S1)] = arr.reshape(newShape.map(_.toLong): _*)

    extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def transpose(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[org.emergentorder.compiletime.TensorShapeDenotation.Reverse[Td]], s: ShapeOf[io.kjaer.compiletime.Shape.Reverse[S]]): DJLNDArray[DType, (Tt,org.emergentorder.compiletime.TensorShapeDenotation.Reverse[Td],io.kjaer.compiletime.Shape.Reverse[S])] = (arr:NDArray).transpose()

  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def transpose(axes: Array[Int], dummy: Option[Boolean])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[org.emergentorder.compiletime.TensorShapeDenotation.Reverse[Td]], s: ShapeOf[io.kjaer.compiletime.Shape.Reverse[S]]): DJLNDArray[DType, (Tt,org.emergentorder.compiletime.TensorShapeDenotation.Reverse[Td],io.kjaer.compiletime.Shape.Reverse[S])] = arr.transpose(axes: _*)

  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def round()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr.round

  //TODO: broaden slice, extra sugar for slice, range, squeeze, ...
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def slice(start: Int, end: Int, dummy: Option[Boolean])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S],tt1: ValueOf[Tt1], td1: TensorShapeDenotationOf[Td1], s1: ShapeOf[S1]): DJLNDArray[DType, (Tt1,Td1,S1)] = arr.get((new NDIndex()).addSliceDim(start, end))

  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def squeeze(index: Array[Int], dummy: Option[Boolean])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S],tt1: ValueOf[Tt1], td1: TensorShapeDenotationOf[Td1], s1: ShapeOf[S1]): DJLNDArray[DType, (Tt1,Td1,S1)] = arr.squeeze(index.toArray)
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def rank: Int = arr.getShape.getShape.size

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def unary_- (using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr.neg
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def abs()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr.abs
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def ceil()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr. ceil
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def floor()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr.floor

  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Shape](arr: Seq[DJLNDArray[DType, (Tt,Td,S)]]) def concat (axis: Int)(using tt: ValueOf[Tt1], td: TensorShapeDenotationOf[Td1], s: ShapeOf[S1]): DJLNDArray[DType, (Tt1,Td1,S1)] = ???
 //TODO: reduceMean
//  def mean[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported](arr: Seq[DJLNDArray[DType]]): DJLNDArray[DType]
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def log()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr.log
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def exp()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr.exp
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def sqrt()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr.sqrt
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def cos()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr.cos
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def cosh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr.cosh
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def sin()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr.sin
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def sinh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr.sinh
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def tan()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr.tan
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def tanh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr.tanh
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def acos()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr.acos
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def acosh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr.acosh
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def asin()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr.asin
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def asinh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr.asinh
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def atan()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr.atan
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def atanh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr.atanh
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def sigmoid()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = ai.djl.nn.Activation.sigmoid(arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def relu()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = ai.djl.nn.Activation.relu(arr)
  

  //Binary DJLNDArray ops
 extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def +(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr add other
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def -(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr sub other
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def *(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr mul other
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def **(d: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr pow d 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def /(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr div other
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def %(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr mod other

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def >(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[Boolean, (Tt,Td,S)] = arr gt other
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def >=(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[Boolean, (Tt,Td,S)] = arr gte other
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def <(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[Boolean, (Tt,Td,S)] = arr lt other
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def <=(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[Boolean, (Tt,Td,S)] = arr lte other
  //Restricted to numeric only because of TF
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def ====(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[Boolean, (Tt,Td,S)] = arr eq other
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def !===(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[Boolean, (Tt,Td,S)] = arr neq other

  //TF-scala conflicts with max and min
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def max(d: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr maximum d
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def min(d: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = arr minimum d

  //DJL-mxnet only actually supports Float here (gpu only?)
    extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Dim0 <: Dimension, Dim1 <: Dimension, Dim2 <: Dimension, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Dim0 #: Dim1 #:SNil, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Dim1 #: Dim2 #: SNil] (arr: DJLNDArray[DType, (Tt,Td,S)]) def matmul(other: DJLNDArray[DType, (Tt1,Td1,S1)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s2: ShapeOf[Dim0 #: Dim2 #: SNil]): DJLNDArray[DType, (Tt,Td,Dim0 #: Dim2 #: SNil)] = arr dot other
  
}
