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
  opaque type DJLNDArray[DType <: AllSupported, Ax <: Axes] =NDArray

  implicit def wrap[DType <: AllSupported, Ax <: Axes](arr:NDArray): DJLNDArray[DType, Ax] = arr
  implicit def unwrap[DType <: AllSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]): NDArray = arr
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
 
  //returns array of numbers
  //TODO: same thing, match on first element type
  //for now just use Axesi
 
  def fromDJLNDArray[DType <: AllSupported : ClassTag, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](t: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = {
    val shape = Array(t.getShape.getShape.toArray: _*).map(x => x.toInt)

    t.getDataType.ordinal match{
    case 4 => {
      val tens = Tensor(Array(t.toIntArray: _*).asInstanceOf[Array[DType]], tt.value, td.value, s.value)
      tens 
    }
    case 6 => {
      val tens = Tensor(Array(t.toLongArray: _*).asInstanceOf[Array[DType]], tt.value, td.value, s.value)
      tens 
    } 
    case 0 => {
      val tens = Tensor(Array(t.toFloatArray: _*).asInstanceOf[Array[DType]], tt.value, td.value, s.value)
      tens 
    }
    case 1 => {
      val tens = Tensor(Array(t.toDoubleArray: _*).asInstanceOf[Array[DType]], tt.value, td.value, s.value)
      tens 
    }
    case 7 => {
      val tens = Tensor(Array(t.toBooleanArray: _*).asInstanceOf[Array[DType]], tt.value, td.value, s.value)
      tens
    }
    }

  }  
}

//class DJLNDArray[DType](ndarray: DJLNDArray) {
//  require(ndarray.getDataType() =
//}

given NDArrayOps[DJLOps.DJLNDArray] with {
import DJLOps._ 
  //Nullary / factory ops
def zeros[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] =  wrap(manager.zeros(new DJLShape(s.value.toSeq.map(_.toLong): _*)))
  def ones[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](shape: Array[Int])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(manager.ones(new DJLShape(shape.map(_.toLong): _*)))
  def full[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](shape: Array[Int], value: DType)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)]  = ??? // manager.full(new DJLShape(shape: _*), value)

  //TOFIX
//  def rand[DType <: Supported : ClassTag: Numeric](shape: Array[Int]): DJLNDArray[DType]

  //Unary ops
  //def reshape[DType <: Supported : ClassTag: Numeric](arr: DJLNDArray[DType], newShape: Array[Int]): DJLNDArray[DType]
    //Unary ops
//  def reshape[DType <: Supported : ClassTag: Numeric](arr: DJLNDArray[DType], newShape: Array[Int])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType] 
extension [DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def reshape[Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Shape](using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S],tt1: ValueOf[Tt1], td1: TensorShapeDenotationOf[Td1], s1: ShapeOf[S1], sizeSeq: NumElements[S] =:= NumElements[S1]): DJLNDArray[DType, (Tt1,Td1,S1)] = wrap((arr: NDArray).reshape(shapeOf[S1].toSeq.toArray.map(x => x.toLong): _*)) //arr.reshape(1) //shapeOf[S1].toSeq.toArray)

    extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def transpose(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[org.emergentorder.compiletime.TensorShapeDenotation.Reverse[Td]], s: ShapeOf[io.kjaer.compiletime.Shape.Reverse[S]]): DJLNDArray[DType, (Tt,org.emergentorder.compiletime.TensorShapeDenotation.Reverse[Td],io.kjaer.compiletime.Shape.Reverse[S])] = wrap((arr).transpose())

    //Forced to take only the first here
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def transpose(axes: Array[Int])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[org.emergentorder.compiletime.TensorShapeDenotation.Reverse[Td]], s: ShapeOf[io.kjaer.compiletime.Shape.Reverse[S]]): DJLNDArray[DType, (Tt,org.emergentorder.compiletime.TensorShapeDenotation.Reverse[Td],io.kjaer.compiletime.Shape.Reverse[S])] = wrap((arr:NDArray).transpose(axes(0)))

  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def round()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.round)

  //TODO: broaden slice, extra sugar for slice, range, squeeze, ...
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation,S1D1 <: Dimension, S1 <: S1D1 #: SNil] (arr: DJLNDArray[DType, (Tt,Td,S)]) def slice[Tt2 <: TensorTypeDenotation, Td2 <: TensorShapeDenotation, S2 <: Shape](start: DJLNDArray[Int, (Tt1,Td1, S1)], end: DJLNDArray[Int, (Tt1,Td1, S1)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S], tt2: ValueOf[Tt2], td2: TensorShapeDenotationOf[Td2], s2: ShapeOf[S2]): DJLNDArray[DType, (Tt2,Td2,S2)] = ??? //wrap(arr.get((new NDIndex()).addSliceDim(start, end)))

  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def squeeze[Tt1 <: TensorTypeDenotation, Axes <: Indices](using tt: ValueOf[Tt1], td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td,Axes,false]], s: ShapeOf[KeepOrReduceDims[S,Axes,false]], i: IndicesOf[Axes]): DJLNDArray[DType, Tuple3[Tt1,KeepOrReduceDimDenotations[Td,Axes,false],KeepOrReduceDims[S,Axes,false]]] = wrap(unwrap(arr).squeeze(indicesOf[Axes].indices.toArray))
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def rank: Int = arr.getShape.getShape.size

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def unary_- (using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.neg)
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def abs()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.abs)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def ceil()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.ceil)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def floor()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.floor)


 //TODO: reduceMean
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Seq[DJLNDArray[DType, (Tt,Td,S)]]) def mean()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = ???
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def log()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.log)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def exp()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.exp)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def sqrt()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.sqrt)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def cos()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.cos)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def cosh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.cosh)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def sin()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.sin)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def sinh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.sinh)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def tan()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.tan)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def tanh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.tanh)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def acos()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.acos)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def acosh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.acosh)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def asin()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.asin)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def asinh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.asinh)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def atan()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.atan)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def atanh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr.atanh)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def sigmoid()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(ai.djl.nn.Activation.sigmoid(arr))
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def relu()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(ai.djl.nn.Activation.relu(arr))
  
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def reduceSum[Tt1 <: TensorTypeDenotation, Axes <: Indices, KeepDims <: (Boolean&Singleton)](using tt: ValueOf[Tt1], td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td,Axes,KeepDims]], s: ShapeOf[KeepOrReduceDims[S,Axes,KeepDims]], i: IndicesOf[Axes], k: ValueOf[KeepDims]): DJLNDArray[DType, Tuple3[Tt1,KeepOrReduceDimDenotations[Td,Axes,KeepDims],KeepOrReduceDims[S,Axes,KeepDims]]] = ???

  //Binary DJLNDArray ops
 extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def +(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr add other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def -(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr sub other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def *(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr mul other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def **(d: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr pow d)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def /(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr div other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def %(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr mod other)

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def >(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[Boolean, (Tt,Td,S)] = wrap(arr gt other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def >=(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[Boolean, (Tt,Td,S)] = wrap(arr gte other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def <(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[Boolean, (Tt,Td,S)] = wrap(arr lt other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def <=(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[Boolean, (Tt,Td,S)] = wrap(arr lte other)
  //Restricted to numeric only because of TF
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def ====(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[Boolean, (Tt,Td,S)] = wrap(arr eq other)
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def !===(other: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[Boolean, (Tt,Td,S)] = wrap(arr neq other)

  //TF-scala conflicts with max and min
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def max(d: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr maximum d)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: DJLNDArray[DType, (Tt,Td,S)]) def min(d: DJLNDArray[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): DJLNDArray[DType, (Tt,Td,S)] = wrap(arr minimum d)

  //DJL-mxnet only actually supports Float here (gpu only?)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Dim0 <: Dimension, Dim1 <: Dimension, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Dim0 #: Dim1 #:SNil] (arr: DJLNDArray[DType, (Tt,Td,S)]) def matmul[Dim2 <: Dimension, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Dim1 #: Dim2 #: SNil](other: DJLNDArray[DType, (Tt1,Td1,S1)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td],vd:ValueOf[scala.compiletime.S[Dim0]], vd1:ValueOf[scala.compiletime.S[Dim1]], vd2: ValueOf[scala.compiletime.S[Dim2]], s2: ShapeOf[Dim0 #: Dim2 #: SNil]): DJLNDArray[DType, (Tt,Td,Dim0 #: Dim2 #: SNil)] = wrap(arr matMul other)
  
}
