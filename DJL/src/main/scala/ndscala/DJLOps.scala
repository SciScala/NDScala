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
import ai.djl.ndarray.types.Shape
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.index.dim.NDIndexSlice
import ai.djl.ndarray.types.DataType
//import ai.djl.nn._
//import ai.djl.nn.core._

//TODO: fix for shape safety
//TODO: Typeful wrapper for DJL's DJLNDArray
object DJLOps {

  //possible leak
  val manager = NDManager.newBaseManager()

  //bug where toArray only returns Number in DJL
  //No enforcement of DType
  type DJLNDArray[DType] =NDArray
  //  implicit def convert[DType: ClassTag](d: DType): DJLNDArray[DType] = DJLNDArray(d) 
  implicit def toDJLNDArray[DType : ClassTag](t: (Array[DType], Array[Int])): DJLNDArray[DType] = t._1 match {
     case arr: Array[DType] => arr.head match {
      case i: Integer => manager.create(arr.asInstanceOf[Array[Int]], new Shape(t._2.map(_.toLong): _*))
      case l: Long => manager.create(arr.asInstanceOf[Array[Long]], new Shape(t._2.map(_.toLong): _*))
      case f: Float => manager.create(arr.asInstanceOf[Array[Float]], new Shape(t._2.map(_.toLong): _*))
      case d: Double => manager.create(arr.asInstanceOf[Array[Double]], new Shape(t._2.map(_.toLong): _*))
      case b: Boolean=> manager.create(arr.asInstanceOf[Array[Boolean]], new Shape(t._2.map(_.toLong): _*))
     }
  }
  //returns array of numbers
  //TODO: same thing, match on first element type
  implicit def fromDJLNDArray[DType <: AllSupported : ClassTag](t: DJLNDArray[DType]): (Array[DType], Array[Int]) = {
    val shape = Array(t.getShape.getShape.toArray: _*).map(x => x.toInt)

    t.getDataType.ordinal match{
    case 4 => (Array(t.toIntArray: _*).asInstanceOf[Array[DType]] , shape) 
    case 6 => (Array(t.toLongArray: _*).asInstanceOf[Array[DType]] , shape) 
    case 0 => (Array(t.toFloatArray: _*).asInstanceOf[Array[DType]] , shape) 
    case 1 => (Array(t.toDoubleArray: _*).asInstanceOf[Array[DType]] , shape)
    case 7 => (Array(t.toBooleanArray: _*).asInstanceOf[Array[DType]] , shape)
    }

  }
}

//class DJLNDArray[DType](ndarray: DJLNDArray) {
//  require(ndarray.getDataType() =
//}

given NDArrayOps[DJLOps.DJLNDArray]{
import DJLOps._ 
  //Nullary / factory ops
  def zeros[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](shape: Array[Int]): DJLNDArray[DType] = manager.zeros(new Shape(shape.map(_.toLong): _*))
  def ones[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](shape: Array[Int]): DJLNDArray[DType] = manager.ones(new Shape(shape.map(_.toLong): _*))
  def full[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](shape: Array[Int], value: DType): DJLNDArray[DType] = ??? // manager.full(new Shape(shape: _*), value)

  
  //TODO: fix rand
//  def rand[DType : ClassTag: Numeric: IsSupported](shape: Array[Int]): DJLNDArray[DType] = ???

  //Unary ops
  extension[DType <: Supported : ClassTag: IsSupported](arr: DJLNDArray[DType]) def reShape (newShape: Array[Int]): DJLNDArray[DType] = arr.reshape(newShape.map(_.toLong): _*)
  extension[DType <: Supported : ClassTag: IsSupported](arr: DJLNDArray[DType]) def transpose: DJLNDArray[DType] = arr.transpose()
  extension[DType <: Supported : ClassTag: IsSupported](arr: DJLNDArray[DType]) def transpose (axes: Array[Int], dummy: Option[Boolean]): DJLNDArray[DType] = arr.transpose(axes: _*)
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]) def round(): DJLNDArray[DType]  = arr.round
  //Top-level slice only
  extension[DType <: Supported : ClassTag: IsSupported](arr: DJLNDArray[DType]) def slice(start: Int, end: Int, dummy: Option[Boolean]): DJLNDArray[DType] = arr.get((new NDIndex()).addSliceDim(start, end))

  extension[DType <: Supported : ClassTag: IsSupported](arr: DJLNDArray[DType]) def squeeze(index: Array[Int], dummy: Option[Boolean]): DJLNDArray[DType] = arr.squeeze(index.toArray)

  extension[DType <: Supported : ClassTag: IsSupported](arr: DJLNDArray[DType]) def rank: Int = arr.getShape.getShape.size
//  def clip[DType : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType], min: DType, max: DType): DJLNDArray[DType] = arr.clip(min, max) 

  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](arr: DJLNDArray[DType]) def unary_- : DJLNDArray[DType] = arr.neg 

  extension[DType <: NumericSupported: ClassTag: Numeric: IsNumericSupported](arr: DJLNDArray[DType]) def abs() : DJLNDArray[DType] = arr.abs 
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]) def ceil() : DJLNDArray[DType] = arr.ceil
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]) def floor() : DJLNDArray[DType] = arr.floor 
//  def concat[DType: ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType]*): DJLNDArray[DType] = onnx.Concat11("concat", Some(ndArrayToDJLNDArray(arr)))))
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]) def log() : DJLNDArray[DType]= arr.log 
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]) def exp() : DJLNDArray[DType] = arr.exp
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]) def sqrt() : DJLNDArray[DType] = arr.sqrt
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]) def cos() : DJLNDArray[DType] = arr.cos
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]) def cosh() : DJLNDArray[DType] = arr.cosh
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]) def sin() : DJLNDArray[DType] = arr.sin
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]) def sinh(): DJLNDArray[DType] = arr.sinh
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]) def tan() : DJLNDArray[DType] = arr.tan
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]) def tanh() : DJLNDArray[DType] = arr.tanh
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]) def acos() : DJLNDArray[DType] = arr.acos
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]) def acosh() : DJLNDArray[DType] = arr.acosh
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]) def asin() : DJLNDArray[DType] = arr.asin
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]) def asinh() : DJLNDArray[DType] = arr.asinh
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]) def atan() : DJLNDArray[DType] = arr.atan
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]) def atanh() : DJLNDArray[DType] = arr.atanh

  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: DJLNDArray[DType]) def sigmoid(): DJLNDArray[DType] = ai.djl.nn.Activation.sigmoid(arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported] (arr: DJLNDArray[DType]) def relu(): DJLNDArray[DType] = ai.djl.nn.Activation.relu(arr)

  //Binary DJLNDArray ops

  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](arr: DJLNDArray[DType]) def + (other: DJLNDArray[DType]): DJLNDArray[DType] = arr add other 
  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](arr: DJLNDArray[DType]) def - (other: DJLNDArray[DType]): DJLNDArray[DType] = arr sub other
  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](arr: DJLNDArray[DType]) def * (other: DJLNDArray[DType]): DJLNDArray[DType] = arr mul other 
  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](arr: DJLNDArray[DType]) def ** (other: DJLNDArray[DType]): DJLNDArray[DType] = arr pow other 
  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](arr: DJLNDArray[DType]) def / (other: DJLNDArray[DType]): DJLNDArray[DType] = arr div other
  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](arr: DJLNDArray[DType]) def % (other: DJLNDArray[DType]): DJLNDArray[DType] = arr mod other 

  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](arr: DJLNDArray[DType]) def > (other: DJLNDArray[DType]): DJLNDArray[Boolean] = arr gt other 
  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](arr: DJLNDArray[DType]) def >= (other: DJLNDArray[DType]): DJLNDArray[Boolean] = arr gte other 
  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](arr: DJLNDArray[DType]) def < (other: DJLNDArray[DType]): DJLNDArray[Boolean] = arr lt other 
  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](arr: DJLNDArray[DType]) def <= (other: DJLNDArray[DType]): DJLNDArray[Boolean] = arr lte other 
  
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported] (arr: DJLNDArray[DType]) def ====(other: DJLNDArray[DType]): DJLNDArray[Boolean] = arr eq other
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported] (arr: DJLNDArray[DType]) def !===(other: DJLNDArray[DType]): DJLNDArray[Boolean] = arr neq other

  extension[DType <: NumericSupported: ClassTag: Numeric: IsNumericSupported](arr: DJLNDArray[DType]) def max (d: DJLNDArray[DType]): DJLNDArray[DType] = arr maximum d 
  extension[DType <: NumericSupported: ClassTag: Numeric: IsNumericSupported](arr: DJLNDArray[DType]) def min (d: DJLNDArray[DType]): DJLNDArray[DType] = arr minimum d 

  //DJL-mxnet only actually supports Float here (gpu only?)
  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](arr: DJLNDArray[DType]) def matmul (other: DJLNDArray[DType]): DJLNDArray[DType] = arr dot other
}
