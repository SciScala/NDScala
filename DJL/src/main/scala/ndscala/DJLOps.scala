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
import org.emergentorder.onnx.Tensors.Axes
import org.emergentorder.onnx.Tensors.Tensor._
import org.emergentorder.=!=

object DJLOps {

  //possible leak
  val manager = NDManager.newBaseManager()

  //bug where toArray only returns Number in DJL
  //No enforcement of DType
  type DJLNDArray[DType <: AllSupported, Ax <: Axes] =NDArray
  //  implicit def convert[DType: ClassTag](d: DType): DJLNDArray[DType] = DJLNDArray(d)
  
  implicit def toDJLNDArray[DType <: AllSupported, Ax <: Axes](t: (Array[DType], Ax)): DJLNDArray[DType, Ax] = t._1 match {
     case arr: Array[DType] => arr match {
      case i: Array[Int] => manager.create(arr, new Shape(t.shape.map(_.toLong): _*))
      case l: Array[Long] => manager.create(arr, new Shape(t.shape.map(_.toLong): _*))
      case f: Array[Float] => manager.create(arr, new Shape(t.shape.map(_.toLong): _*))
      case d: Array[Double] => manager.create(arr, new Shape(t.shape.map(_.toLong): _*))
      case b: Array[Boolean]=> manager.create(arr, new Shape(t.shape.map(_.toLong): _*))
     }
  }
 
  //returns array of numbers
  //TODO: same thing, match on first element type
  //for now just use Axes
  implicit def fromDJLNDArray[DType <: AllSupported : ClassTag, Ax <: Axes](t: DJLNDArray[DType, Ax]): (Array[DType], Ax) = {
    val shape = Array(t.getShape.getShape.toArray: _*).map(x => x.toInt)

    t.getDataType.ordinal match{
    case 4 => {
      val tens = create(Array(t.toIntArray: _*).asInstanceOf[Array[DType]] , shape)
      (tens.data, tens._2)
    }
    case 6 => {
      val tens = create(Array(t.toLongArray: _*).asInstanceOf[Array[DType]] , shape)
      (tens.data, tens._2)
    } 
    case 0 => {
      val tens = create(Array(t.toFloatArray: _*).asInstanceOf[Array[DType]] , shape)
      (tens.data, tens._2)
    }
    case 1 => {
      val tens = create(Array(t.toDoubleArray: _*).asInstanceOf[Array[DType]] , shape)
      (tens.data, tens._2)
    }
    case 7 => {
      val tens = create(Array(t.toBooleanArray: _*).asInstanceOf[Array[DType]] , shape)
      (tens.data, tens._2)
    }
    }

  }
}

//class DJLNDArray[DType](ndarray: DJLNDArray) {
//  require(ndarray.getDataType() =
//}

given NDArrayOps[DJLOps.DJLNDArray]{
import DJLOps._ 
  //Nullary / factory ops
  def zeros[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](shape: Array[Int]): DJLNDArray[DType, Axes] = manager.zeros(new Shape(shape.map(_.toLong): _*))
  def ones[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](shape: Array[Int]): DJLNDArray[DType, Axes] = manager.ones(new Shape(shape.map(_.toLong): _*))
  def full[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported](shape: Array[Int], value: DType): DJLNDArray[DType, Axes] = ??? // manager.full(new Shape(shape: _*), value)

  
  //TODO: fix rand
//  def rand[DType : ClassTag: Numeric: IsSupported](shape: Array[Int]): DJLNDArray[DType] = ???

  //Unary ops
  extension[DType <: Supported : ClassTag: IsSupported, Ax <: Axes, Bx <: Axes](arr: DJLNDArray[DType, Ax]) def reShape (newShape: Array[Int]): DJLNDArray[DType, Bx] = arr.reshape(newShape.map(_.toLong): _*)
  extension[DType <: Supported : ClassTag: IsSupported, Ax <: Axes, Bx <: Axes](arr: DJLNDArray[DType, Ax]) def transpose: DJLNDArray[DType, Bx] = arr.transpose()
  extension[DType <: Supported : ClassTag: IsSupported, Ax <: Axes, Bx <: Axes](arr: DJLNDArray[DType, Ax]) def transpose (axes: Array[Int], dummy: Option[Boolean]): DJLNDArray[DType, Bx] = arr.transpose(axes: _*)
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def round(): DJLNDArray[DType, Ax]  = arr.round
  //Top-level slice only
  extension[DType <: Supported : ClassTag: IsSupported, Ax <: Axes, Bx <: Axes](arr: DJLNDArray[DType, Ax]) def slice(start: Int, end: Int, dummy: Option[Boolean]): DJLNDArray[DType, Bx] = arr.get((new NDIndex()).addSliceDim(start, end))

  extension[DType <: Supported : ClassTag: IsSupported, Ax <: Axes, Bx <: Axes](arr: DJLNDArray[DType, Ax]) def squeeze(index: Array[Int], dummy: Option[Boolean]): DJLNDArray[DType, Bx] = arr.squeeze(index.toArray)

  extension[DType <: Supported : ClassTag: IsSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def rank: Int = arr.getShape.getShape.size
//  def clip[DType : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType], min: DType, max: DType): DJLNDArray[DType] = arr.clip(min, max) 

  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def unary_- : DJLNDArray[DType, Ax] = arr.neg 

  extension[DType <: NumericSupported: ClassTag: Numeric: IsNumericSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def abs() : DJLNDArray[DType, Ax] = arr.abs 
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def ceil() : DJLNDArray[DType, Ax] = arr.ceil
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def floor() : DJLNDArray[DType, Ax] = arr.floor 
//  def concat[DType: ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType]*): DJLNDArray[DType] = onnx.Concat11("concat", Some(ndArrayToDJLNDArray(arr)))))
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def log() : DJLNDArray[DType, Ax]= arr.log 
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def exp() : DJLNDArray[DType, Ax] = arr.exp
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def sqrt() : DJLNDArray[DType, Ax] = arr.sqrt
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def cos() : DJLNDArray[DType, Ax] = arr.cos
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def cosh() : DJLNDArray[DType, Ax] = arr.cosh
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def sin() : DJLNDArray[DType, Ax] = arr.sin
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def sinh(): DJLNDArray[DType, Ax] = arr.sinh
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def tan() : DJLNDArray[DType, Ax] = arr.tan
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def tanh() : DJLNDArray[DType, Ax] = arr.tanh
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def acos() : DJLNDArray[DType, Ax] = arr.acos
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def acosh() : DJLNDArray[DType, Ax] = arr.acosh
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def asin() : DJLNDArray[DType, Ax] = arr.asin
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def asinh() : DJLNDArray[DType, Ax] = arr.asinh
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def atan() : DJLNDArray[DType, Ax] = arr.atan
  extension[DType <: FloatSupported : ClassTag: Numeric: IsFloatSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def atanh() : DJLNDArray[DType, Ax] = arr.atanh

  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: DJLNDArray[DType, Ax]) def sigmoid(): DJLNDArray[DType, Ax] = ai.djl.nn.Activation.sigmoid(arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Ax <: Axes] (arr: DJLNDArray[DType, Ax]) def relu(): DJLNDArray[DType, Ax] = ai.djl.nn.Activation.relu(arr)

  //Binary DJLNDArray ops

  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def + (other: DJLNDArray[DType, Ax])(implicit ev: Ax =!= Axes): DJLNDArray[DType, Ax] = arr add other 
  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def - (other: DJLNDArray[DType, Ax])(implicit ev: Ax =!= Axes): DJLNDArray[DType, Ax] = arr sub other
  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def * (other: DJLNDArray[DType, Ax])(implicit ev: Ax =!= Axes): DJLNDArray[DType, Ax] = arr mul other 
  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def ** (other: DJLNDArray[DType, Ax])(implicit ev: Ax =!= Axes): DJLNDArray[DType, Ax] = arr pow other 
  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def / (other: DJLNDArray[DType, Ax])(implicit ev: Ax =!= Axes): DJLNDArray[DType, Ax] = arr div other
  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def % (other: DJLNDArray[DType, Ax])(implicit ev: Ax =!= Axes): DJLNDArray[DType, Ax] = arr mod other 

  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def > (other: DJLNDArray[DType, Ax])(implicit ev: Ax =!= Axes): DJLNDArray[Boolean, Ax] = arr gt other 
  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def >= (other: DJLNDArray[DType, Ax])(implicit ev: Ax =!= Axes): DJLNDArray[Boolean, Ax] = arr gte other 
  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def < (other: DJLNDArray[DType, Ax])(implicit ev: Ax =!= Axes): DJLNDArray[Boolean, Ax] = arr lt other 
  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def <= (other: DJLNDArray[DType, Ax])(implicit ev: Ax =!= Axes): DJLNDArray[Boolean, Ax] = arr lte other 
  
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Ax <: Axes] (arr: DJLNDArray[DType, Ax]) def ====(other: DJLNDArray[DType, Ax])(implicit ev: Ax =!= Axes): DJLNDArray[Boolean, Ax] = arr eq other
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Ax <: Axes] (arr: DJLNDArray[DType, Ax]) def !===(other: DJLNDArray[DType, Ax])(implicit ev: Ax =!= Axes): DJLNDArray[Boolean, Ax] = arr neq other

  extension[DType <: NumericSupported: ClassTag: Numeric: IsNumericSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def max (d: DJLNDArray[DType, Ax])(implicit ev: Ax =!= Axes): DJLNDArray[DType, Ax] = arr maximum d 
  extension[DType <: NumericSupported: ClassTag: Numeric: IsNumericSupported, Ax <: Axes](arr: DJLNDArray[DType, Ax]) def min (d: DJLNDArray[DType, Ax])(implicit ev: Ax =!= Axes): DJLNDArray[DType, Ax] = arr minimum d 

  //DJL-mxnet only actually supports Float here (gpu only?)
  extension[DType <: NumericSupported : ClassTag: Numeric: IsNumericSupported, Ax <: Axes, Bx <: Axes, Cx <: Axes](arr: DJLNDArray[DType, Ax]) def matmul (other: DJLNDArray[DType, Bx]): DJLNDArray[DType, Cx] = arr dot other
}
