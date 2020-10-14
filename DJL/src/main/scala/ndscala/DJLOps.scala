package org.sciscala.ndscala 

import scala.util.Random
import scala.collection.immutable.ArraySeq
import scala.reflect.ClassTag
import spire.random.Dist
import spire.math.Numeric
import spire.implicits._
import org.sciscala.ndscala.union._

import ai.djl._
import ai.djl.ndarray._
import ai.djl.ndarray.types.Shape
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.index.dim.NDIndexSlice
//import ai.djl.nn._
//import ai.djl.nn.core._

//TODO: Typeful wrapper for DJL's DJLNDArray
object DJLOps {

  //possible leak
  val manager = NDManager.newBaseManager()

  type Supported = Union[Int]#or[Long]#or[Float]#or[Double]#create
  type FloatSupported = Union[Float]#or[Double]#create
  type IsSupported[T] = Contains[T, Supported]
  type IsFloatSupported[T] = Contains[T, FloatSupported]

  //No enforcement of DType
  type DJLNDArray[DType] = NDArray
  //  implicit def convert[DType: ClassTag](d: DType): DJLNDArray[DType] = DJLNDArray(d) 
  implicit def toDJLNDArray[DType : ClassTag](t: (ArraySeq[DType], ArraySeq[Int])): DJLNDArray[DType] = t._1 match {
    //fails on empty array
    case arr: ArraySeq[DType] => arr.head match {
      case i: Int => manager.create(arr.toArray.asInstanceOf[Array[Int]], new Shape(t._2.map(_.toLong): _*))
      case l: Long => manager.create(arr.toArray.asInstanceOf[Array[Long]], new Shape(t._2.map(_.toLong): _*))
      case f: Float => manager.create(arr.toArray.asInstanceOf[Array[Float]], new Shape(t._2.map(_.toLong): _*))
      case d: Double => manager.create(arr.toArray.asInstanceOf[Array[Double]], new Shape(t._2.map(_.toLong): _*))
    }
  }
  //returns array of numbers
  //TODO: same thing, match on first element type
  implicit def fromDJLNDArray[DType: ClassTag](t: DJLNDArray[DType]): (ArraySeq[DType], ArraySeq[Int]) = {
    val shape = ArraySeq(t.getShape.getShape.toArray: _*).map(x => x.toInt)

    t.toArray.head match{
    case i: java.lang.Integer => (ArraySeq(t.toIntArray: _*).asInstanceOf[ArraySeq[DType]] , shape) 
    case l: java.lang.Long => (ArraySeq(t.toLongArray: _*).asInstanceOf[ArraySeq[DType]] , shape) 
    case f: java.lang.Float => (ArraySeq(t.toFloatArray: _*).asInstanceOf[ArraySeq[DType]] , shape) 
    case d: java.lang.Double => (ArraySeq(t.toDoubleArray: _*).asInstanceOf[ArraySeq[DType]] , shape)
    case b: java.lang.Byte => (ArraySeq(t.toBooleanArray: _*).asInstanceOf[ArraySeq[DType]] , shape)
    }

  }
}

//class DJLNDArray[DType](ndarray: DJLNDArray) {
//  require(ndarray.getDataType() =
//}

class DJLOps extends NDArrayOps[DJLOps.DJLNDArray]{
  import DJLOps._
 
  //Nullary / factory ops
  def zeros[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int]): DJLNDArray[DType] = manager.zeros(new Shape(shape.map(_.toLong): _*))
  def ones[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int]): DJLNDArray[DType] = manager.ones(new Shape(shape.map(_.toLong): _*))
  def full[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int], value: DType): DJLNDArray[DType] = ??? // manager.full(new Shape(shape: _*), value)

  
  //TODO: fix rand
  def rand[DType : ClassTag: Numeric: IsSupported](shape: ArraySeq[Int]): DJLNDArray[DType] = ???

  //Unary ops
  def reshape[DType : ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType], newShape: ArraySeq[Int]): DJLNDArray[DType] = arr.reshape(newShape.map(_.toLong): _*)
  def transpose[DType : ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType]): DJLNDArray[DType] = arr.transpose()
  def transpose[DType : ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType], axes: ArraySeq[Int]): DJLNDArray[DType] = arr.transpose(axes: _*)
  def round[DType : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]): DJLNDArray[DType]  = arr.round
  //Top-level slice only
  def slice[DType : ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType], start: Int, end: Int): DJLNDArray[DType] = arr.get((new NDIndex()).addSliceDim(start, end))

  def squeeze[DType : ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType], index: ArraySeq[Int]): DJLNDArray[DType] = arr.squeeze(index.toArray)

  def rank[DType : ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType]): Int = arr.getShape.getShape.size
//  def clip[DType : ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType], min: DType, max: DType): DJLNDArray[DType] = arr.clip(min, max) 

  def unary_-[DType : ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType]) : DJLNDArray[DType] = arr.neg 

  def abs[DType: ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType]): DJLNDArray[DType] = arr.abs 
  def ceil[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]): DJLNDArray[DType] = arr.ceil
  def floor[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]): DJLNDArray[DType] = arr.floor 
//  def concat[DType: ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType]*): DJLNDArray[DType] = onnx.Concat11("concat", Some(ndArrayToDJLNDArray(arr)))))
  def log[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]): DJLNDArray[DType]= arr.log 
  def exp[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]): DJLNDArray[DType] = arr.exp
  def sqrt[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]): DJLNDArray[DType] = arr.sqrt
  def cos[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]): DJLNDArray[DType] = arr.cos
  def cosh[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]): DJLNDArray[DType] = arr.cosh
  def sin[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]): DJLNDArray[DType] = arr.sin
  def sinh[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]): DJLNDArray[DType] = arr.sinh
  def tan[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]): DJLNDArray[DType] = arr.tan
  def tanh[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]): DJLNDArray[DType] = arr.tanh
  def acos[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]): DJLNDArray[DType] = arr.acos
  def acosh[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]): DJLNDArray[DType] = arr.acosh
  def asin[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]): DJLNDArray[DType] = arr.asin
  def asinh[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]): DJLNDArray[DType] = arr.asinh
  def atan[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]): DJLNDArray[DType] = arr.atan
//  def atanh[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType]): DJLNDArray[DType] = onnx.Atanh9("atanh", Some(arr))

  //Binary DJLNDArray ops

  def +[DType : ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType], other: DJLNDArray[DType]): DJLNDArray[DType] = arr add other 
  def -[DType : ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType], other: DJLNDArray[DType]): DJLNDArray[DType] = arr sub other
  def *[DType : ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType], other: DJLNDArray[DType]): DJLNDArray[DType] = arr mul other 
  def **[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType], other: DJLNDArray[DType]): DJLNDArray[DType] = arr pow other 
  def /[DType : ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType], other: DJLNDArray[DType]): DJLNDArray[DType] = arr div other
  def %[DType : ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType], other: DJLNDArray[DType]): DJLNDArray[DType] = arr mod other 

  def >[DType : ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType], other: DJLNDArray[DType]): DJLNDArray[Boolean] = arr gt other 
  def >=[DType : ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType], other: DJLNDArray[DType]): DJLNDArray[Boolean] = arr gte other 
  def <[DType : ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType], other: DJLNDArray[DType]): DJLNDArray[Boolean] = arr lt other 
  def <=[DType : ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType], other: DJLNDArray[DType]): DJLNDArray[Boolean] = arr lte other 
  def ===[DType : ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType], other: DJLNDArray[DType]): DJLNDArray[Boolean] = arr eq other 
  def !==[DType : ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType], other: DJLNDArray[DType]): DJLNDArray[Boolean] = arr neq other 

  def max[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType], d: DJLNDArray[DType]): DJLNDArray[DType] = arr maximum d 
  def min[DType: ClassTag: Numeric: IsFloatSupported](arr: DJLNDArray[DType], d: DJLNDArray[DType]): DJLNDArray[DType] = arr minimum d 

  def dot[DType : ClassTag: Numeric: IsSupported](arr: DJLNDArray[DType], other: DJLNDArray[DType]): DJLNDArray[DType] = arr dot other
}
