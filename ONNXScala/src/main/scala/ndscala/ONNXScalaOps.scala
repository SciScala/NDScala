package org.sciscala.ndscala 

import scala.util.Random
import scala.collection.immutable.ArraySeq
import scala.reflect.ClassTag
import spire.random.Dist
import spire.math._
//import spire.implicits.Numeric
import scala.language.implicitConversions

import io.kjaer.compiletime._
import io.kjaer.compiletime.Shape.NumElements
import org.emergentorder.compiletime._
import org.emergentorder.onnx.Tensors._
import org.emergentorder.onnx.Tensors.Tensor._
import org.emergentorder.onnx.backends.ORTOperatorBackendAll
import org.emergentorder.compiletime.TensorShapeDenotation.Reverse
import org.emergentorder.compiletime.TensorShapeDenotation.Concat
//import org.emergentorder.=!=

object ONNXScalaOps {
  //implicit def convert[DType <: Supported : ClassTag](d: DType)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (?, ?, 1 #: SNil)] = Tensor(Array(d), (?,?,1 #: SNil))
//  implicit def toTensor[DType <: Supported : ClassTag, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](t: (Array[DType], (Tt,Td,S)))(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = Tensor(t.data.toIAy,)
//  implicit def fromTensor[DType <: Supported : ClassTag, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](t: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): (Array[DType], (Tt,Td,S)) = {
//    (t.data, t._2)
//   
//  }
  val onnx = new ORTOperatorBackendAll()
}

//TODO: Stricter type bounds because ORT doesn't implement them all
//TODO: dotty-style typeclass
//Trying to load ORT shared lib from elsewhere
given NDArrayOps[Tensor] with {
//  type Supported = Int | Long | Float | Double //Union[Int]#or[Long]#or[Float]#or[Double]#create
//  type FloatSupported = Float | Double //Union[Float]#or[Double]#create

//  implicit def convert[DType <: Supported : ClassTag](d: DType)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType] = TensorFactory.getTensor(Array(d).toArray, Array(1).toArray)
//  implicit def toTensor[DType <: Supported : ClassTag](t: (Array[DType], Array[Int]))(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType] = TensorFactory.getTensor(t._1.toArray, t._2.toArray)
//  implicit def fromTensor[DType <: Supported : ClassTag](t: Tensor[DType])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): (Array[DType], Array[Int]) = (t._1, t._2)
/*
  given convert as Conversion[Int, Tensor[Int]] {
    def apply(d:Int)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Int] = TensorFactory.getTensor(Array(d).toArray, Array(1).toArray)
  }

  given convertLong as Conversion[Long, Tensor[Long]] {
    def apply(d:Long)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Long] = TensorFactory.getTensor(Array(d).toArray, Array(1).toArray)
  }

  given convertFloat as Conversion[Float, Tensor[Float]] {
    def apply(d:Float)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Float] = TensorFactory.getTensor(Array(d).toArray, Array(1).toArray)
  }

  given convertDouble as Conversion[Double, Tensor[Double]] {
    def apply(d:Double)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Double] = TensorFactory.getTensor(Array(d).toArray, Array(1).toArray)
  }

  given toTensor as Conversion[(Array[Int], Array[Int]), Tensor[Int]]{
    def apply(x: (Array[Int], Array[Int]))(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Int] = TensorFactory.getTensor(x._1.toArray, x._2.toArray)
  }

  given toTensorLong as Conversion[(Array[Long], Array[Int]), Tensor[Long]]{
    def apply(x: (Array[Long], Array[Int]))(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Long] = TensorFactory.getTensor(x._1.toArray, x._2.toArray)
  }
  given toTensorFloat as Conversion[(Array[Float], Array[Int]), Tensor[Float]]{
    def apply(x: (Array[Float], Array[Int]))(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Float] = TensorFactory.getTensor(x._1.toArray, x._2.toArray)
  }

  given toTensorDouble as Conversion[(Array[Double], Array[Int]), Tensor[Double]]{
    def apply(x: (Array[Double], Array[Int]))(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Double] = TensorFactory.getTensor(x._1.toArray, x._2.toArray)
  }

 
  given fromTensor as Conversion[Tensor[Int], (Array[Int], Array[Int])]{
    def apply(x: Tensor[Int])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]):(Array[Int], Array[Int]) = (Array.from(x._1), Array.from(x._2))
  }

  given fromTensorLong as Conversion[Tensor[Long], (Array[Long], Array[Int])]{
    def apply(x: Tensor[Long])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]):(Array[Long], Array[Int]) = (Array.from(x._1), Array.from(x._2))
  }

  given fromTensorFloat as Conversion[Tensor[Float], (Array[Float], Array[Int])]{
    def apply(x: Tensor[Float])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]):(Array[Float], Array[Int]) = (Array.from(x._1), Array.from(x._2))
  }

  given fromTensorDouble as Conversion[Tensor[Double], (Array[Double], Array[Int])]{
    def apply(x: Tensor[Double])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]):(Array[Double], Array[Int]) = (Array.from(x._1), Array.from(x._2))
  }
*/


//class ONNXScalaOps extends NDArrayOps[Tensor]{
//  import ONNXScalaOps._
 /*
  import org.sciscala.ndscala.ONNXScalaOps.toTensor
  import org.sciscala.ndscala.ONNXScalaOps.toTensorLong 
  import org.sciscala.ndscala.ONNXScalaOps.toTensorFloat
  import org.sciscala.ndscala.ONNXScalaOps.toTensorDouble

  import org.sciscala.ndscala.ONNXScalaOps.fromTensor
  import org.sciscala.ndscala.ONNXScalaOps.fromTensorLong 
  import org.sciscala.ndscala.ONNXScalaOps.fromTensorFloat
  import org.sciscala.ndscala.ONNXScalaOps.fromTensorDouble

  import org.sciscala.ndscala.ONNXScalaOps.convert
  import org.sciscala.ndscala.ONNXScalaOps.convertLong 
  import org.sciscala.ndscala.ONNXScalaOps.convertFloat
  import org.sciscala.ndscala.ONNXScalaOps.convertDouble
*/
  val onnx = ONNXScalaOps.onnx
//  val seed = Array(1l,2l,3l,4l,5l)
  val rng = spire.random.rng.Cmwc5()

  //Nullary / factory ops

  def zeros[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = Tensor(Array.fill(s.value.toSeq.foldLeft(1)(_*_))(implicitly[Numeric[DType]].zero),tt.value,td.value, s.value)
  def ones[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](shape: Array[Int])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = Tensor(Array.fill(shape.foldLeft(1)(_*_))(implicitly[Numeric[DType]].one), tt.value, td.value,s.value)
  def full[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](shape: Array[Int], value: DType)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = Tensor(Array.fill(shape.foldLeft(1)(_*_))(value), tt.value, td.value, s.value)

  //TODO: fix rand
//  def rand[DType <: Supported : ClassTag: Numeric](shape: Array[Int])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType] = ???


  //Unary ops
  extension [DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def reshape[Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Shape](using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S],tt1: ValueOf[Tt1], td1: TensorShapeDenotationOf[Td1], s1: ShapeOf[S1], sizeSeq: NumElements[S] =:= NumElements[S1]): Tensor[DType, (Tt1,Td1,S1)] = onnx.ReshapeV13("reshape", arr,
    Tensor(shapeOf[S1].toSeq.toArray.map(x => x.toLong), tt1.value, td1.value, Shape.fromSeq(ArraySeq.unsafeWrapArray(Array(shapeOf[S1].toSeq.size))))) //wrong denotations
    extension [DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](arr: Tensor[DType, (Tt,Td,S)]) def expand[Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Shape](using tt1: ValueOf[Tt1], td1: TensorShapeDenotationOf[Td1], s1: ShapeOf[S1]): Tensor[DType, (Tt1,Td1,S1)] = onnx.ExpandV13("expand", arr, Tensor(shapeOf[S1].toSeq.toArray.map(x => x.toLong), tt1.value, td1.value, Shape.fromSeq(ArraySeq.unsafeWrapArray(Array(shapeOf[S1].toSeq.size)))))
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def transpose(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Reverse[Td]], s: ShapeOf[io.kjaer.compiletime.Shape.Reverse[S]]): Tensor[DType, (Tt,Reverse[Td],io.kjaer.compiletime.Shape.Reverse[S])] = onnx.TransposeV13("transpose", None, arr)
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def transpose(axes: Array[Int])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Reverse[Td]], s: ShapeOf[io.kjaer.compiletime.Shape.Reverse[S]]): Tensor[DType, (Tt,Reverse[Td],io.kjaer.compiletime.Shape.Reverse[S])] = onnx.TransposeV13("transpose", Some(axes.toArray), arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def round()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)]  = onnx.RoundV11("round", arr)


  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape]  (arr: Tensor[DType, (Tt,Td,S)]) def slice[Tt2 <: TensorTypeDenotation, AxesStart <: Indices, AxesEnd <: Indices](using tt: ValueOf[Tt2], td: TensorShapeDenotationOf[Td], s2: ShapeOf[SlicedShape[AxesStart,AxesEnd]], i: IndicesOf[AxesStart], i2: IndicesOf[AxesEnd]): Tensor[DType, (Tt2,Td,SlicedShape[AxesStart,AxesEnd])] = onnx.SliceV13("slice", arr, indicesOf[AxesStart], indicesOf[AxesEnd]) //wrong denotations


//  extension[DType <: NumericSupported : ClassTag : Numeric: IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def gather[Tt2 <: TensorTypeDenotation, Td2 <: TensorShapeDenotation, AxisIndex <: Index ::: INil, AxisIndices <: Indices](using tt: ValueOf[Tt2], td: TensorShapeDenotationOf[Td2], s2: ShapeOf[GatheredShape[S, AxisIndex, AxisIndices]], i: IndicesOf[AxisIndex], i2: IndicesOf[AxisIndices]): Tensor[DType, (Tt2,Td2,GatheredShape[S, AxisIndex, AxisIndices])] = onnx.GatherV13("gather", indicesOf[AxisIndex], arr, indicesOf[AxisIndices])
  extension[DType <: NumericSupported : ClassTag : Numeric: IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape]  (arr: Tensor[DType, (Tt,Td,S)]) def flatten[Tt2 <: TensorTypeDenotation, AxisIndex <: Index ::: INil](using tt: ValueOf[Tt2], td: TensorShapeDenotationOf[Td], s2: ShapeOf[FlattenedShape[S, AxisIndex]], i: IndicesOf[AxisIndex]): Tensor[DType, (Tt2,Td,FlattenedShape[S, AxisIndex])] = onnx.FlattenV13("flatten", indicesOf[AxisIndex], arr)

  //Note: currently fixed mode, constant value

  extension[DType <: NumericSupported : ClassTag : Numeric:  IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape]  (arr: Tensor[DType, (Tt,Td,S)]) def pad[Tt2 <: TensorTypeDenotation, AxesBefore <: Shape, AxesAfter <: Shape](constantValue: DType)(using tt: ValueOf[Tt2], td: TensorShapeDenotationOf[Td], s2: ShapeOf[PaddedShape[S,AxesBefore,AxesAfter]], i: ShapeOf[AxesBefore], i2: ShapeOf[AxesAfter]): Tensor[DType, (Tt2,Td,PaddedShape[S,AxesBefore,AxesAfter])] = onnx.PadV13("pad", mode = "constant", arr, shapeOf[AxesBefore], shapeOf[AxesAfter], Some(Tensor(Array(constantValue), tt.value, td.value, SNil)))
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape]  (arr: Tensor[DType, (Tt,Td,S)]) def tile[Tt2 <: TensorTypeDenotation, AxisRepeats <: Indices](using tt: ValueOf[Tt2], td: TensorShapeDenotationOf[Td], s2: ShapeOf[TiledShape[S, AxisRepeats]], i: IndicesOf[AxisRepeats]): Tensor[DType, (Tt2,Td,TiledShape[S, AxisRepeats])] = onnx.TileV13("tile", arr, indicesOf[AxisRepeats])

  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape]  (arr: Tensor[DType, (Tt,Td,S)]) def shape[Tt2 <: TensorTypeDenotation, Td2 <: TensorShapeDenotation](using tt: ValueOf[Tt2], td: TensorShapeDenotationOf[Td2], s2: ShapeOf[io.kjaer.compiletime.Shape.Rank[S] & Dimension #: SNil]): Tensor[Long, (Tt2,Td2,io.kjaer.compiletime.Shape.Rank[S] & Dimension #: SNil)] = onnx.ShapeV13[DType,Long, Tt,Td, S, Tt2, Td2]("shape", arr)
 
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def squeeze[Tt1 <: TensorTypeDenotation, Axes <: Indices](using tt: ValueOf[Tt1], td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td,Axes,false]], s: ShapeOf[KeepOrReduceDims[S,Axes,false]], i: IndicesOf[Axes]): Tensor[DType, Tuple3[Tt1,KeepOrReduceDimDenotations[Td,Axes,false],KeepOrReduceDims[S,Axes,false]]] = onnx.SqueezeV13("squeeze",Some(indicesOf[Axes]),arr)
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def unsqueeze[Tt1 <: TensorTypeDenotation, Axes <: Indices](using tt: ValueOf[Tt1], td: TensorShapeDenotationOf[Td], s: ShapeOf[UnsqueezeShape[S,Axes]], i: IndicesOf[Axes]): Tensor[DType, Tuple3[Tt1,Td,UnsqueezeShape[S,Axes]]] = onnx.UnsqueezeV13("unsqueeze",Some(indicesOf[Axes]),arr)

  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def rank: Int = arr.shape.size

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def unary_- (using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.NegV13("neg", arr)


  //reduction ops:
  //TODO: tests for all but reduceSum
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def reduceSum[Tt1 <: TensorTypeDenotation, Axes <: Indices, KeepDims <: (Boolean&Singleton)] (using tt: ValueOf[Tt1], td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td,Axes,KeepDims]], s: ShapeOf[KeepOrReduceDims[S,Axes,KeepDims]], i: IndicesOf[Axes], k: ValueOf[KeepDims]): Tensor[DType, Tuple3[Tt1,KeepOrReduceDimDenotations[Td,Axes,KeepDims],KeepOrReduceDims[S,Axes,KeepDims]]] = onnx.ReduceSumV13[DType, Tt, Td, S, Tt1, Axes, KeepDims ]("reduceSum", Some(indicesOf[Axes]), valueOf[KeepDims], arr )
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def reduceLogSum[Tt1 <: TensorTypeDenotation, Axes <: Indices, KeepDims <: (Boolean&Singleton)] (using tt: ValueOf[Tt1], td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td,Axes,KeepDims]], s: ShapeOf[KeepOrReduceDims[S,Axes,KeepDims]], i: IndicesOf[Axes], k: ValueOf[KeepDims]): Tensor[DType, Tuple3[Tt1,KeepOrReduceDimDenotations[Td,Axes,KeepDims],KeepOrReduceDims[S,Axes,KeepDims]]] = onnx.ReduceLogSumV13[DType, Tt, Td, S, Tt1, Axes, KeepDims ]("reduceLogSum", Some(indicesOf[Axes]), valueOf[KeepDims], arr )
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def reduceMax[Tt1 <: TensorTypeDenotation, Axes <: Indices, KeepDims <: (Boolean&Singleton)] (using tt: ValueOf[Tt1], td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td,Axes,KeepDims]], s: ShapeOf[KeepOrReduceDims[S,Axes,KeepDims]], i: IndicesOf[Axes], k: ValueOf[KeepDims]): Tensor[DType, Tuple3[Tt1,KeepOrReduceDimDenotations[Td,Axes,KeepDims],KeepOrReduceDims[S,Axes,KeepDims]]] = onnx.ReduceMaxV13[DType, Tt, Td, S, Tt1, Axes, KeepDims ]("reduceMax", Some(indicesOf[Axes]), valueOf[KeepDims], arr )
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def reduceMean[Tt1 <: TensorTypeDenotation, Axes <: Indices, KeepDims <: (Boolean&Singleton)] (using tt: ValueOf[Tt1], td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td,Axes,KeepDims]], s: ShapeOf[KeepOrReduceDims[S,Axes,KeepDims]], i: IndicesOf[Axes], k: ValueOf[KeepDims]): Tensor[DType, Tuple3[Tt1,KeepOrReduceDimDenotations[Td,Axes,KeepDims],KeepOrReduceDims[S,Axes,KeepDims]]] = onnx.ReduceMeanV13[DType, Tt, Td, S, Tt1, Axes, KeepDims ]("reduceMean", Some(indicesOf[Axes]), valueOf[KeepDims], arr )
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def reduceMin[Tt1 <: TensorTypeDenotation, Axes <: Indices, KeepDims <: (Boolean&Singleton)] (using tt: ValueOf[Tt1], td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td,Axes,KeepDims]], s: ShapeOf[KeepOrReduceDims[S,Axes,KeepDims]], i: IndicesOf[Axes], k: ValueOf[KeepDims]): Tensor[DType, Tuple3[Tt1,KeepOrReduceDimDenotations[Td,Axes,KeepDims],KeepOrReduceDims[S,Axes,KeepDims]]] = onnx.ReduceMinV13[DType, Tt, Td, S, Tt1, Axes, KeepDims ]("reduceMin", Some(indicesOf[Axes]), valueOf[KeepDims], arr )
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def reduceProd[Tt1 <: TensorTypeDenotation, Axes <: Indices, KeepDims <: (Boolean&Singleton)] (using tt: ValueOf[Tt1], td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td,Axes,KeepDims]], s: ShapeOf[KeepOrReduceDims[S,Axes,KeepDims]], i: IndicesOf[Axes], k: ValueOf[KeepDims]): Tensor[DType, Tuple3[Tt1,KeepOrReduceDimDenotations[Td,Axes,KeepDims],KeepOrReduceDims[S,Axes,KeepDims]]] = onnx.ReduceProdV13[DType, Tt, Td, S, Tt1, Axes, KeepDims ]("reduceProd", Some(indicesOf[Axes]), valueOf[KeepDims], arr )
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def reduceSumSquare[Tt1 <: TensorTypeDenotation, Axes <: Indices, KeepDims <: (Boolean&Singleton)] (using tt: ValueOf[Tt1], td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td,Axes,KeepDims]], s: ShapeOf[KeepOrReduceDims[S,Axes,KeepDims]], i: IndicesOf[Axes], k: ValueOf[KeepDims]): Tensor[DType, Tuple3[Tt1,KeepOrReduceDimDenotations[Td,Axes,KeepDims],KeepOrReduceDims[S,Axes,KeepDims]]] = onnx.ReduceSumSquareV13[DType, Tt, Td, S, Tt1, Axes, KeepDims ]("reduceSumSquared", Some(indicesOf[Axes]), valueOf[KeepDims], arr )
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def argMax[Tt1 <: TensorTypeDenotation, Axes <: Index ::: INil, KeepDims <: (Boolean&Singleton)] (using tt: ValueOf[Tt1], td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td,Axes,KeepDims]], s: ShapeOf[KeepOrReduceDims[S,Axes,KeepDims]], i: IndicesOf[Axes], k: ValueOf[KeepDims]): Tensor[Long, Tuple3[Tt1,KeepOrReduceDimDenotations[Td,Axes,KeepDims],KeepOrReduceDims[S,Axes,KeepDims]]] = onnx.ArgMaxV13[DType, Tt, Td, S, Tt1, Axes, KeepDims ]("argMax", indicesOf[Axes], valueOf[KeepDims], data = arr )
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def argMin[Tt1 <: TensorTypeDenotation, Axes <: Index ::: INil, KeepDims <: (Boolean&Singleton)] (using tt: ValueOf[Tt1], td: TensorShapeDenotationOf[KeepOrReduceDimDenotations[Td,Axes,KeepDims]], s: ShapeOf[KeepOrReduceDims[S,Axes,KeepDims]], i: IndicesOf[Axes], k: ValueOf[KeepDims]): Tensor[Long, Tuple3[Tt1,KeepOrReduceDimDenotations[Td,Axes,KeepDims],KeepOrReduceDims[S,Axes,KeepDims]]] = onnx.ArgMinV13[DType, Tt, Td, S, Tt1, Axes, KeepDims ]("argMin", indicesOf[Axes], valueOf[KeepDims], data = arr )


  
  extension[DType <: NumericSupported : ClassTag : Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def abs()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.AbsV13("abs", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def ceil()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.CeilV13("ceil", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def floor()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.FloorV13("floor", arr)
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, SSuffix <: Shape, S <: Dimension #: SSuffix, S1 <: Dimension #: SSuffix](arr: Tuple2[Tensor[DType, (Tt, Td, S)],Tensor[DType, (Tt, Td, S1)]]) def concat[Axis <: Index ::: INil](using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[AddGivenAxisSize[S,S1,Axis]], i: IndicesOf[Axis] ): Tensor[DType, (Tt,Td,AddGivenAxisSize[S,S1,Axis])] = onnx.ConcatV13("concat", indicesOf[Axis], arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](arr: Seq[Tensor[DType, (Tt,Td,S)]]) def mean()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.MeanV13("mean", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape](arr: Seq[Tensor[DType, (Tt,Td,S)]]) def ndsum()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.SumV13("sum", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def log()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)]= onnx.LogV13("log", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def exp()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.ExpV13("exp", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def sqrt()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.SqrtV13("sqrt", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def cos()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.CosV7("cos", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def cosh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.CoshV9("cosh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def sin()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.SinV7("sin", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def sinh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.SinhV9("sinh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def tan()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.TanV7("tan", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def tanh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.TanhV13("tanh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def acos()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.AcosV7("acos", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def acosh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.AcoshV9("acosh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def asin()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.AsinV7("asin", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def asinh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.AsinhV9("asinh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def atan()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.AtanV7("atan", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def atanh()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.AtanhV9("atanh", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def sigmoid()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.SigmoidV13("sig", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def relu()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.ReluV13("relu", arr)
//TODO: test
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def leakyRelu(alpha: Float = 0.01)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.LeakyReluV6("leakyRelu", alpha, arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def celu(alpha: Float = 1.0)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.CeluV12("celu", alpha, arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def elu(alpha: Float = 1.0)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.EluV6("elu", alpha, arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def selu(alpha: Float = 1.67326, gamma: Float = 1.0507)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.SeluV6("selu", alpha, gamma, arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def prelu(slope: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.PReluV9("prelu", arr, slope)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def isNaN()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Boolean, (Tt,Td,S)] = onnx.IsNaNV13[DType, Boolean, Tt, Td, S]("isNaN", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def sign()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.SignV13("sign", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def clip(min: DType, max: DType)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.ClipV13("clip", arr, Tensor(Array(min), tt.value, td.value, SNil), Tensor(Array(max), tt.value, td.value, SNil))
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def dropout(seed: Int = 42, ratio: Float, trainingMode: Boolean = false)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.DropoutV13("dropout", seed, arr, Tensor(Array(ratio), tt.value, td.value, SNil), Tensor(Array(trainingMode), tt.value, td.value, SNil))
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Dimension #: Dimension #: SNil] (arr: Tensor[DType, (Tt,Td,S)]) def softmax(axis: Int = 1)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.SoftmaxV13("softmax", input = arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Dimension #: Dimension #: Dimension #: Dimension #: SNil] (arr: Tensor[DType, (Tt,Td,S)]) def instanceNormalization[Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Dimension #: SNil](epsilon: Float = 1e-5, scale: Tensor[DType, (Tt1,Td1,S1)], b: Tensor[DType, (Tt1,Td1,S1)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.InstanceNormalizationV6("instanceNormalization", epsilon, arr, scale, b)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Dimension #: Dimension #: Dimension #: Dimension #: SNil] (arr: Tensor[DType, (Tt,Td,S)]) def lrn(alpha: Float = 0.0001, beta: Float = 0.75, bias: Float = 1.0, size: Int)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.LRNV13("lrn", alpha, beta, bias, size, arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, N <: Dimension, C <: Dimension, H <: Dimension, W <: Dimension, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: N #: C #: H #: W #: SNil, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: C #: SNil] (arr: Tensor[DType, (Tt,Td,S)]) def batchnorm(epsilon: Float = 1e-05, momentum: Float = 0.9, scale: Tensor[DType, Tuple3[Tt1, Td1, S1]], b: Tensor[DType, Tuple3[Tt1, Td1, S1]], mean: Tensor[DType, Tuple3[Tt1, Td1, S1]], someVar: Tensor[DType, Tuple3[Tt1, Td1, S1]])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.BatchNormalizationV9("batchnorm", epsilon, momentum, arr, scale, b, mean, someVar)

  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, N <: Dimension, C <: Dimension, H <: Dimension, W <: Dimension, KH <: Dimension, KW <: Dimension, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: N #: C #: H #: W #: SNil, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: 1 #: C #: KH #: KW #: SNil, Tt2 <: TensorTypeDenotation, Td2 <: TensorShapeDenotation, S2 <: 1 #: SNil, S3 <: KH #: KW #: SNil, PadsBefore <: None.type | Dimension #: Dimension #: SNil, PadsAfter <: None.type | Dimension #: Dimension #: SNil] (arr: Tensor[DType, (Tt,Td,S)]) def conv(kernelShape: S3, w: Tensor[DType, Tuple3[Tt1,Td1,S1]], padsBefore: PadsBefore = None, padsAfter: PadsAfter = None)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[PaddedShape[PoolShape[S,S3], PadsBefore, PadsAfter]], s3: ShapeOf[S3]): Tensor[DType, (Tt,Td,PaddedShape[PoolShape[S,S3], PadsBefore, PadsAfter])] = onnx.ConvV11[DType, N, C, H, W, KH, KW, Tt, Td, S, Tt1, Td1, S1, Tt2, Td2, S2, Tt, Td, S3, PadsBefore, PadsAfter]("conv", X = arr, W = w, padsBefore = padsBefore, padsAfter = padsAfter, kernel_shape = kernelShape)

  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Dimension #: Dimension #: Dimension #: Dimension #: SNil, S1 <: Dimension #: Dimension #: SNil, PadsBefore <: None.type | Dimension #: Dimension #: SNil, PadsAfter <: None.type | Dimension #: Dimension #: SNil] (arr: Tensor[DType, (Tt,Td,S)]) def averagePool(kernelShape: S1, padsBefore: PadsBefore = None, padsAfter: PadsAfter = None)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[PaddedShape[PoolShape[S,S1], PadsBefore, PadsAfter]], s1: ShapeOf[S1]): Tensor[DType, (Tt,Td,PaddedShape[PoolShape[S,S1], PadsBefore, PadsAfter])] = onnx.AveragePoolV11("avgpool", X = arr, padsBefore = padsBefore, padsAfter = padsAfter, kernel_shape = kernelShape)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Dimension #: Dimension #: Dimension #: Dimension #: SNil, S1 <: Dimension #: Dimension #: SNil, PadsBefore <: None.type | Dimension #: Dimension #: SNil, PadsAfter <: None.type | Dimension #: Dimension #: SNil] (arr: Tensor[DType, (Tt,Td,S)]) def maxPool(kernelShape: S1, padsBefore: PadsBefore = None, padsAfter: PadsAfter = None)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[PaddedShape[PoolShape[S,S1], PadsBefore, PadsAfter]], s1: ShapeOf[S1]): Tensor[DType, (Tt,Td,PaddedShape[PoolShape[S,S1], PadsBefore, PadsAfter])] = onnx.MaxPoolV12("maxpool", X = arr, padsBefore = padsBefore, padsAfter = padsAfter, kernel_shape = kernelShape)

  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, N <: Dimension, C <: Dimension, H <: Dimension, W <: Dimension, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: N #: C #: H #: W #: SNil] (arr: Tensor[DType, (Tt,Td,S)]) def globalAveragePool()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[N #: C #: 1 #: 1 #: SNil]): Tensor[DType, (Tt,Td,N #: C #: 1 #: 1 #: SNil)] = onnx.GlobalAveragePoolV1("globalavgpool", X = arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, N <: Dimension, C <: Dimension, H <: Dimension, W <: Dimension, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: N #: C #: H #: W #: SNil] (arr: Tensor[DType, (Tt,Td,S)]) def globalMaxPool()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[N #: C #: 1 #: 1 #: SNil]): Tensor[DType, (Tt,Td,N #: C #: 1 #: 1 #: SNil)] = onnx.GlobalMaxPoolV1("globalmaxpool", X = arr)

  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def reciprocal()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.ReciprocalV13("reciprocal", arr)


  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def inverse()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.InverseV1("inverse", arr)
  extension[DType <: FloatSupported : ClassTag: Numeric : IsFloatSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def constant()(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.ConstantV13("constant", value = Some(arr))

  // logical ops - TODO: tests
  extension[DType <: Boolean : ClassTag, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def unary_!(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.NotV1("not", arr)
  extension[DType <: Boolean : ClassTag, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def &&(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.AndV7("and", arr, other)
  extension[DType <: Boolean : ClassTag, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def ||(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.OrV7("or", arr, other)
  extension[DType <: Boolean : ClassTag, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def ^(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.XorV7("xor", arr, other)


  //Binary Tensor ops

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def +(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.AddV13("add", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def -(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.SubV13("sub", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def *(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.MulV13("mul", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, DType1 <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def **(d: Tensor[DType1, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.PowV13[DType,DType1,Tt,Td,S]("pow", arr, d)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def /(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.DivV13("div", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def %(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.ModV13("mod", A=arr, B=other)

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def >(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Boolean, (Tt,Td,S)] = onnx.GreaterV13[DType, Boolean, Tt,Td,S, Tt, Td]("gt", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def >=(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Boolean, (Tt,Td,S)] = onnx.GreaterOrEqualV12[DType, Boolean, Tt,Td,S, Tt, Td]("gte", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def <(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Boolean, (Tt,Td,S)] = onnx.LessV13[DType, Boolean, Tt,Td,S, Tt, Td]("lt", arr, other)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def <=(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Boolean, (Tt,Td,S)] = onnx.LessOrEqualV12[DType, Boolean, Tt,Td,S, Tt, Td]("lte", arr, other)
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def ====(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Boolean, (Tt,Td,S)] = onnx.EqualV13[DType, Boolean, Tt,Td,S, Tt, Td]("eq", arr, other)
  extension[DType <: Supported : ClassTag : IsSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def !===(other: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[Boolean, (Tt,Td,S)] = onnx.NotV1("not", onnx.EqualV13[DType, Boolean, Tt,Td,S, Tt, Td]("eq", arr, other))
 
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def max(d: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.MaxV13("max", Seq(arr, d))
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Shape] (arr: Tensor[DType, (Tt,Td,S)]) def min(d: Tensor[DType, (Tt,Td,S)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td], s: ShapeOf[S]): Tensor[DType, (Tt,Td,S)] = onnx.MinV13("min", Seq(arr, d))

  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Dim0 <: Dimension, Dim1 <: Dimension, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Dim0 #: Dim1 #:SNil] (arr: Tensor[DType, (Tt,Td,S)]) def gemm[Dim2 <: Dimension, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Dim1 #: Dim2 #: SNil](other: Tensor[DType, (Tt1,Td1,S1)], C: Option[Tensor[DType, Tuple3[Tt,Td,Dim0 #: Dim2 #: SNil]]] = None, alpha: Float = 1.0, beta: Float = 1.0, transA: Int = 0, transB: Int = 0)(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td],vd:ValueOf[scala.compiletime.ops.int.S[Dim0]], vd1:ValueOf[scala.compiletime.ops.int.S[Dim1]], vd2: ValueOf[scala.compiletime.ops.int.S[Dim2]], s2: ShapeOf[Dim0 #: Dim2 #: SNil]): Tensor[DType, (Tt,Td,Dim0 #: Dim2 #: SNil)] = onnx.GemmV13("gemm",alpha,beta,transA,transB,arr,other,C)
  extension[DType <: NumericSupported : ClassTag: Numeric : IsNumericSupported, Dim0 <: Dimension, Dim1 <: Dimension, Tt <: TensorTypeDenotation, Td <: TensorShapeDenotation, S <: Dim0 #: Dim1 #:SNil] (arr: Tensor[DType, (Tt,Td,S)]) def matmul[Dim2 <: Dimension, Tt1 <: TensorTypeDenotation, Td1 <: TensorShapeDenotation, S1 <: Dim1 #: Dim2 #: SNil](other: Tensor[DType, (Tt1,Td1,S1)])(using tt: ValueOf[Tt], td: TensorShapeDenotationOf[Td],vd:ValueOf[scala.compiletime.ops.int.S[Dim0]], vd1:ValueOf[scala.compiletime.ops.int.S[Dim1]], vd2: ValueOf[scala.compiletime.ops.int.S[Dim2]], s2: ShapeOf[Dim0 #: Dim2 #: SNil]): Tensor[DType, (Tt,Td,Dim0 #: Dim2 #: SNil)] = onnx.MatMulV13("matmul", arr, other)
}
