package org.sciscala

import org.sciscala.ndscala.union._

package object ndscala
{
//TODO: Add sigmoid, relu, etc.
  type AllSupported = Int | Long | Float | Double | Boolean
  type AllUnionSupported = Union[Int]#or[Long]#or[Float]#or[Double]#or[Boolean]#create
  type AllIsSupported[T] = Contains[T, AllUnionSupported] 

}

