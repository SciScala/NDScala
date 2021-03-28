import scala.xml.transform.{RewriteRule, RuleTransformer}

//val dottyVersion = dottyLatestNightlyBuild.get
val dottyVersion = "3.0.0-RC1"
val scala213Version = "2.13.4"

resolvers in Global += "scala-integration" at "https://scala-ci.typesafe.com/artifactory/scala-integration/"

ThisBuild / scalaVersion     := dottyVersion
ThisBuild / version          := "0.1.0-SNAPSHOT"
ThisBuild / organization     := "org.sciscala"
ThisBuild / organizationName := "sciscala"

Global / concurrentRestrictions := Seq(
  Tags.limit(Tags.Test, 1)
)

crossScalaVersions := Seq(dottyVersion, scala213Version)
lazy val scalaTest = ("org.scalatest" %% "scalatest" % "3.2.5")
lazy val core = (project in file("core"))
  .settings(
    name := "ndscala-core",
    scalacOptions += "-Ymacro-annotations",
    resolvers += Resolver.mavenLocal,
    resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots",
    libraryDependencies += ("org.typelevel" %% "simulacrum" % "1.0.1").withDottyCompat(dottyVersion), //TODO: Move to simulacrum-scalafix
    libraryDependencies += ("org.typelevel" %% "spire" % "0.17.0").withDottyCompat(dottyVersion),
    libraryDependencies += "com.github.EmergentOrder" %% "onnx-scala" % "0.12.0",
  //Local only  
//  libraryDependencies += "io.kjaer" % "tf-dotty-compiletime" % "0.0.0+134-f1f8d0ba+20201020-1123-SNAPSHOT",
//    libraryDependencies += "io.kjaer" % "tf-dotty-compiletime_0.27" % "0.0.0+134-f1f8d0ba+20201020-1123-SNAPSHOT",
    libraryDependencies += scalaTest % Test,
    crossScalaVersions := Seq(dottyVersion, scala213Version)
   )
lazy val onnxscala = (project in file("ONNXScala"))
  .dependsOn(core)
  .settings(
    name := "ndscala-onnx-scala",
    libraryDependencies += ("org.typelevel" %% "spire" % "0.17.0").withDottyCompat(dottyVersion),
    libraryDependencies += "com.github.EmergentOrder" %% "onnx-scala-backends" % "0.12.0",
//    libraryDependencies += "org.bytedeco" % "dnnl-platform" % "1.6.4-1.5.5-SNAPSHOT",
//    libraryDependencies += "com.github.rssh" %% "dotty-cps-async" % "0.2.1-RC1",
    libraryDependencies += scalaTest % Test,
    crossScalaVersions := Seq(dottyVersion, scala213Version)
  )


lazy val djl = (project in file("DJL"))
  .dependsOn(core)
  .settings(
    name := "ndscala-djl",
    libraryDependencies += ("org.typelevel" %% "spire" % "0.17.0").withDottyCompat(dottyVersion),
    libraryDependencies += "ai.djl" % "api" % "0.10.0",
//    libraryDependencies += "ai.djl.mxnet" % "mxnet-engine" % "0.10.0",
//    libraryDependencies += "ai.djl.mxnet" % "mxnet-native-auto" % "1.7.0-backport",
    libraryDependencies += "ai.djl.pytorch" % "pytorch-engine" % "0.10.0",
    libraryDependencies += "ai.djl.pytorch" % "pytorch-native-auto" % "1.7.1",
//    libraryDependencies += "ai.djl.tensorflow" % "tensorflow-engine" % "0.10.0",
//    libraryDependencies += "ai.djl.tensorflow" % "tensorflow-native-auto"% "2.3.1",
//    libraryDependencies += "ai.djl.paddlepaddle" % "paddlepaddle-engine" % "0.10.0",
//    libraryDependencies += "ai.djl.paddlepaddle" % "paddlepaddle-native-auto" % "2.0.0",
//    libraryDependencies += "ai.djl.tflite" % "tflite-engine" % "0.10.0",
//    libraryDependencies += "ai.djl.tflite" % "tflite-native-auto" % "2.4.1",
    //Only needed for Axes
    libraryDependencies += "com.github.EmergentOrder" %% "onnx-scala" % "0.12.0",
    libraryDependencies += scalaTest % Test,
    crossScalaVersions := Seq(dottyVersion, scala213Version)
  )
/*
lazy val mixTest = (project in file("mixtest"))
  .dependsOn(onnxscala, djl)
  .settings(
    name := "mixTest"
)
*/
