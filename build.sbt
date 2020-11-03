import scala.xml.transform.{RewriteRule, RuleTransformer}

val dottyVersion = "0.27.0-RC1"
val scala213Version = "2.13.4-bin-57ae2a6"

resolvers in Global += "scala-integration" at "https://scala-ci.typesafe.com/artifactory/scala-integration/"

ThisBuild / scalaVersion     := dottyVersion
ThisBuild / version          := "0.1.0-SNAPSHOT"
ThisBuild / organization     := "org.sciscala"
ThisBuild / organizationName := "sciscala"

Global / concurrentRestrictions := Seq(
  Tags.limit(Tags.Test, 1)
)

crossScalaVersions := Seq(dottyVersion, scala213Version)
lazy val scalaTest = ("org.scalatest" %% "scalatest" % "3.2.2")
lazy val core = (project in file("core"))
  .settings(
    name := "ndscala-core",
    scalacOptions += "-Ymacro-annotations",
    resolvers += Resolver.mavenLocal,
    resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots",
    libraryDependencies += ("org.typelevel" %% "simulacrum" % "1.0.1").withDottyCompat(dottyVersion), //TODO: Move to simulacrum-scalafix
    libraryDependencies += ("org.typelevel" %% "spire" % "0.17.0").withDottyCompat(dottyVersion),
    libraryDependencies += "com.github.EmergentOrder" %% "onnx-scala" % "0.8.0",
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
    libraryDependencies += "com.github.EmergentOrder" %% "onnx-scala-backends" % "0.8.0",
//    libraryDependencies += "org.bytedeco" % "dnnl-platform" % "1.6.4-1.5.5-SNAPSHOT",
//    libraryDependencies += "com.github.rssh" %% "dotty-cps-async" % "0.2.1-RC1",
    libraryDependencies += scalaTest % Test,
    crossScalaVersions := Seq(dottyVersion, scala213Version)
  )

lazy val tensorflow = (project in file("TensorFlow"))
  .dependsOn(core)
  .settings(
    name := "ndscala-tensorflow",
    libraryDependencies += ("org.typelevel" %% "spire" % "0.17.0").withDottyCompat(dottyVersion),
//TF-scala 0.5.7 broken, 0.5.1-SNAPSHOT with correct classifier works
    //Only needed for Axes
    libraryDependencies += "com.github.EmergentOrder" %% "onnx-scala" % "0.8.0",
  //Local only  
    libraryDependencies += ("org.platanios" %% "tensorflow" % "0.5.7" classifier "linux").withDottyCompat(dottyVersion),
    libraryDependencies += scalaTest % Test,
    crossScalaVersions := Seq(dottyVersion, scala213Version)
  )

lazy val djl = (project in file("DJL"))
  .dependsOn(core)
  .settings(
    name := "ndscala-djl",
    libraryDependencies += ("org.typelevel" %% "spire" % "0.17.0").withDottyCompat(dottyVersion),
    libraryDependencies += "ai.djl" % "api" % "0.8.0",
    libraryDependencies += "ai.djl.mxnet" % "mxnet-engine" % "0.8.0", 
    libraryDependencies += "ai.djl.mxnet" % "mxnet-native-mkl" % "1.7.0-backport",
    //Only needed for Axes
    libraryDependencies += "com.github.EmergentOrder" %% "onnx-scala" % "0.8.0",
    libraryDependencies += scalaTest % Test,
    crossScalaVersions := Seq(dottyVersion, scala213Version)
  )

lazy val mixTest = (project in file("mixtest"))
  .dependsOn(onnxscala, tensorflow, djl)
  .settings(
    name := "mixTest"
)

