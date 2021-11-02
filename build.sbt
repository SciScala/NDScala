import scala.xml.transform.{RewriteRule, RuleTransformer}

//val dottyVersion = dottyLatestNightlyBuild.get
val scala3Version = "3.1.0"
val scala213Version = "2.13.6"

resolvers in Global += "scala-integration" at "https://scala-ci.typesafe.com/artifactory/scala-integration/"

ThisBuild / scalaVersion     := scala3Version
ThisBuild / version          := "0.1.0-SNAPSHOT"
ThisBuild / organization     := "org.sciscala"
ThisBuild / organizationName := "sciscala"

Global / concurrentRestrictions := Seq(
  Tags.limit(Tags.Test, 1)
)

crossScalaVersions := Seq(scala3Version, scala213Version)
lazy val scalaTest = ("org.scalatest" %% "scalatest" % "3.2.9")
lazy val core = (project in file("core"))
  .settings(
    name := "ndscala-core",
    scalacOptions += "-Ymacro-annotations",
    resolvers += Resolver.mavenLocal,
    resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots",
    libraryDependencies += ("org.typelevel" %% "spire" % "0.18.0-M1"),
    libraryDependencies += "org.emergent-order" %% "onnx-scala" % "0.15.0",
  //Local only  
//  libraryDependencies += "io.kjaer" % "tf-dotty-compiletime" % "0.0.0+134-f1f8d0ba+20201020-1123-SNAPSHOT",
//    libraryDependencies += "io.kjaer" % "tf-dotty-compiletime_0.27" % "0.0.0+134-f1f8d0ba+20201020-1123-SNAPSHOT",
//    libraryDependencies += scalaTest % Test,
    crossScalaVersions := Seq(scala3Version, scala213Version)
   )
lazy val onnxscala = (project in file("ONNXScala"))
  .dependsOn(core)
  .settings(
    name := "ndscala-onnx-scala",
    libraryDependencies += ("org.typelevel" %% "spire" % "0.18.0-M1"),
    libraryDependencies += "org.emergent-order" %% "onnx-scala-backends" % "0.15.0",
//    libraryDependencies += "org.bytedeco" % "dnnl-platform" % "1.6.4-1.5.5-SNAPSHOT",
//    libraryDependencies += "com.github.rssh" %% "dotty-cps-async" % "0.2.1-RC1",
    libraryDependencies += scalaTest % Test,
    crossScalaVersions := Seq(scala3Version, scala213Version)
  )
