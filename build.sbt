import scala.xml.transform.{RewriteRule, RuleTransformer}
import Dependencies._

ThisBuild / scalaVersion     := "2.13.1"
ThisBuild / version          := "0.1.0-SNAPSHOT"
ThisBuild / organization     := "org.sciscala"
ThisBuild / organizationName := "sciscala"

lazy val core = (project in file("core"))
  .settings(
    name := "ndscala-core",
    scalacOptions += "-Ymacro-annotations",
    resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots",
    libraryDependencies += simulacrum, //TODO: Move to simulacrum-scalafix
    libraryDependencies += "org.typelevel" %% "spire" % "0.17.0-M1", 
    libraryDependencies += scalaTest % Test
  )

lazy val onnxscala = (project in file("ONNXScala"))
  .dependsOn(core)
  .settings(
    name := "ndscala-onnx-scala",
    libraryDependencies += "org.typelevel" %% "spire" % "0.17.0-M1",
    libraryDependencies += "com.github.EmergentOrder" %% "onnx-scala-backends" % "0.4.0",
    libraryDependencies += scalaTest % Test
  )
