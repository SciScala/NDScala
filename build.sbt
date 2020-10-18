import scala.xml.transform.{RewriteRule, RuleTransformer}
import Dependencies._

ThisBuild / scalaVersion     := "2.13.3"
ThisBuild / version          := "0.1.0-SNAPSHOT"
ThisBuild / organization     := "org.sciscala"
ThisBuild / organizationName := "sciscala"

lazy val core = (project in file("core"))
  .settings(
    name := "ndscala-core",
    scalacOptions += "-Ymacro-annotations",
    resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots",
    libraryDependencies += simulacrum, //TODO: Move to simulacrum-scalafix
    libraryDependencies += "org.typelevel" %% "spire" % "0.17.0",
    libraryDependencies += scalaTest % Test
  )

lazy val onnxscala = (project in file("ONNXScala"))
  .dependsOn(core)
  .settings(
    name := "ndscala-onnx-scala",
    libraryDependencies += "org.typelevel" %% "spire" % "0.17.0",
    libraryDependencies += "com.github.EmergentOrder" %% "onnx-scala-backends" % "0.7.0",
    libraryDependencies += scalaTest % Test
  )

def tensorflowOSClassifier: String = System.getProperty("os.name").toLowerCase match {
    case mac if mac.contains("mac")  => "darwin"
    case win if win.contains("win") => "windows"
    case linux if linux.contains("linux") => "linux"
    case osName => throw new RuntimeException(s"Unknown operating system $osName")
}

lazy val tensorflow = (project in file("TensorFlow"))
  .dependsOn(core)
  .settings(
      name := "ndscala-tensorflow",
      libraryDependencies += "org.typelevel" %% "spire" % "0.17.0",
      libraryDependencies += "org.platanios" %% "tensorflow" % "0.5.7" classifier tensorflowOSClassifier,
      libraryDependencies += scalaTest % Test
  )

lazy val djl = (project in file("DJL"))
  .dependsOn(core)
  .settings(
    name := "ndscala-djl",
    libraryDependencies += "org.typelevel" %% "spire" % "0.17.0",
    libraryDependencies += "ai.djl" % "api" % "0.8.0",
    libraryDependencies += "ai.djl.mxnet" % "mxnet-engine" % "0.8.0",
    libraryDependencies += "ai.djl.mxnet" % "mxnet-native-auto" % "1.7.0-backport",

    libraryDependencies += scalaTest % Test
  )
