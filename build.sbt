import scala.xml.transform.{RewriteRule, RuleTransformer}
import sbtcrossproject.CrossPlugin.autoImport.{crossProject, CrossType}

val scala3Version = "3.2.1"
val scalaTestVersion = "3.2.14"

ThisBuild / scalaVersion     := scala3Version
ThisBuild / version          := "0.1.0-SNAPSHOT"
ThisBuild / organization     := "org.sciscala"
ThisBuild / organizationName := "sciscala"

Global / concurrentRestrictions := Seq(
  Tags.limit(Tags.Test, 1)
)


lazy val core = (crossProject(JSPlatform, JVMPlatform) //, NativePlatform)
  .crossType(CrossType.Pure) in file("core"))
  .settings(
    name := "ndscala-core",
//    scalacOptions += "-release:19",
    scalacOptions += "-source:3.2",
    resolvers += Resolver.mavenLocal,
    resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots",
    libraryDependencies += ("org.typelevel" %%% "spire" % "0.18.0"),
    libraryDependencies += "org.emergent-order" %%% "onnx-scala" % "0.17.0"
  //Local only  
//  libraryDependencies += "io.kjaer" % "tf-dotty-compiletime" % "0.0.0+134-f1f8d0ba+20201020-1123-SNAPSHOT",
//    libraryDependencies += "io.kjaer" % "tf-dotty-compiletime_0.27" % "0.0.0+134-f1f8d0ba+20201020-1123-SNAPSHOT",
//    libraryDependencies += scalaTest % Test,
   )
   .jsSettings(
   scalaJSStage := FullOptStage
   )
lazy val onnxscala = (crossProject(JSPlatform, JVMPlatform) //, NativePlatform)
  .crossType(CrossType.Pure) in file("ONNXScala"))
  .dependsOn(core)
  .settings(
//    javaOptions += "-Dcats.effect.stackTracingMode=full -Dcats.effect.traceBufferSize=1024",
    name := "ndscala-onnx-scala",
    scalacOptions += "-release:19",
    scalacOptions += "-source:3.2",
    libraryDependencies += ("org.typelevel" %%% "spire" % "0.18.0"),
    libraryDependencies += "org.emergent-order" %%% "onnx-scala-backends" % "0.17.0",
    libraryDependencies += ("org.scalatest" %%% "scalatest" % scalaTestVersion) % Test,
     libraryDependencies += "org.typelevel" %%% "cats-effect-testing-scalatest" % "1.5.0" % Test
//    libraryDependencies += "org.bytedeco" % "dnnl-platform" % "1.6.4-1.5.5-SNAPSHOT",
//    libraryDependencies += "com.github.rssh" %% "dotty-cps-async" % "0.2.1-RC1",
//    libraryDependencies += scalaTest % Test
  )
   .jsSettings(
     webpack / version := "5.74.0",
     webpackCliVersion := "4.10.0",
     startWebpackDevServer / version := "4.11.1",
     scalaJSUseMainModuleInitializer                := true, // , //Testing
     Compile / npmDependencies += "onnxruntime-node" -> "1.13.1",
     Compile / npmDependencies += "onnxruntime-common" -> "1.13.1",
     libraryDependencies += "org.typelevel" %%% "cats-effect-testing-scalatest" % "1.5.0" % Test,
     stOutputPackage := "org.emergentorder.ndscala",
     scalaJSStage := FullOptStage
   )
   .jsConfigure { project => project.enablePlugins(ScalablyTypedConverterPlugin) } //For distribution as a library use: ScalablyTypedConverterGenSourcePlugin

