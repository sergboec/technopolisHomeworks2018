plugins {
    scala
}

repositories {
    mavenCentral()
}

dependencies {
    testCompile("junit", "junit", "4.12")
    compile("org.scala-lang:scala-library:2.12.8")
    compile("org.apache.spark","spark-core_2.12","2.4.0")
    compile("org.apache.spark", "spark-mllib_2.12", "2.4.0")
    compile("org.apache.spark", "spark-sql_2.12", "2.4.0")
}

