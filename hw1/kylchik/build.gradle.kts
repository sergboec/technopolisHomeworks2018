plugins {
    java
    application
}

repositories {
    mavenCentral()
    jcenter()
}

dependencies {
    testCompile("junit", "junit", "4.12")
    compile("com.sparkjava","spark-core","2.7.2")
    compile("org.apache.spark","spark-core_2.11","2.3.2")

}

configure<JavaPluginConvention> {
    sourceCompatibility = JavaVersion.VERSION_1_8
}

application {
    mainClassName = "Main"
}