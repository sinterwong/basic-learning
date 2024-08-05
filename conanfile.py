from conan import ConanFile
from conan.tools.cmake import CMakeToolchain


class BasicLearnConan(ConanFile):
    name = "basic-learning"
    version = "v0.0.1"
    url = "https://github.com/sinterwong/basic-learning"
    description = "A self-made C++ learning framework for easily learning basic knowledge and the usage of various libraries."
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps", "CMakeToolchain"

    options = {"shared": [True, False], "fPIC": [True, False]}
    default_options = {"shared": False, "fPIC": True}

    def config_options(self):
        if self.settings.os == "Windows":
            del self.options.fPIC

    def configure(self):
        self.settings.compiler.libcxx = "libstdc++11"
        self.settings.compiler.cppstd = "20"
        self.settings.compiler.version = "13"

        if self.options.shared:
            del self.options.fPIC

        # self.options["libcurl"].system_libs = False

        self.options["spdlog"].use_std_fmt = True

    def requirements(self):
        self.requires("pcl/1.13.1")
        self.requires("gflags/2.2.2")
        self.requires("spdlog/1.14.1")
        self.requires("taskflow/3.7.0")
        self.requires("eigen/3.4.0")
        self.requires("libcurl/8.8.0")

    def layout(self):
        self.folders.build = "build"
        self.folders.generators = "build/generators"
