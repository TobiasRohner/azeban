# azeban: a spectral viscosity method

## Dependencies
On Linux all dependencies of azeban can be installed as follows

    bin/install_dependencies.sh COMPILER third_pary --zisa_has_cuda=1

which will propose a partial CMake command to configure the build system.
It is recommended to additionally at least choose a build type, e.g.
`-DCMAKE_BUILD_TYPE=Debug`. Valid build types include `FastDebug` and
`Release`.

Dependencies fall into three categories. The first are common dependencies
which are either hard to compile or otherwise difficult to install. Since these
dependencies are common, they are usually present on any system suitable for
HPC. We shall assume they are present on the machine, e.g. though the package
manager or though the infamous `module` system. We find these dependencies using
the CMake standard `find_package`.

The next type of dependencies are common, yet easy to distribute, dependencies.
We can use Conan to install these dependencies.

The third type of dependencies are internal dependencies, in this case on our
family of libraries called zisa. They can either be cloned, built and installed
like any other source dependency; or one can use a 'super build' to integrate
them more tightly with azeban.

The fourth type of dependencies are highly specialized HPC codes. We'll try and
avoid those.

### Conan
Conan is a package manager for C++, not unlike `pip install --user`. Conan
itself is a Python package and should be installed through `pip`, e.g.

    pip install --user conan
    
One can then use `conan` to install the dependencies listed in `conanfile.txt`
as follows

    conan install AZEBAN_DIR
    
However, a quick inspection of `conanfile.txt` shows no indication of which
compiler is to be used. Nor any other information relevant to ABI. Conan
resolves this through so called profiles. As a user of Conan you define the
compiler, compiler version, version of libc++ and any else that is relevant in a
profile. You then install the dependencies for that profile. In a different
context "profile" might be referred to as a toolchain.

Probably, the only stumbling block will be that we use the C++11 (or newer) and
therefore our dependencies must be built again a C++11 version of the C++
standard library. Therefore, we should use

    conan install AZEBAN_DIR -s compiler.libcxx=libstdc++11

to download the correct version of the dependencies.

The final thing to know about Conan is that it installs the requested versions
of the libraries in one local folder. Thereby building a little repository of
installed libraries. Additionally, it will create a couple of files in the
current working directory. These files are needed to instruct CMake which
libraries where it can find the dependencies listed in the `conanfile.txt`.

### Internal dependencies
azeban reuses code from zisa. This repository does not directly include these
dependencies. They are cloned, built and installed through
`bin/install_dependencies.sh`. In this repository, we treat them like any other
third-party dependency which needs to installed from source.

If the need arises to regularly modify parts of zisa, then there are two
options. The first is to embrace the cycle of working on the library in
isolation. Then commit the changes, and reinstall the dependency.

The second option is to use the development repository.

### Development repository
An alternate option of organizing code is to use a super build. Which has the
advantage of combining the source of some or all internal dependencies. An
implementation of this is available at

&emsp;github.com/1uc/azeban-dev.git

## Using CMake
Since important improvements specifically concerning CUDA where made in version
3.18, we need what is currently an almost cutting edge version of CMake. If it is
not installed, on Linux, it can be installed using the following script

    bin/install_cmake.sh
    
### Listing Source Files
We will not glob source files, and instead list them manually. Naturally,
keeping the corresponding `CMakeLists.txt` up to date manually is unacceptable.
Hence, there is a Python script to take care of keeping the files updated. After
adding a new source file, execute

    bin/update_cmake.py
