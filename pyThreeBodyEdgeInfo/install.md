## Compadre
Since Kokkos can be automatically installed by _Compadre_, there is no need to install _Trilinos_ in advance.

Before the installation, the target folder should be created as
```
mkdir pythonAuxiliary
```
new version of CMake and GCC should be invoked by
```
module load CMake/3.16.5
module load GCC/8.3.0
```

Download _Compadre_
```
git clone https://github.com/sandialabs/compadre.git
```

Change the installation directory of _Compadre_ by editing the bash file with the command
```
cd compadre
mkdir build
cp ./script/basic_configure.sh build
cd build
vi basic_configure.sh
```
Change line as
```
'~\pythonAuxiliary'
```

Install _Compadre_
```
./basic_configure.sh
make install -j
```

