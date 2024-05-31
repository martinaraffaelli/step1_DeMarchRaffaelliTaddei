# SE4HPC DevOps Project: Step 1

## Directory Details

- `.vscode/`: Contains configuration files for the Visual Studio Code editor.
- `googletest/`: Submodule for the Google Test framework.
- `include/`: Contains the header files for the project.
- `lib/`: Contains the precompiled object code for the matrix multiplication library (`libmatrix_multiplication.a`).
- `src/`: Contains the source files, including a reference implementation of the matrix multiplication function.
- `test/`: Contains the test cases for matrix multiplication.
- `CMakeLists.txt`: CMake build configuration file.
- `build.sh`: Script to automate the build process.

## Testing

The goal of this step is to identify and automate the execution of test cases to detect errors in the matrix multiplication implementation using Google Test.
To do that we have extended the file `test_matrix_multiplication.cpp` with multiple test cases to spot various errors. 
Below is a description of the test suites and individual tests, along with the rationale behind their selection.

### `StandardMatrixMultiplicationTest` Test Suites

This suite groups tests based on different types of matrices created. These test cases aim to spot errors in typical scenarios with varying matrix dimensions.
In particular we have considered:

- `TEST(StandardMatrixMultiplicationTest, RectangularMatrices)`: both rectangular matrices
- `TEST(StandardMatrixMultiplicationTest, SquareMatrices)`: both square matrices
- `TEST(StandardMatrixMultiplicationTest, RectangularandSquareMatrices)`: a rectangular matrix and a square one
- `TEST(StandardMatrixMultiplicationTest, SquareandRectangularMatrices)`: a square matrix and a rectangular one
- `TEST(StandardMatrixMultiplicationTest, IntermediateRectangularMatrices)`: both square matrices with intermediate dimensions
- `TEST(StandardMatrixMultiplicationTest, BigRectangularMatrices)`: both square matrices with big dimensions



