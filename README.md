# SE4HPC DevOps Project: Step 1

## Directory Details

- `.vscode/`: Contains configuration files for the Visual Studio Code editor.
- `googletest/`: Submodule for the Google Test framework.
- `include/`: Contains the header files for the project.
- `lib/`: Contains the precompiled object code for the matrix multiplication library (`libmatrix_multiplication_with_errors.a`).
- `src/`: Contains the source files, including a reference implementation of the matrix multiplication function (`matrix_mult.cpp`).
- `test/`: Contains the test cases for matrix multiplication.
- `CMakeLists.txt`: CMake build configuration file.
- `build.sh`: Script to automate the build process.

## Testing

The goal of this step is to identify and automate the execution of test cases to detect errors in the matrix multiplication implementation using Google Test.
To do that we have extended the file `test_matrix_multiplication.cpp` with multiple test cases to spot various errors. 
Below is a description of the test suites and individual tests, along with the rationale behind their selection.

### `StandardMatrixMultiplicationTest` Test Suites

This suite groups tests based on different types of matrices created. 
These test cases aim to spot errors in typical scenarios with varying matrix dimensions.
Testing various matrix dimensions ensures the function can handle different shapes and sizes, spotting dimension-related errors.
In particular we considered:

- `TEST(StandardMatrixMultiplicationTest, RectangularMatrices)`

This test checks if the function correctly handles rectangular matrices, which is a common use case.
- `TEST(StandardMatrixMultiplicationTest, SquareMatrices)`

This test checks if the function correctly handles square matrices, which is another common use case.
- `TEST(StandardMatrixMultiplicationTest, RectangularandSquareMatrices)`

This test examines the function's correctness with a rectangular matrix and a square matrix.
- `TEST(StandardMatrixMultiplicationTest, SquareandRectangularMatrices)`

This test examines the function's correctness with a square matrix and a rectangular matrix, which is the previous test case with reversed order of multiplication.
- `TEST(StandardMatrixMultiplicationTest, IntermediateRectangularMatrices)`

This test checks if the function can correctly handle intermediate-sized rectangular matrices (11 to 20 rows/cols).
- `TEST(StandardMatrixMultiplicationTest, BigRectangularMatrices)`

This test checks if the function can correctly handle large rectangular matrices (21 to 100 rows/cols).

### `CriticalMatrixMultiplicationTest`Test Suites

This suite is designed to group specific cases aimed at detecting more subtle errors: those concerning their dimensions (Structural Tests), but particularly those related to the values within the matrices (Value-Based Tests). 

#### Structural Tests
These tests focus on the structure and specific types of matrices (e.g.: vectors, that are Nx1 matrices, and scalars that are 1x1 matrices), to uncover potential edge cases in matrix multiplication.

- `TEST(CriticalMatrixMultiplicationTest, Vectors)`

This test verifies the function's handling of vector multiplication.
- `TEST(CriticalMatrixMultiplicationTest, VectorMatrix)`

This test checks multiplication between a vector and a matrix.
- `TEST(CriticalMatrixMultiplicationTest, MatrixVector)`

This test checks multiplication between a matrix and a vector.
- `TEST(CriticalMatrixMultiplicationTest, Scalars)`

This test verifies the function's handling of scalar multiplication.
- `TEST(CriticalMatrixMultiplicationTest, ScalarVector)`

This test checks multiplication between a scalar and a vector.
- `TEST(CriticalMatrixMultiplicationTest, OddDimMatrices)`

This test examines matrices with odd dimensions to detect any odd-dimension related issues.
- `TEST(CriticalMatrixMultiplicationTest, EvenDimMatrices)`

This test examines matrices with even dimensions to detect any even-dimension related issues.

#### Value-Based Tests
In addition to the above, we've focused on specific numerical values within matrices to identify errors that depend on element values. 
For this suite we've used an *incremental approach*, starting with common scenarios (e.g.: Identity Matrices) and moving towards more specific ones (e.g.: SquareZerosMatrices). This approach was effective in detecting all 20 errors by running the function with matrices containing specific sets of values.

- `TEST(CriticalMatrixMultiplicationTest, IdentityMatrix)`

This test uses identity matrices to check if the multiplication results are correct when using an identity matrix.
- `TEST(CriticalMatrixMultiplicationTest, SquareZerosMatrices)`

This test checks the function's handling of matrices filled with zeros.
- `TEST(CriticalMatrixMultiplicationTest, SquareOnesMatrices)`

This test examines the function's performance with matrices filled with ones.
- `TEST(CriticalMatrixMultiplicationTest, SmallValuesRectangularMatrices)`

This test uses rectangular matrices with small values (0 to 10) to detect issues with small numerical values.
- `TEST(CriticalMatrixMultiplicationTest, IntermediateValuesRectangularMatrices)`

This test uses rectangular matrices with intermediate values (11 to 20) to spot errors related to medium-sized values.
- `TEST(CriticalMatrixMultiplicationTest, BigValuesRectangularMatrices)`

This test uses rectangular matrices with large values (75 to 125) to detect issues with large numerical values.
- `TEST(CriticalMatrixMultiplicationTest, NegativeValuesRectangularMatrices)`

This test uses rectangular matrices with negative values (-10 to 0) to uncover problems related to negative numbers.

### Errors Detected
Running the test cases many time, to generate as many random values as possible inside the matrices, we've detected the following errors:

- Error 1: Element-wise multiplication of ones detected!
- Error 2: Matrix A contains the number 7!
- Error 3: Matrix A contains a negative number!
- Error 4: Matrix B contains the number 3!
- Error 5: Matrix B contains a negative number!
- Error 6: Result matrix contains a number bigger than 100!
- Error 7: Result matrix contains a number between 11 and 20!
- Error 8: Result matrix contains zero!
- Error 9: Result matrix contains the number 99!
- Error 10: A row in matrix A contains more than one '1'!
- Error 11: Every row in matrix B contains at least one '0'!
- Error 12: The number of rows in A is equal to the number of columns in B!
- Error 13: The first element of matrix A is equal to the first element of matrix B!
- Error 14: The result matrix C has an even number of rows!
- Error 15: A row in matrix A is filled entirely with 5s!
- Error 16: Matrix B contains the number 6!
- Error 17: Result matrix C contains the number 17!
- Error 18: Matrix A is a square matrix!
- Error 19: Every row in matrix A contains the number 8!
- Error 20: Number of columns in matrix A is odd!
