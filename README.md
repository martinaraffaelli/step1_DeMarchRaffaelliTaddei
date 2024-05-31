# SE4HPC DevOps Project

## Step 1: Testing

The goal of this step is to identify and automate the execution of test cases to detect errors in the matrix multiplication implementation. We use Google Test for this purpose.

### Matrix Multiplication Function

The matrix multiplication function we are testing has the following signature:

```cpp
void multiplyMatrices(const std::vector<std::vector<int>>& A,
                      const std::vector<std::vector<int>>& B,
                      std::vector<std::vector<int>>& C,
                      int rowsA, int colsA, int colsB);

prova
