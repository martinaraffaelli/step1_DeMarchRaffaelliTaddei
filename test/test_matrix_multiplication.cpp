#include "matrix_multiplication.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include <cmath>
#include <climits>
#include <cstdlib> // for std::rand
#include "../src/matrix_mult.cpp"

const int N_max = 10;
const int N_min = 1;
// ######################### Source code of multiplyMatrices in src/matrix_mult

// Fills the matrix with random values of the standard interval [-5, 10]
void fillMatrixRandomly(std::vector<std::vector<int>>& A) {
    for (auto& row : A) {
        for (auto& element : row) {
            element = (std::rand() % 15) - 5; // Generate random number between 0 and 99
        }
    }
}
// Fills the matrix with random values of the interval [lowerLimit, upperLimit]
void fillMatrixRandomly(std::vector<std::vector<int>>& A, int lowerLimit, int upperLimit) {
    int range = upperLimit - lowerLimit + 1;
    for (auto& row : A) {
        for (auto& element : row) {
            element = lowerLimit + std::rand() % range; // Generate random number between lowerLimit and upperLimit
        }
    }
}
// Generates a random int of the standard interval [1, 10]
int generateRandomNumber() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(N_min, N_max);
    return dis(gen);
}
// Generates a random int of the interval [lowerLimit, upperLimit]
int generateRandomNumber(int lowerLimit, int upperLimit) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(lowerLimit, upperLimit);
    return dis(gen);
}


//StandardMatrixMultiplicationTest is the suite test that groups the tests with random
//matrix generation

//The tests that are included in this suite differ in the types of matrices that are created,
//in particular we considered: 
// - both rectangular matrices
// - both square matrices
// - a rectangular matrix and a square one
// - a square matrix and a rectangular one

TEST(StandardMatrixMultiplicationTest, RectangularMatrices) {

    // Randomly generate the dimensions of A
    int rowsA = generateRandomNumber();
    int colsA = generateRandomNumber();

    // Create A matrix 
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill A
    fillMatrixRandomly(A);
    
    // Set proper dim for B
    int rowsB = colsA;
    int colsB = generateRandomNumber();

    // Create B matrix 
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill B
    fillMatrixRandomly(B);

    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(StandardMatrixMultiplicationTest, SquareMatrices) {

    // Randomly generate dimensions of A
    int rowsA = generateRandomNumber();
    int colsA = rowsA;

    // Create A matrix with random dim
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill A
    fillMatrixRandomly(A);
    
    // Set proper dim for B
    int rowsB = colsA;
    int colsB = colsA;

    // Create B matrix with random dim
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill B
    fillMatrixRandomly(B);

    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(StandardMatrixMultiplicationTest, RectangularandSquareMatrices) {

    // Randomly generate the dimensions of A
    int rowsA = generateRandomNumber();
    int colsA = generateRandomNumber();

    // Create the A matrix 
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill A
    fillMatrixRandomly(A);

    // Set proper dim for B matrix
    int rowsB = colsA;
    int colsB = rowsB;

    // Create the B matrix
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill B
    fillMatrixRandomly(B);

    // Create C matrix with 0 values 
    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);


    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(StandardMatrixMultiplicationTest, SquareandRectangularMatrices) {

    // Randomly generate a dimension of A
    int rowsA = generateRandomNumber();
    int colsA = rowsA;

    // Create the A matrix 
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill the matrix with casual values
    fillMatrixRandomly(A);

    // Set proper dim for B matrix
    int rowsB = colsA;
    int colsB = generateRandomNumber();

    // Create B matrix 
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill B
    fillMatrixRandomly(B);

    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}


//ExtremeMatrixMultiplicationTest is the suite test that groups the specific cases we want to test,
//such as:
// - vectors 
// - matrix and vector 
// - vector and matrix 
// - scalars 
// - identity matrices 
// - odd and even dimensions 
// - null matrices 
// - nan matrices 
// - overflow

TEST(CriticalMatrixMultiplicationTest, Vectors) {

    // Randomly generate the dimensions of A
    int rowsA = 1;
    int colsA = generateRandomNumber();

    // Create A matrix 
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill A
    fillMatrixRandomly(A);

    // Set proper dim for B matrix
    int rowsB = colsA;
    int colsB = 1;

    // Create B matrix 
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill B
    fillMatrixRandomly(B);

    // Create C matrix with 0 values 
    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);


    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(CriticalMatrixMultiplicationTest, VectorMatrix) {

    // Randomly generate the dimensions of A
    int rowsA = 1;
    int colsA = generateRandomNumber();

    // Create A matrix 
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill A
    fillMatrixRandomly(A);

    // Set proper dim for B matrix
    int rowsB = colsA;
    int colsB = generateRandomNumber();

    // Create B matrix
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill the matrix with casual values
    fillMatrixRandomly(B);

    // Create C matrix with 0 values 
    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);


    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(CriticalMatrixMultiplicationTest, MatrixVector) {
    
    // Randomly generate the dimensions of A
    int rowsA = generateRandomNumber();
    int colsA = generateRandomNumber();

    // Create A matrix
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill the matrix with casual values
    fillMatrixRandomly(A);

    // Set proper dim for B matrix
    int rowsB = colsA;
    int colsB = 1;

    // Create B matrix
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill the matrix with casual values
    fillMatrixRandomly(B);

    // Create C matrix with 0 values 
    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);


    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(CriticalMatrixMultiplicationTest, Scalars) {

    // Randomly generate the dimensions of A
    int rowsA = 1;
    int colsA = 1;

    // Create A matrix
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill A
    fillMatrixRandomly(A);

    // Set proper dim for B matrix
    int rowsB = 1;
    int colsB = 1;

    // Create B matrix
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill B
    fillMatrixRandomly(B);

    // Create C matrix with 0 values 
    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);


    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(CriticalMatrixMultiplicationTest, ScalarVector) {

    // Randomly generate the dimensions of A
    int rowsA = 1;
    int colsA = 1;

    // Create A matrix 
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill A
    fillMatrixRandomly(A);

    // Set proper dim for B matrix
    int rowsB = 1;
    int colsB = generateRandomNumber();

    // Create B matrix
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill B
    fillMatrixRandomly(B);

    // Create C matrix with 0 values 
    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);


    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(CriticalMatrixMultiplicationTest, OddDimMatrices) {

    // Randomly generate the dimensions of A
    int rowsA = 2 * generateRandomNumber(1, 10) + 1;
    int colsA = 2 * generateRandomNumber(1, 10) + 1;

    // Create A matrix 
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill A
    fillMatrixRandomly(A);

    // Set proper dim for B matrix
    int rowsB = colsA;
    int colsB = 2 * generateRandomNumber(1, 10) + 1;

    // Create B matrix
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill B
    fillMatrixRandomly(B);

    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(CriticalMatrixMultiplicationTest, EvenDimMatrices) {
    
    // Randomly generate the dimensions of A
    int rowsA = 2 * generateRandomNumber();
    int colsA = 2 * generateRandomNumber();

    // Create A matrix 
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill A
    fillMatrixRandomly(A);

    // Set proper dim for B matrix
    int rowsB = colsA;
    int colsB = 2 * generateRandomNumber();

    // Create B matrix
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill B
    fillMatrixRandomly(B);

    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(CriticalMatrixMultiplicationTest, OverflowMatrices) {

    // Randomly generate the dimensions of A
    int rowsA = generateRandomNumber();
    int colsA = generateRandomNumber();

    // Create A matrix 
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill A
    fillMatrixRandomly(A, 2e32, 2e32 + 1);

    // Set proper dim for B matrix
    int rowsB = colsA;
    int colsB = generateRandomNumber();

    // Create B matrix
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill B
    fillMatrixRandomly(B);

    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(CriticalMatrixMultiplicationTest, IdentityMatrix) {

    // Randomly generate the dimensions of A
    int rowsA = generateRandomNumber();
    int colsA = rowsA;

    // Create A matrix 
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(rowsA, 0));

    // Fill A
    for (int i = 0; i < rowsA; ++i) {
        A[i][i] = 1;
    }

    // Set proper dim for B matrix
    int rowsB = colsA;
    int colsB = generateRandomNumber();

    // Create B matrix
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill B
    fillMatrixRandomly(B);

    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(CriticalMatrixMultiplicationTest, SquareZerosMatrices) {

    // Randomly generate dimensions of A
    int rowsA = generateRandomNumber();
    int colsA = rowsA;

    // Create A matrix
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA, 0));

    // Set proper dim for B
    int rowsB = colsA;
    int colsB = colsA;

    // Create B matrix
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB, 0));

    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(CriticalMatrixMultiplicationTest, SquareOnesMatrices) {

    // Randomly generate dimensions of A
    int rowsA = generateRandomNumber();
    int colsA = rowsA;

    // Create A matrix 
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA, 1));

    // Set proper dim for B
    int rowsB = colsA;
    int colsB = colsA;

    // Create B matrix 
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB, 1));

    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(CriticalMatrixMultiplicationTest, SmallValuesRectangularMatrices) {

    // Randomly generate the dimensions of A
    int rowsA = generateRandomNumber();
    int colsA = generateRandomNumber();

    // Create A matrix 
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill A
    fillMatrixRandomly(A, 0, 10);

    // Set proper dim for B matrix
    int rowsB = colsA;
    int colsB = generateRandomNumber();

    // Create B matrix 
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill B
    fillMatrixRandomly(B, 0, 10);

    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(CriticalMatrixMultiplicationTest, IntermediateValuesRectangularMatrices) {

    // Randomly generate the dimensions of A
    int rowsA = generateRandomNumber();
    int colsA = generateRandomNumber();

    // Create A matrix 
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill A
    fillMatrixRandomly(A, 11, 20);

    // Set proper dim for B matrix
    int rowsB = colsA;
    int colsB = generateRandomNumber();

    // Create B matrix 
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill B
    fillMatrixRandomly(B, 11, 20);

    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(CriticalMatrixMultiplicationTest, BigValuesRectangularMatrices) {

    // Randomly generate the dimensions of A
    int rowsA = generateRandomNumber();
    int colsA = generateRandomNumber();

    // Create A matrix 
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill A
    fillMatrixRandomly(A, 75, 125);

    // Set proper dim for B matrix
    int rowsB = colsA;
    int colsB = generateRandomNumber();

    // Create B matrix
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill B
    fillMatrixRandomly(B, 75, 125);

    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(CriticalMatrixMultiplicationTest, NegativeValuesRectangularMatrices) {

    // Randomly generate the dimensions of A
    int rowsA = generateRandomNumber();
    int colsA = generateRandomNumber();

    // Create A matrix
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill A
    fillMatrixRandomly(A, -10, 0);

    // Set proper dim for B matrix
    int rowsB = colsA;
    int colsB = generateRandomNumber();

    // Create B matrix
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill B
    fillMatrixRandomly(B, -10, 0);

    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

/*
TEST(CriticalMatrixMultiplicationTest, NULLMatrices) {

    // Create A matrix 
    std::vector<std::vector<int>> A;

    // Set proper dim for B matrix
    int rowsB = generateRandomNumber();
    int colsB = generateRandomNumber();

    // Create B matrix
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill B
    fillMatrixRandomly(B);

    std::vector<std::vector<int>> C(0, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, 0, 0, colsB);

    std::vector<std::vector<int>> expected(0, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,0,0,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}
*/

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
