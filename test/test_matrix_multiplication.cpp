#include "matrix_multiplication.h"
#include <iostream>
#include <vector>
#include <gtest/gtest.h>
#include <algorithm>
#include <random>
#include "../src/matrix_mult.cpp"
// ######################### Source code of multiplyMatrices in src/matrix_mult

//StandardMatrixMultiplicationTest is the suite test that groups the tests with random
//matrix generation

//The tests that are included in this suite differ in the types of matrices that are created,
//in particular we considered: 
// - both rectangular matrices
// - both square matrices
// - a rectangular matrix and a square one
// - a square matrix and a rectangular one

TEST(StandardMatrixMultiplicationTest, RectangularMatrices) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_dim(1, 10); // dimesions interval
    std::uniform_int_distribution<> dis_val(-2, 2); // values interval

    // Randomly generate the dimensions of A
    int rowsA = dis_dim(gen);
    int colsA = dis_dim(gen);

    // Create A matrix with random dim
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill the matrix with random values
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
            A[i][j] = dis_val(gen);
        }
    }

    // Set proper dim for B matrix
    int rowsB = colsA;
    int colsB = dis_dim(gen);

    // Create B matrix with random dim
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // fill the matrix with random values
    for (int i = 0; i < rowsB; ++i) {
        for (int j = 0; j < colsB; ++j) {
            B[i][j] = dis_val(gen);
        }
    }

    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(StandardMatrixMultiplicationTest, SquareMatrices) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_dim(1, 10); // dimensions interval
    std::uniform_int_distribution<> dis_val(-2, 2); // values interval

    // Randomly generate dimensions of A
    int rowsA = dis_dim(gen);
    int colsA = rowsA;

    // Create A matrix with random dim
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill the matrix with random values 
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
            A[i][j] = dis_val(gen);
        }
    }

    // Set proper dim for B
    int rowsB = colsA;
    int colsB = colsA;

    // Create B matrix with random dim
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill the matrix with random values
    for (int i = 0; i < rowsB; ++i) {
        for (int j = 0; j < colsB; ++j) {
            B[i][j] = dis_val(gen);
        }
    }

    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}


TEST(StandardMatrixMultiplicationTest, RectangularandSquareMatrices) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_dim(1, 10);  // dimensions interval 
    std::uniform_int_distribution<> dis_val(-2, 2); // values interval

    // Randomly generate the dimensions of A
    int rowsA = dis_dim(gen);
    int colsA = dis_dim(gen);

    // Create the A matrix with random dim
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill the matrix with casual values
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
            A[i][j] = dis_val(gen);
        }
    }

    // Set proper dim for B matrix
    int rowsB = colsA;
    int colsB = rowsB;

    // Create the B matrix with random dim
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill the matrix with casual values
    for (int i = 0; i < rowsB; ++i) {
        for (int j = 0; j < colsB; ++j) {
            B[i][j] = dis_val(gen);
        }
    }

    // Create C matrix with 0 values 
    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);


    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(StandardMatrixMultiplicationTest, SquareandRectangularMatrices) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_dim(1, 10); // dimensions interval
    std::uniform_int_distribution<> dis_val(-2, 2); // values interval

    // Randomly generate a dimension of A
    int rowsA = dis_dim(gen);
    int colsA = rowsA;

    // Create the A matrix with random dim
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill the matrix with casual values
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
            A[i][j] = dis_val(gen);
        }
    }

    // Set proper dim for B matrix
    int rowsB = colsA;
    int colsB = dis_dim(gen);

    // Create B matrix with random dim
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // Fill the matrix with random values
    for (int i = 0; i < rowsB; ++i) {
        for (int j = 0; j < colsB; ++j) {
            B[i][j] = dis_val(gen);
        }
    }

    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

//ExtremeMatrixMultiplicationTest is the suite test that groups the specific cases we want to test,
//such as:
// - empty matrices
// - vectors
// - matrix and vector
// - vector and matrix
// - scalars
// - identity matrices
// - null matrices
// - nan matrices
// - overflow

/*
TEST(ExtremeMatrixMultiplicationTest, RectangularandSquaredMatrices) {
    std::vector<std::vector<int>> A = {
        {1, 2, 3},
        {4, 5, 6}
    };
    std::vector<std::vector<int>> B = {
        {7, 8},
        {9, 10},
        {11, 12}
    };
    std::vector<std::vector<int>> C(2, std::vector<int>(2, 0));

    multiplyMatrices(A, B, C, 2, 3, 2);

    std::vector<std::vector<int>> expected = {
        {58, 64},
        {139, 154}
    };

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}*/


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
