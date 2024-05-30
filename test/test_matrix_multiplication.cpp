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
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_dim(1, 10);  // dimensions interval 
    std::uniform_int_distribution<> dis_val(-2, 2); // values interval

    // Randomly generate the dimensions of A
    int rowsA = 1;
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
    int colsB = 1;

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

TEST(CriticalMatrixMultiplicationTest, VectorMatrix) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_dim(1, 10);  // dimensions interval 
    std::uniform_int_distribution<> dis_val(-2, 2); // values interval

    // Randomly generate the dimensions of A
    int rowsA = 1;
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
    int colsB = dis_dim(gen);

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

TEST(CriticalMatrixMultiplicationTest, MatrixVector) {
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
    int colsB = 1;

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

TEST(CriticalMatrixMultiplicationTest, Scalars) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_val(-2, 2); // values interval

    // Randomly generate the dimensions of A
    int rowsA = 1;
    int colsA = 1;

    // Create the A matrix with random dim
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill the matrix with casual values
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
            A[i][j] = dis_val(gen);
        }
    }

    // Set proper dim for B matrix
    int rowsB = 1;
    int colsB = 1;

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

TEST(CriticalMatrixMultiplicationTest, ScalarVector) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_dim(1, 10);  // dimensions interval 
    std::uniform_int_distribution<> dis_val(-2, 2); // values interval

    // Randomly generate the dimensions of A
    int rowsA = 1;
    int colsA = 1;

    // Create the A matrix with random dim
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill the matrix with casual values
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
            A[i][j] = dis_val(gen);
        }
    }

    // Set proper dim for B matrix
    int rowsB = 1;
    int colsB = dis_dim(gen);

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

TEST(CriticalMatrixMultiplicationTest, OddDimMatrices) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_dim(1, 10); // dimesions interval
    std::uniform_int_distribution<> dis_val(-2, 2); // values interval

    // Randomly generate the dimensions of A
    int rowsA = 2 * dis_dim(gen) + 1;
    int colsA = 2 * dis_dim(gen) + 1;

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
    int colsB = 2 * dis_dim(gen) + 1;

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

TEST(CriticalMatrixMultiplicationTest, EvenDimMatrices) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_dim(1, 10); // dimesions interval
    std::uniform_int_distribution<> dis_val(-2, 2); // values interval

    // Randomly generate the dimensions of A
    int rowsA = 2 * dis_dim(gen);
    int colsA = 2 * dis_dim(gen);

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
    int colsB = 2 * dis_dim(gen);

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

TEST(CriticalMatrixMultiplicationTest, OverflowMatrices) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_dim(1, 10); // dimesions interval
    std::uniform_int_distribution<> dis_val(10, 15); // values interval

    // Randomly generate the dimensions of A
    int rowsA = dis_dim(gen);
    int colsA = dis_dim(gen);

    // Create A matrix with random dim
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA));

    // Fill the matrix with random values
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsA; ++j) {
            A[i][j] = 2e32;
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

TEST(CriticalMatrixMultiplicationTest, IdentityMatrix) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_dim(1, 10); // dimesions interval
    std::uniform_int_distribution<> dis_val(-2, 2); // values interval

    // Randomly generate the dimensions of A
    int rowsA = dis_dim(gen);
    int colsA = rowsA;

    // Create A matrix with random dim
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(rowsA, 0));

    // Fill the matrix with random values
    for (int i = 0; i < rowsA; ++i) {
        A[i][i] = 1;
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

TEST(CriticalMatrixMultiplicationTest, SquareZerosMatrices) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_dim(1, 10); // dimensions interval

    // Randomly generate dimensions of A
    int rowsA = dis_dim(gen);
    int colsA = rowsA;

    // Create A matrix with random dim
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA, 0));

    // Set proper dim for B
    int rowsB = colsA;
    int colsB = colsA;

    // Create B matrix with random dim
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB, 0));

    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(CriticalMatrixMultiplicationTest, SquareOnesMatrices) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_dim(1, 10); // dimensions interval

    // Randomly generate dimensions of A
    int rowsA = dis_dim(gen);
    int colsA = rowsA;

    // Create A matrix with random dim
    std::vector<std::vector<int>> A(rowsA, std::vector<int>(colsA, 1));

    // Set proper dim for B
    int rowsB = colsA;
    int colsB = colsA;

    // Create B matrix with random dim
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB, 1));

    std::vector<std::vector<int>> C(rowsA, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, rowsA, colsA, colsB);

    std::vector<std::vector<int>> expected(rowsA, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,rowsA,colsA,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}

TEST(CriticalMatrixMultiplicationTest, SmallValuesRectangularMatrices) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_dim(1, 10); // dimesions interval
    std::uniform_int_distribution<> dis_val(0, 10); // values interval

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

TEST(CriticalMatrixMultiplicationTest, IntermediateValuesRectangularMatrices) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_dim(1, 10); // dimesions interval
    std::uniform_int_distribution<> dis_val(11, 20); // values interval

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

TEST(CriticalMatrixMultiplicationTest, BigValuesRectangularMatrices) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_dim(1, 10); // dimesions interval
    std::uniform_int_distribution<> dis_val(75, 125); // values interval

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

TEST(CriticalMatrixMultiplicationTest, NegativeValuesRectangularMatrices) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_dim(1, 10); // dimesions interval
    std::uniform_int_distribution<> dis_val(-10, 0); // values interval

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

TEST(CriticalMatrixMultiplicationTest, NULLMatrices) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_dim(1, 10); // dimesions interval
    std::uniform_int_distribution<> dis_val(-2, 2); // values interval

    // Create A matrix with random dim
    std::vector<std::vector<int>> A;

    // Set proper dim for B matrix
    int rowsB = dis_dim(gen);
    int colsB = dis_dim(gen);

    // Create B matrix with random dim
    std::vector<std::vector<int>> B(rowsB, std::vector<int>(colsB));

    // fill the matrix with random values
    for (int i = 0; i < rowsB; ++i) {
        for (int j = 0; j < colsB; ++j) {
            B[i][j] = dis_val(gen);
        }
    }

    std::vector<std::vector<int>> C(0, std::vector<int>(colsB, 0));

    multiplyMatrices(A, B, C, 0, 0, colsB);

    std::vector<std::vector<int>> expected(0, std::vector<int>(colsB, 0));
    multiplyMatricesWithoutErrors(A,B,expected,0,0,colsB);

    ASSERT_EQ(C, expected) << "Matrix multiplication test failed! :(((()";
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
