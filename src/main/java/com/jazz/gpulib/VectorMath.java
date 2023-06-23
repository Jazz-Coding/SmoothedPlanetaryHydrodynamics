package com.jazz.gpulib;

import java.util.concurrent.ForkJoinPool;

/**
 * CPU vector math operations.
 * Mostly redundant, used for the debugging GUIs.
 */
public class VectorMath {
    public static float[] vectorSubtract(float[] vectorA, float[] vectorB){
        float[] acc = new float[vectorA.length];
        for (int i = 0; i < vectorA.length; i++) {
            acc[i] = vectorA[i] - vectorB[i];
        }
        return acc;
    }

    public static float[][] vectorSubtract(float[][] matA, float[][] matB){
        int rows = matA.length;
        int cols = matA[0].length;

        float[][] matC = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matC[i][j] = matA[i][j] - matB[i][j];
            }
        }
        return matC;
    }

    public static float[][] scale(float[][] mat, float scalar){
        int rows = mat.length;
        int cols = mat[0].length;

        float[][] matC = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matC[i][j] = mat[i][j] * scalar;
            }
        }
        return matC;
    }

    public static float[] scale(float[] vector, float scalar){
        float[] scaled = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            scaled[i] = vector[i] * scalar;
        }
        return scaled;
    }

    /**
     * Collapses the columns of a matrix by summing, then divides by their
     * number, performing the mean average of each row.
     */
    public static float[] averageColumns(float[][] matrix){
        int rows = matrix.length;
        float[] average = new float[rows];
        for (int i = 0; i < rows; i++) {
            float[] row = matrix[i];
            float sum = 0f;

            for (float v : row) {
                sum += v;
            }
            sum /= row.length;
            average[i] = sum;
        }
        return average;
    }

    public static void averageColumnsAndScaleIP(float[][] matrix, float scalar, float[] acc){
        int rows = matrix.length;
        for (int i = 0; i < rows; i++) {
            float[] row = matrix[i];
            float sum = 0f;

            for (float v : row) {
                sum += v;
            }
            sum /= row.length;
            acc[i] = scalar * sum;
        }
    }

    public static int argMax(float[] vector){
        float max = Float.NEGATIVE_INFINITY;
        int maxIndex = 0;

        for (int i = 0; i < vector.length; i++) {
            float contender = vector[i];
            if(contender >= max){
                max = contender;
                maxIndex = i;
            }
        }
        return maxIndex;
    }

    public static void vectorSubtractIP(float[] vectorA, float[] vectorB, float[] acc){
        for (int i = 0; i < vectorA.length; i++) {
            acc[i] = vectorA[i] - vectorB[i];
        }
    }
    public static float[][] transpose(float[] vector){
        int cols = vector.length;

        float[][] transposed = new float[cols][1];
        for (int i = 0; i < cols; i++) {
            transposed[i][0] = vector[i];
        }

        return transposed;
    }

    public static float[][] transpose(float[][] matrix){
        int rows = matrix.length;
        int cols = matrix[0].length;

        float[][] transposed = new float[cols][rows];
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < rows; j++) {
                transposed[i][j] = matrix[j][i];
            }
        }

        return transposed;
    }

    public static float norm2(float[] vector){
        float sum = 0F;
        for (float v : vector) {
            sum += v * v;
        }
        return (float) Math.sqrt(sum);
    }

    public static float norm1(float[] vector){
        float sum = 0F;
        for (float v : vector) {
            sum += v * v;
        }
        return sum;
    }

    private static ForkJoinPool pool = new ForkJoinPool();

    public static float[][] multiplySlowly(float[][] matA, float[][] matB){
        int rowsA = matA.length;
        int rowsB = matB.length;
        int colsB = matB[0].length;

        float[][] matC = new float[rowsA][colsB];
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                float sum = 0f;
                for (int k = 0; k < rowsB; k++) {
                    sum += matA[i][k] * matB[k][j];
                }
                matC[i][j] = sum;
            }
        }

        return matC;
    }
    public static float[][] multiply(float[][] matA, float[][] matB){
        //return multiplySlowly(matA,matB);
        int tileSize = 32;

        // n = rowsA
        // m = colsA and rowsB
        // p = colsB
        int rowsA = matA.length;
        int rowsB = matB.length;
        int colsB = matB[0].length;

        float[][] acc = new float[rowsA][colsB];

        for (int I = 0; I < rowsA; I+=tileSize) {
            int minI = Math.min(I + tileSize, rowsA);
            for (int J = 0; J < colsB; J+=tileSize) {
                int minJ = Math.min(J + tileSize, colsB);
                for (int K = 0; K < rowsB; K+=tileSize) {
                    int minK = Math.min(K + tileSize, rowsB);

                    // Process the tile.
                    for (int i = I; i < minI; i++) {
                        for (int j = J; j < minJ; j++) {
                            float sum = 0f;
                            for (int k = K; k < minK; k++) {
                                sum += matA[i][k] * matB[k][j];
                            }
                            acc[i][j] = sum;
                        }
                    }
                }
            }
        }

        return acc;
    }

    public static void multiplyIP(float[][] matA, float[][] matB, float[][] acc){
        int rowsA = matA.length;
        int rowsB = matB.length;
        int colsB = matB[0].length;

        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                float sum = 0f;
                for (int k = 0; k < rowsB; k++) {
                    sum += matA[i][k] * matB[k][j];
                }
                acc[i][j] = sum;
            }
        }
    }


    public static void mAT(float[][] matA, float scalarA,
                           float[][] matB, float scalarB,
                           float[][] acc){
        float scalar = scalarA*scalarB;

        int tileSize = 32;

        // n = rowsA
        // m = colsA and rowsB
        // p = colsB
        int rowsA = matA.length;
        int colsA = matA[0].length;
        int rowsB = matB.length;
        int colsB = matB[0].length;

        for (int I = 0; I < colsA; I += tileSize) {
            int minI = Math.min(I + tileSize, colsA);

            for (int J = 0; J < colsB; J += tileSize) {
                int minJ = Math.min(J + tileSize, colsB);
                for (int K = 0; K < rowsB; K += tileSize) {
                    int minK = Math.min(K + tileSize, rowsB);

                    // Process the tile.
                    for (int i = I; i < minI; i++) {
                        for (int j = J; j < minJ; j++) {
                            float sum = 0f;
                            for (int k = K; k < minK; k++) {
                                sum += matA[k][i] * matB[k][j];
                            }
                            acc[i][j] += scalar * sum;
                        }
                    }
                }
            }
        }
    }
    public static void mBT(float[][] matA, float scalarA,
                           float[][] matB, float scalarB,
                           float[][] acc){
        int tileSize = 32;

        // n = rowsA
        // m = colsA and rowsB
        // p = colsB
        int rowsA = matA.length;
        int rowsB = matB.length;
        int colsB = matB[0].length;

        float superScalar = scalarA*scalarB;

        for (int I = 0; I < rowsA; I+=tileSize) {
            int minI = Math.min(I + tileSize, rowsA);
            for (int J = 0; J < rowsB; J+=tileSize) {
                int minJ = Math.min(J + tileSize, rowsB);

                for (int K = 0; K < colsB; K+=tileSize) {
                    int minK = Math.min(K + tileSize, colsB);

                    // Process the tile.
                    for (int i = I; i < minI; i++) {
                        for (int j = J; j < minJ; j++) {
                            float sum = 0f;
                            for (int k = K; k < minK; k++) {
                                sum += matA[i][k] * matB[j][k];
                            }
                            acc[i][j] += superScalar * sum;
                        }
                    }
                }
            }
        }
    }
    /**
     * Matrix multiplication with optional transposition and scaling.
     */
    public static void multiplySmart(float[][] matA, float scalarA, boolean transposeA,
                                     float[][] matB, float scalarB, boolean transposeB,
                                     float[][] acc){
        if(transposeA){
            mAT(matA,scalarA,matB,scalarB,acc);
        } else if (transposeB){
            mBT(matA,scalarA,matB,scalarB,acc);
        } else {
            float scalar = scalarA*scalarB;

            int tileSize = 32;

            // n = rowsA
            // m = colsA and rowsB
            // p = colsB
            int rowsA = matA.length;
            int rowsB = matB.length;
            int colsB = matB[0].length;

            for (int I = 0; I < rowsA; I += tileSize) {
                int minI = Math.min(I + tileSize, rowsA);
                for (int J = 0; J < colsB; J += tileSize) {
                    int minJ = Math.min(J + tileSize, colsB);
                    for (int K = 0; K < rowsB; K += tileSize) {
                        int minK = Math.min(K + tileSize, rowsB);

                        // Process the tile.
                        for (int i = I; i < minI; i++) {
                            for (int j = J; j < minJ; j++) {
                                float sum = 0f;
                                for (int k = K; k < minK; k++) {
                                    sum += matA[i][k] * matB[k][j];
                                }
                                acc[i][j] += scalar * sum;
                            }
                        }
                    }
                }
            }
        }
    }
    public static void multiplyIPAndScale(float[][] matA, float[][] matB, float scalar, boolean transposeA, boolean transposeB, float[][] acc){
        int tileSize = 32;

        // n = rowsA
        // m = colsA and rowsB
        // p = colsB
        int rowsA = matA.length;
        int rowsB = matB.length;
        int colsB = matB[0].length;

        for (int I = 0; I < rowsA; I+=tileSize) {
            int minI = Math.min(I + tileSize, rowsA);
            for (int J = 0; J < colsB; J+=tileSize) {
                int minJ = Math.min(J + tileSize, colsB);
                for (int K = 0; K < rowsB; K+=tileSize) {
                    int minK = Math.min(K + tileSize, rowsB);

                    // Process the tile.
                    for (int i = I; i < minI; i++) {
                        for (int j = J; j < minJ; j++) {
                            float sum = 0f;
                            for (int k = K; k < minK; k++) {
                                sum += matA[i][k] * matB[k][j];
                            }
                            acc[i][j] += scalar * sum;
                        }
                    }
                }
            }
        }
    }

    public static float[][] hadamardMultiply(float[][] matA, float[][] matB){
        int rows = matA.length;
        int cols = matA[0].length;

        float[][] matC = new float[rows][cols];

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matC[i][j] = matA[i][j] * matB[i][j];
            }
        }
        return matC;
    }

    public static void hadamardMultiplyIP(float[][] matA, float[][] matB, float[][] acc){
        int rows = matA.length;
        int cols = matA[0].length;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                acc[i][j] = matA[i][j] * matB[i][j];
            }
        }
    }

    // Casts a vector into a matrix. Useful for applying the same bias
    // to a batch of different inputs.
    public static float[][] vectorBroadcast(float[] vector, int copies){
        int n = vector.length;
        float[][] mat = new float[n][copies];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < copies; j++) {
                mat[i][j] = vector[i];
            }
        }

        return mat;
    }

    public static float[][] add(float[][] matA, float[][] matB){
        int rowsA = matA.length;
        int colsA = matA[0].length;

        float[][] matC = new float[rowsA][colsA];
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsA; j++) {
                matC[i][j] = matA[i][j] + matB[i][j];
            }
        }
        return matC;
    }

    public static void addIP(float[][] matA, float[][] matB){
        int rowsA = matA.length;
        int colsA = matA[0].length;
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsA; j++) {
                matA[i][j] = matA[i][j] + matB[i][j];
            }
        }
    }

    /**
     * Automatically broadcasts "vector" to match the dimensions of matA.
     */
    public static void addIPWithBroadcast(float[][] matA, float[] vector){
        int rowsA = matA.length;
        int colsA = matA[0].length;
        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsA; j++) {
                matA[i][j] += vector[i];
            }
        }
    }

    public static void addIP(float[] vecA, float[] vecB){
        for (int i = 0; i < vecA.length; i++) {
            vecA[i] += vecB[i];
        }
    }

    public static float[] add(float[] vecA, float[] vecB){
        float[] sum = new float[vecA.length];
        for (int i = 0; i < vecA.length; i++) {
            sum[i] = vecA[i] + vecB[i];
        }

        return sum;
    }

    public static float max(float[] vec){
        float biggest = Float.NEGATIVE_INFINITY;
        for (float contender : vec) {
            biggest = Math.max(contender,biggest);
        }
        return biggest;
    }

    public static float maxAbs(float[] vec){
        float biggest = Float.NEGATIVE_INFINITY;
        for (float contender : vec) {
            biggest = Math.max(Math.abs(contender),biggest);
        }
        return biggest;
    }

    public static float maxAbs(float[][] mat){
        float biggestSquared = Float.NEGATIVE_INFINITY;
        for (float[] contender : mat) {
            // To save time, work out the biggest by norm1, which maintains the size order, and square root only at the very eend.
            biggestSquared = Math.max(norm1(contender),biggestSquared);
        }
        return (float) Math.sqrt(biggestSquared);
    }

    public static float[] flatten(float[][] matrix){
        int rows = matrix.length;
        int cols = matrix[0].length;

        float[] vector = new float[rows*cols];

        int superindex = 0;
        for (float[] row : matrix) {
            for (float value : row) {
                vector[superindex] = value;
                superindex++;
            }
        }
        return vector;
    }

    public static float[] flattenColumnMajor(float[][] matrix){
        int rows = matrix.length;
        int cols = matrix[0].length;

        float[] vector = new float[rows*cols];

        int superindex = 0;
        for (int i = 0; i < cols; i++) {
            for (int j = 0; j < rows; j++) {
                vector[superindex] = matrix[j][i];
                superindex++;
            }
        }

        return vector;
    }

    public static float[][] inflateColumnMajor(float[] vector, int rows, int cols){
        float[][] matrix = new float[rows][cols];

        int superindex = 0;
        for (int j = 0; j < cols; j++) {
            for (int i = 0; i < rows; i++) {
                matrix[i][j] = vector[superindex];
                superindex++;
            }
        }

        return matrix;
    }

    public static float[][] inflate(float[] vector, int rows, int cols){
        float[][] matrix = new float[rows][cols];

        int superindex = 0;
        for (int rowIndex = 0; rowIndex < rows; rowIndex++) {
            // Create row.
            float[] row = new float[cols];
            for (int colIndex = 0; colIndex < cols; colIndex++) {
                row[colIndex] = vector[superindex];
                superindex++;
            }
            matrix[rowIndex] = row;
        }
        return matrix;
    }
}
