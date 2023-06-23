package com.jazz.gpulib;

import java.util.Arrays;

/**
 * Utilities for printing 2D and even 3D arrays neatly to the console.
 */
public class PrettyPrint {
    public static String matrixPrint(float[][] mat){
        StringBuilder sb = new StringBuilder();
        for (float[] row : mat) {
            sb.append(Arrays.toString(row)).append("\n");
        }
        return sb.toString();
    }

    public static String matrixPrint(float[][][] mat){
        StringBuilder sb = new StringBuilder();

        int aisles = mat.length;
        for (int i = 0; i < aisles; i++) {
            sb.append(matrixPrint(mat[i]));

            if(i != aisles-1) {
                sb.append("\n");
            }
        }
        return sb.toString();
    }
}
