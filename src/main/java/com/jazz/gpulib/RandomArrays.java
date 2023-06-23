package com.jazz.gpulib;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Multiple forms of randomly initializing arrays.
 */
public class RandomArrays {
    public static void fillRandomly(float[] array){
        for (int i = 0; i < array.length; i++) {
            array[i] = (float) ThreadLocalRandom.current().nextGaussian();
        }
    }
    public static void fillSequentially(float[] array){
        for (int i = 0; i < array.length; i++) {
            array[i] = i;
        }
    }

    public static void fillRandomly(float[][] array2D){
        for (float[] row : array2D) {
            fillRandomly(row);
        }
    }

    public static void fillRandomly(float[] array, int seed){
        for (int i = 0; i < array.length; i++) {
            array[i] = (float) new Random(seed+(299*i)).nextGaussian();
        }
    }
    public static void fillRandomly(float[][] array2D, int seed){
        for (float[] row : array2D) {
            fillRandomly(row, seed);
        }
    }

    public static void fillOnes(float[][] array2D){
        for (int i = 0; i < array2D.length; i++) {
            for (int j = 0; j < array2D[0].length; j++) {
                array2D[i][j] = 1F;
            }
        }
    }
}
