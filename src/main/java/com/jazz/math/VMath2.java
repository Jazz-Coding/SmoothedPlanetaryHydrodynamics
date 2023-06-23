package com.jazz.math;

import java.util.concurrent.ThreadLocalRandom;

/**
 * CPU bound vector math operations.
 * Many of these are used to initialize the simulation on the CPU, especially "randomNSphere".
 */
public class VMath2 {
    public static final float SQRT_OF_PI = (float) Math.sqrt(Math.PI);

    public static float[] diff(float[] vecA, float[] vecB){
        float[] diff = new float[vecA.length];
        for (int i = 0; i < vecA.length; i++) {
            diff[i] = vecA[i] - vecB[i];
        }
        return diff;
    }

    public static float[] interpolate(float[] vecA, float[] vecB){
       return scalDown(vadd(vecA, vecB), 2);
    }

    // One way to normalize a vector, calculates the euclidean distance from the origin.
    public static float euclideanNorm(float[] vector){
        float sum = 0;
        for (float v : vector) {
            sum += Math.pow(v, 2);
        }
        return (float) Math.sqrt(sum);
    }

    // Computationally more efficient than ||x[]||^2, a component of the gaussian kernel.
    public static float sumSquares(float[] vector){
        float sum = 0;
        for (float v : vector) {
            sum += Math.pow(v, 2);
        }
        return sum;
    }

    public static float[] vecSquared(float[] vec){
        float[] squared = new float[vec.length];
        for (int i = 0; i < vec.length; i++) {
            squared[i] = (float) Math.pow(vec[i],2);
        }
        return squared;
    }

    public static void squareVecIP(float[] vec){
        for (int i = 0; i < vec.length; i++) {
            vec[i] = (float) Math.pow(vec[i], 2);
        }
    }

    public static float[] randGaussVec(int size){
        float[] vec = new float[size];
        for (int i = 0; i < size; i++) {
            vec[i] = (float) ThreadLocalRandom.current().nextFloat(-1,1);
        }
        return vec;
    }

    public static float invNorm(float[] vec){
        return 1 / euclideanNorm(vec);
    }

    // Scales a vector down by its euclidean norm.
    public static void normalizeIP(float[] vec){
        scalIP(vec,invNorm(vec));
    }

    // In-place vector operations.
    public static void scalIP(float[] vector, float scalar){
        for (int i = 0; i < vector.length; i++) {
            vector[i] *= scalar;
        }
    }

    public static float[] scalDown(float[] vector, float divisor){
        float[] scaled = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            scaled[i] =  vector[i] / divisor;
        }
        return scaled;
    }

    public static void scalDownIP(float[] vector, float divisor){
        for (int i = 0; i < vector.length; i++) {
            vector[i] /= divisor;
        }
    }
    public static void vaddIP(float[] vecA, float[] vecB){
        for (int i = 0; i < vecA.length; i++) {
            vecA[i] += vecB[i];
        }
    }
    public static void vsubIP(float[] vecA, float[] vecB){
        for (int i = 0; i < vecA.length; i++) {
            vecA[i] -= vecB[i];
        }
    }

    // Out-of-place vector operations.
    public static float[] scal(float[] vector, float scalar){
        float[] newVec = new float[vector.length];
        for (int i = 0; i < vector.length; i++) {
            newVec[i] = vector[i] * scalar;
        }
        return newVec;
    }
    public static float[] vadd(float[] vecA, float[] vecB){
        float[] newVec = new float[vecA.length];
        for (int i = 0; i < vecA.length; i++) {
            newVec[i] = vecA[i] + vecB[i];
        }
        return newVec;
    }
    public static float[] vsub(float[] vecA, float[] vecB){
        float[] newVec = new float[vecA.length];
        for (int i = 0; i < vecA.length; i++) {
            newVec[i] = vecA[i] - vecB[i];
        }
        return newVec;
    }

    public static float[] randomNSphere(float radius, int dimensions){
        switch (dimensions){
            case 2 -> {
                float r = (float) (radius * Math.sqrt(ThreadLocalRandom.current().nextFloat()));
                float theta = (float) (2 * Math.PI * ThreadLocalRandom.current().nextFloat());
                return new float[]{(float) (r * Math.cos(theta)),
                                   (float) (r * Math.sin(theta))};
            }
            case 3 -> {
                // https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability/87238#87238
                float distanceFromOrigin = (float) (Math.cbrt(ThreadLocalRandom.current().nextGaussian()) * radius);
                float[] surfaceCoordinates = randGaussVec(dimensions);

                normalizeIP(surfaceCoordinates);
                scalIP(surfaceCoordinates,distanceFromOrigin);

                return surfaceCoordinates;
            }
            default -> {
                return new float[]{0f};
            }
        }
    }
}