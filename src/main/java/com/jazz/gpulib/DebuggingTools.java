package com.jazz.gpulib;

public class DebuggingTools {
    // Mean of the error squared. Error = difference between actual and expected outputs.
    // Useful for checking GPU computation agrees with the CPU implementation, highlights
    // discrepancies in row vs column major form well.
    public static float MSE(float[] outputs, float[] expectedOutputs){
        float sum = 0f;
        int n = outputs.length;

        for (int i = 0; i < n; i++) {
            float error = expectedOutputs[i]-outputs[i];
            sum += error*error;
        }

        return sum / n;
    }
}
