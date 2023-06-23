package com.jazz.gpulib;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUmodule;
import jcuda.driver.CUresult;

import static com.jazz.gpulib.VectorMath.*;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.jcublas.JCublas.*;

/**
 * Primary interface between the CPU and GPU.
 * Handles loading data onto and off of the GPU.
 */
public class GPULoader {
    /**
     * Loads vector into VRAM.
     */
    public static Pointer loadVector(float[] vector){
        Pointer devicePointer = new Pointer();
        cublasAlloc(vector.length, Sizeof.FLOAT, devicePointer);
        cublasSetVector(vector.length, Sizeof.FLOAT, Pointer.to(vector), 1, devicePointer, 1);
        return devicePointer;
    }

    public static Pointer loadVectorFP16(float[] vector){
        Pointer devicePointer = new Pointer();
        cublasAlloc(vector.length, Sizeof.FLOAT/2, devicePointer);
        cublasSetVector(vector.length, Sizeof.FLOAT, Pointer.to(vector), 1, devicePointer, 1);
        return devicePointer;
    }

    public static Pointer loadVector(int[] vector){
        Pointer devicePointer = new Pointer();
        cublasAlloc(vector.length, Sizeof.INT, devicePointer);
        cublasSetVector(vector.length, Sizeof.INT, Pointer.to(vector), 1, devicePointer, 1);
        return devicePointer;
    }

    /**
     * Loads matrix into VRAM.
     * Wraps in a "flatten" call and uses the vector loading.
     * We could use cublasSetMatrix but this simplifies things to just one type of call.
     */
    public static Pointer loadMatrix(float[][] matrix){
        return loadVector(flatten(matrix));
    }

    /**
     * Flattens into column-major format.
     */
    public static Pointer loadMatrixColumnMajor(float[][] matrix){
        return loadVector(flattenColumnMajor(matrix));
    }

    public static Pointer loadMatrixColumnMajorFP16(float[][] matrix){
        return loadVectorFP16(flattenColumnMajor(matrix));
    }

    /**
     * Loads vector out of VRAM.
     */
    public static void unloadVector(Pointer devicePointer, Pointer extractionPoint, int length){
        cublasGetVector(length, Sizeof.FLOAT, devicePointer, 1, extractionPoint, 1);
        cublasFree(devicePointer);
    }

    public static void unloadVectorFP16(Pointer devicePointer, Pointer extractionPoint, int length){
        cublasGetVector(length, Sizeof.FLOAT/2, devicePointer, 1, extractionPoint, 1);
        cublasFree(devicePointer);
    }

    /**
     * Copies vector out of VRAM.
     */
    public static void peekVector(Pointer devicePointer, Pointer extractionPoint, int length){
        cublasGetVector(length, Sizeof.FLOAT, devicePointer, 1, extractionPoint, 1);
    }



    /**
     * Copies whole matrix out of VRAM.
     */
    public static float[][] peekMatrix(Pointer source, int rows, int cols){
        int length1d = rows*cols;
        float[] vector = new float[length1d];
        peekVector(source,Pointer.to(vector),length1d);
        return inflate(vector,rows,cols);
    }

    /**
     * Loads matrix out of VRAM.
     */
    public static float[][] unloadMatrix(Pointer source, int rows, int cols){
        int length1d = rows*cols;
        float[] vector = new float[length1d];
        unloadVector(source,Pointer.to(vector),length1d);
        return inflate(vector,rows,cols);
    }

    /**
     * Copies whole matrix out of VRAM using a matrix pointer.
     */
    public static float[][] peekMatrix(MatrixPointer matrixPointer){
        return peekMatrix(matrixPointer.getDevicePointer(), matrixPointer.getRows(), matrixPointer.getCols());
    }

    public static float[] peekVector(MatrixPointer matrixPointer){
        float[] extract = new float[matrixPointer.getLength1d()];
        peekVector(matrixPointer.getDevicePointer(),Pointer.to(extract),matrixPointer.getLength1d());
        return extract;
    }

    /**
     * Same for full unloading.
     */
    public static float[][] unloadMatrix(MatrixPointer matrixPointer){
        return unloadMatrix(matrixPointer.getDevicePointer(),matrixPointer.getRows(),matrixPointer.getCols());
    }

    /**
     * Copies whole matrix out of VRAM.
     */
    public static float[][] peekMatrixColumnMajor(Pointer source, int rows, int cols){
        int length1d = rows*cols;
        float[] vector = new float[length1d];
        peekVector(source,Pointer.to(vector),length1d);
        return inflateColumnMajor(vector,rows,cols);
    }

    /**
     * Loads matrix out of VRAM.
     */
    public static float[][] unloadMatrixColumnMajor(Pointer source, int rows, int cols){
        int length1d = rows*cols;
        float[] vector = new float[length1d];
        unloadVector(source,Pointer.to(vector),length1d);
        return inflateColumnMajor(vector,rows,cols);
    }

    public static float[][] unloadMatrixColumnMajorFP16(Pointer source, int rows, int cols){
        int length1d = rows*cols;
        float[] vector = new float[length1d];
        unloadVectorFP16(source,Pointer.to(vector),length1d);
        return inflateColumnMajor(vector,rows,cols);
    }

    /**
     * Copies whole matrix out of VRAM using a matrix pointer.
     */
    public static float[][] peekMatrixColumnMajor(MatrixPointer matrixPointer){
        return peekMatrixColumnMajor(matrixPointer.getDevicePointer(), matrixPointer.getRows(), matrixPointer.getCols());
    }

    /**
     * Same for full unloading.
     */
    public static float[][] unloadMatrixColumnMajor(MatrixPointer matrixPointer){
        return unloadMatrixColumnMajor(matrixPointer.getDevicePointer(),matrixPointer.getRows(),matrixPointer.getCols());
    }

    public static float[][] unloadMatrixColumnMajorFP16(MatrixPointer matrixPointer){
        return unloadMatrixColumnMajorFP16(matrixPointer.getDevicePointer(),matrixPointer.getRows(),matrixPointer.getCols());
    }

    public static void setGPUConstantInt(String name, int value, CUmodule module){
        // Obtain the pointer to the constant memory, and print some info
        CUdeviceptr constantMemoryPointer = new CUdeviceptr();
        long[] constantMemorySizeArray = {0};
        cuModuleGetGlobal(constantMemoryPointer, constantMemorySizeArray, module, name);
        int constantMemorySize = (int)constantMemorySizeArray[0];

        // Copy value over.
        int result = cuMemcpyHtoD(constantMemoryPointer, Pointer.to(new int[]{value}), constantMemorySize);
        if (result != CUresult.CUDA_SUCCESS) {
            System.err.println("Error copying to device constant '" + name + "': " + CUresult.stringFor(result));
            throw new RuntimeException("err");
        }
    }

    public static void setGPUConstantFloat(String name, float value, CUmodule module){
        // Obtain the pointer to the constant memory, and print some info
        CUdeviceptr constantMemoryPointer = new CUdeviceptr();
        long[] constantMemorySizeArray = {0};
        cuModuleGetGlobal(constantMemoryPointer, constantMemorySizeArray, module, name);
        int constantMemorySize = (int)constantMemorySizeArray[0];

        // Copy value over.
        int result = cuMemcpyHtoD(constantMemoryPointer, Pointer.to(new float[]{value}), constantMemorySize);
        // Check for errors
        if (result != CUresult.CUDA_SUCCESS) {
            System.err.println("Error copying to device constant '" + name + "': " + CUresult.stringFor(result));
            throw new RuntimeException("err");
        }
    }

    public static void setGPUConstantDouble(String name, float value, CUmodule module){
        // Obtain the pointer to the constant memory, and print some info
        CUdeviceptr constantMemoryPointer = new CUdeviceptr();
        long[] constantMemorySizeArray = {0};
        cuModuleGetGlobal(constantMemoryPointer, constantMemorySizeArray, module, name);
        int constantMemorySize = (int)constantMemorySizeArray[0];

        // Copy value over.
        cuMemcpyHtoD(constantMemoryPointer,Pointer.to(new double[]{value}),constantMemorySize);
    }

    public static void free(Pointer devicePointer){
        cublasFree(devicePointer);
    }

}
