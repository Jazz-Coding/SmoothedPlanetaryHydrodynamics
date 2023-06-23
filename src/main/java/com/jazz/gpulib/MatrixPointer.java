package com.jazz.gpulib;

import jcuda.Pointer;

import static com.jazz.gpulib.PrettyPrint.matrixPrint;
import static com.jazz.gpulib.RandomArrays.fillRandomly;


/**
 * Helper class that remembers how big the matrices pointed to actually are.
 * Also allows for convenient loading/unloading from VRAM and ensures
 * memory is freed.
 *
 * The immense utility of this class is encapsulated by the following, human-centred description:
 * "Here's my matrix, just give me the pointer, I don't care, no I don't know
 *  how many rows it has, work it out, that's your job."
 */
public class MatrixPointer {
    private Pointer devicePointer;

    private int rows;
    private int cols;
    private int length1d; // Frequently used, saves recalculating each time.
    private boolean rowMajor = false;

    public MatrixPointer(float[][] matrix) {
        this.rows = matrix.length;
        this.cols = matrix[0].length;
        this.length1d = this.rows*this.cols;
        this.devicePointer = GPULoader.loadMatrixColumnMajor(matrix);
    }

    /**
     * Row major-form is laid out in VRAM differently.
     * By default, column-major form is assumed, to ensure interoperability with cuBLAS.
     */
    public MatrixPointer(float[][] matrix, boolean rowMajor) {
        this.rows = matrix.length;
        this.cols = matrix[0].length;
        this.length1d = this.rows*this.cols;
        this.devicePointer = GPULoader.loadMatrix(matrix);
        this.rowMajor = rowMajor;
    }

    /**
     * Single vectors need to be treated differently.
     */
    public MatrixPointer(float[] vector){
        this.rows = 1;
        this.cols = vector.length;
        this.length1d = cols;
        this.devicePointer = GPULoader.loadVector(vector);
    }

    public MatrixPointer(int[] vector){
        this.rows = 1;
        this.cols = vector.length;
        this.length1d = cols;
        this.devicePointer = GPULoader.loadVector(vector);
    }

    /**
     * Some commonly used default matrices.
     */
    public static MatrixPointer randomMatrix(int rows, int cols){
        float[][] matrix = new float[rows][cols];
        fillRandomly(matrix);

        return new MatrixPointer(matrix);
    }
    public static MatrixPointer zeroMatrix(int rows, int cols){
        float[][] matrix = new float[rows][cols];
        return new MatrixPointer(matrix);
    }
    public static MatrixPointer zeroMatrix(int rows, int cols, boolean rowMajor){
        float[][] matrix = new float[rows][cols];
        return new MatrixPointer(matrix,rowMajor);
    }


    public static MatrixPointer oneMatrix(int rows, int cols){
        float[][] matrix = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = 1F;
            }
        }
        return new MatrixPointer(matrix);
    }

    public Pointer getDevicePointer() {
        return devicePointer;
    }

    public MatrixPointer[] splitColumns(){
        MatrixPointer[] columns = new MatrixPointer[cols];

        float[][] transposed = new float[cols][rows];
        float[][] mat = peek();

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposed[j][i] = mat[i][j];
            }
        }

        for (int i = 0; i < cols; i++) {
            MatrixPointer mat2 = new MatrixPointer(transposed[i]);
            columns[i] = mat2;
        }

        return columns;
    }

    public int getRows() {
        return rows;
    }

    /**
     * Aliases.
     */
    public int getCols() {
        return cols;
    }
    public int getLD(){
        return rows;
    }
    public int getLength1d() {
        return length1d;
    }

    /**
     * Copies contents off GPU.
     */
    public float[][] peek(){
        if(rowMajor){
            return GPULoader.peekMatrix(this);
        } else {
            return GPULoader.peekMatrixColumnMajor(this);
        }
    }

    public float[] peekVector(){
        return GPULoader.peekVector(this);
    }

    /**
     * Loads off GPU (frees memory afterwards).
     */
    public float[][] unload(){
        if(rowMajor){
            return GPULoader.unloadMatrix(this);
        } else {
            return GPULoader.unloadMatrixColumnMajor(this);
        }
    }

    public void free(){
        GPULoader.free(this.devicePointer);
    }

    public float[][] unloadFP16(){
        if(rowMajor){
            return GPULoader.unloadMatrix(this);
        } else {
            return GPULoader.unloadMatrixColumnMajorFP16(this);
        }
    }

    @Override
    public String toString() {
        return matrixPrint(peek());
    }

    public void setDevicePointer(Pointer devicePointer) {
        this.devicePointer = devicePointer;
    }

    public static void swapPointers(MatrixPointer pointerA, MatrixPointer pointerB){
        Pointer _pointerA = pointerA.getDevicePointer();
        Pointer _pointerB = pointerB.getDevicePointer();

        pointerA.setDevicePointer(_pointerB);
        pointerB.setDevicePointer(_pointerA);
    }

    public void assign(MatrixPointer newValue){
        free(); // Free the existing data.
        this.devicePointer = newValue.getDevicePointer(); // Assumes the pointer already exists.
    }
}
