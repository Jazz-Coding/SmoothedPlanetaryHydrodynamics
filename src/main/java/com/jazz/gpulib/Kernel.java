package com.jazz.gpulib;

import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

import static jcuda.driver.JCudaDriver.*;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;

/**
 * Wrapper class for custom CUDA kernels.
 */
public class Kernel {
    private CUmodule module;
    private CUfunction function;

    public Kernel(String ptxFilePath, String mainFunction) {
        this.module = new CUmodule();
        this.function = new CUfunction();

        cuModuleLoad(this.module, ptxFilePath);
        cuModuleGetFunction(this.function, this.module, mainFunction);
    }

    public void call(Pointer parameters,
                     int gridSize,
                     int blockSize){
        cuLaunchKernel(function,
                gridSize,  1, 1,      // Grid dimension, processors are divide into a "grid" of blocks.
                blockSize, 1, 1,      // Block dimension, each block contains this many threads.
                0, null,               // Shared memory size and stream, must be set if shared memory used.
                parameters, null // Kernel- and extra parameters, passed to the .ptx file.
        );
        cuCtxSynchronize(); // Blocks to completion.
    }

    public CUmodule getModule() {
        return module;
    }

    public CUfunction getFunction() {
        return function;
    }

    public void destroy(){
        cuModuleUnload(this.module);
    }
}
