package com.jazz.gpulib;

import jcuda.Pointer;
import jcuda.driver.*;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasHandle;

import static com.jazz.gpulib.GPULoader.*;
import static jcuda.cudaDataType.CUDA_R_32F;
import static jcuda.driver.JCudaDriver.*;
import static jcuda.jcublas.cublasGemmAlgo.CUBLAS_GEMM_DEFAULT;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_N;
import static jcuda.jcublas.cublasOperation.CUBLAS_OP_T;
import static jcuda.runtime.JCuda.cudaDeviceSynchronize;

public class MatrixMath {
    /**
     * Custom CUDA kernels.
     */
    private static CUfunction calculate_densities_F;
    private static CUfunction calculate_accelerations_F;
    private static CUfunction copy_to_gui_F;
    private static CUfunction calculate_momenta_F;
    private static CUfunction calculate_properties_F;

    private static CUmodule physicsModule;
    private static CUmodule calculate_accelerations_M;
    private static CUmodule calculate_densities_M;
    private static CUmodule copy_to_gui_M;
    private static CUmodule calculate_momenta_M;
    private static CUmodule calculate_properties_M;

    private static CUmodule constants;

    private static cublasHandle handle;

    /**
     * Initialize cuBLAS  and CUDA backend.
     */
    public static void initialize(){
        System.out.println("Initializing GPU backend...");
        // Enable exceptions and omit all subsequent error checks
        JCudaDriver.setExceptionsEnabled(true);
        // Initialize the driver and create a context for the first device.
        cuInit(0);
        CUdevice device = new CUdevice();
        cuDeviceGet(device, 0);
        CUcontext context = new CUcontext();
        cuCtxCreate(context, 0, device);

        JCublas2.initialize();
        handle = new cublasHandle();
        JCublas2.cublasCreate(handle);
        loadKernels();
    }

    /**
     * Load and initialize CUDA kernels.
     */
    private static void loadKernels(){
        physicsModule = new CUmodule();
        cuModuleLoad(physicsModule, "kernels/physics.ptx");

        calculate_densities_F = new CUfunction();
        cuModuleGetFunction(calculate_densities_F, physicsModule, "calculate_densities");

        calculate_momenta_F = new CUfunction();
        cuModuleGetFunction(calculate_momenta_F, physicsModule, "calculate_momentum");

        calculate_accelerations_F = new CUfunction();
        cuModuleGetFunction(calculate_accelerations_F, physicsModule, "calculate_accelerations");

        calculate_properties_F = new CUfunction();
        cuModuleGetFunction(calculate_properties_F, physicsModule, "calculate_properties");

        copy_to_gui_M = new CUmodule();
        cuModuleLoad(copy_to_gui_M, "kernels/copy_to_gui.ptx");
        copy_to_gui_F = new CUfunction();
        cuModuleGetFunction(copy_to_gui_F, copy_to_gui_M, "copy_to_gui");
    }

    public static void shutdown(){
        JCublas2.cublasDestroy(handle);
    }

    /**
     * Performs a particle density update on the GPU.
     */
    public static void updateDensitiesGPU(MatrixPointer densities,
                                          MatrixPointer positions, float splineConst,float smoothingLength){
        Pointer kernelParameters = Pointer.to(
                Pointer.to(densities.getDevicePointer()),
                Pointer.to(positions.getDevicePointer()),
                Pointer.to(new float[]{splineConst}),
                Pointer.to(new float[]{smoothingLength})
        );
        executeKernel1P1T(calculate_densities_F,kernelParameters);
    }

    /**
     * Performs a particle "momentum" update on the GPU.
     * Here "momentum" refers to the intermediate quantity Pressure/Density^2; which
     * would otherwise need to be wastefully recalculated n^2 times.
     */
    public static void updateMomentumGPU(MatrixPointer momenta,
                                         MatrixPointer densities){
        Pointer kernelParameters = Pointer.to(
                Pointer.to(momenta.getDevicePointer()),
                Pointer.to(densities.getDevicePointer())
        );
        executeKernel1P1T(calculate_momenta_F,kernelParameters);
    }

    /**
     * Initialize the constants (values that do not change).
     * This memory is typically faster than local memory.
     */
    public static void setGPUConstants(int dimensions, int N_particles, float particleMass,
                                       float G, float K, float adiabaticIndex, float epsilon, float alpha, float beta){
        setGPUConstantInt("dimensions",dimensions,physicsModule);
        setGPUConstantInt("N_particles",N_particles,physicsModule);

        setGPUConstantFloat("particleMass",particleMass,physicsModule);
        setGPUConstantFloat("G",G,physicsModule);

        // Equation of state.
        setGPUConstantFloat("K",K,physicsModule);
        setGPUConstantFloat("adiabaticIndex",adiabaticIndex,physicsModule);

        // Viscosity constants.
        setGPUConstantFloat("epsilon",epsilon,physicsModule);
        setGPUConstantFloat("alpha",alpha,physicsModule);
        setGPUConstantFloat("beta",beta,physicsModule);
        cudaDeviceSynchronize();
    }

    /**
     * CUDA-OpenGL interoperability.
     * We use a custom a kernel that copies particle positions from the positions array in VRAM to the
     * vertex buffer object (VBO) holding positional offsets for the mesh instances. This provides an efficient
     * way of rendering the particles.
     */
    public static void GPU_GUI_UPDATE(MatrixPointer positions,CUgraphicsResource vbo_resource, int N_particles, int vbo_offset_position){
        // Map the vbo_resource for CUDA usage
        cuGraphicsMapResources(1, new CUgraphicsResource[]{vbo_resource}, null);
        // Get a device pointer that can be used in CUDA kernels
        CUdeviceptr vboPtr = new CUdeviceptr();
        cuGraphicsResourceGetMappedPointer(vboPtr, new long[1], vbo_resource);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(positions.getDevicePointer()),
                Pointer.to(vboPtr),
                Pointer.to(new int[]{vbo_offset_position}),
                Pointer.to(new int[]{N_particles}));
        executeKernel1P1T(copy_to_gui_F,kernelParameters);
        cuGraphicsUnmapResources(1, new CUgraphicsResource[]{vbo_resource}, null);
    }


    /**
     * Perform a particle acceleration update on the GPU.
     */
    public static void updateAccelerationsGPU(
                                          MatrixPointer densities,
                                          MatrixPointer velocities,
                                          MatrixPointer accelerations,
                                          MatrixPointer momenta,
                                          MatrixPointer positions, float splineGradConst, float gravGradConst, float smoothingLength, boolean reversed){
        Pointer kernelParameters = Pointer.to(
                Pointer.to(densities.getDevicePointer()),
                Pointer.to(velocities.getDevicePointer()),
                Pointer.to(accelerations.getDevicePointer()),
                Pointer.to(momenta.getDevicePointer()),
                Pointer.to(positions.getDevicePointer()),
                Pointer.to(new float[]{splineGradConst}),
                Pointer.to(new float[]{gravGradConst}),
                Pointer.to(new float[]{smoothingLength}),
                Pointer.to(new int[]{reversed ? 1 : 0}));
        executeKernel1P1T(calculate_accelerations_F,kernelParameters);
    }

    /**
     * Matrix multiplication, various different forms.
     * Dispatches to cuBLAS functions.
     * Presently unused.
     */
    public static void executeMultiplicationCM( MatrixPointer _matA,
                                                float scaleA,
                                                MatrixPointer _matB,
                                                float scaleResult,
                                                MatrixPointer _acc){
        int m = _matA.getRows();
        int n = _matB.getCols();
        int k = _matA.getCols();

        int lda = _matA.getRows();
        int ldb = _matB.getRows();
        int ldc = _acc.getRows();

        JCublas2.cublasGemmEx_new(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k, Pointer.to(new float[]{scaleA}),
                _matA.getDevicePointer(), CUDA_R_32F, lda,
                _matB.getDevicePointer(), CUDA_R_32F, ldb,
                Pointer.to(new float[]{scaleResult}), _acc.getDevicePointer(), CUDA_R_32F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    }

    public static void executeMultiplicationCMBT( MatrixPointer _matA,
                                                float scaleA,
                                                MatrixPointer _matB,
                                                float scaleResult,
                                                MatrixPointer _acc){
        int m = _matA.getRows();
        int n = _matB.getRows();
        int k = _matA.getCols();

        int lda = _matA.getRows();
        int ldb = _matB.getRows();
        int ldc = _acc.getRows();

        JCublas2.cublasGemmEx(handle,
                CUBLAS_OP_N, CUBLAS_OP_T,
                m, n, k, Pointer.to(new float[]{scaleA}),
                _matA.getDevicePointer(), CUDA_R_32F, lda,
                _matB.getDevicePointer(), CUDA_R_32F, ldb,
                Pointer.to(new float[]{scaleResult}), _acc.getDevicePointer(), CUDA_R_32F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    }

    public static void executeMultiplicationCMAT( MatrixPointer _matA,
                                                float scaleA,
                                                MatrixPointer _matB,
                                                float scaleResult,
                                                MatrixPointer _acc){
        int m = _matA.getRows();
        int n = _matB.getCols();
        int k = _matA.getCols();

        int lda = _matA.getRows();
        int ldb = _matB.getRows();
        int ldc = _acc.getRows();

        JCublas2.cublasGemmEx(handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                k, n, m, Pointer.to(new float[]{scaleA}),
                _matA.getDevicePointer(), CUDA_R_32F, lda,
                _matB.getDevicePointer(), CUDA_R_32F, ldb,
                Pointer.to(new float[]{scaleResult}), _acc.getDevicePointer(), CUDA_R_32F, ldc, CUDA_R_32F, CUBLAS_GEMM_DEFAULT);
    }


    public static void executeAddition(MatrixPointer _matA,
                                       float scalarA,
                                       MatrixPointer _matB,
                                       float scalarB,
                                       MatrixPointer _acc){
        int lda = _matA.getRows();
        int ldb = _matB.getRows();
        int ldc = _acc.getRows();

        int m = _matA.getRows();
        int n = _matB.getCols();

        JCublas2.cublasSgeam(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, n,
                Pointer.to(new float[]{scalarA}),
                _matA.getDevicePointer(), lda, Pointer.to(new float[]{scalarB}),
                _matB.getDevicePointer(), ldb,
                _acc.getDevicePointer(),  ldc);
        cudaDeviceSynchronize();
    }

    public static void executeDiff(MatrixPointer _matA,
                                    MatrixPointer _matB,
                                    MatrixPointer _acc){
        int lda = _matA.getRows();
        int ldb = _matB.getRows();
        int ldc = _acc.getRows();

        int m = _matA.getRows();
        int n = _matB.getCols();
        JCublas2.cublasSgeam(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                m, n,
                Pointer.to(new float[]{1F}),
                _matA.getDevicePointer(), lda, Pointer.to(new float[]{-1F}),
                _matB.getDevicePointer(), ldb,
                _acc.getDevicePointer(),  ldc);
    }

    public static void executeScale(MatrixPointer _mat,
                                   float scalar){
        int n = _mat.getLength1d();

        JCublas2.cublasSscal(handle,
                n,
                Pointer.to(new float[]{scalar}),
                _mat.getDevicePointer(), 1);
        cudaDeviceSynchronize();
    }

    // (int) ceil(a/b)
    private static int intDivCeil(int a, int b){
        return (a+b-1)/b;
    }

    // Execute kernel 1 particle 1 thread (a thread for each particle)
    private static void executeKernel1P1T(CUfunction kernel,
                                      Pointer parameters){

        //int threadsPerBlock = 1024; // Max value supported.
        int blocks = intDivCeil(SPH_GPU.N_PARTICLES, THREADS_PER_BLOCK);
        executeKernel(kernel,parameters,blocks, THREADS_PER_BLOCK);
    }

    public static int THREADS_PER_BLOCK = 1024;

    private static void executeKernel(CUfunction kernel,
                                      Pointer parameters,
                                      int gridDimension,
                                      int blockDimension){
        cuLaunchKernel(kernel,
                gridDimension,  1, 1,      // Grid dimension
                blockDimension, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                parameters, null // Kernel- and extra parameters
        );
        cuCtxSynchronize();
    }

    private static MatrixPointer add(MatrixPointer A, MatrixPointer B){
        MatrixPointer accumulator = MatrixPointer.zeroMatrix(A.getRows(), A.getCols()); // Create an accumulator.
        executeAddition(A,1F,B,1F,accumulator);
        return accumulator;
    }
    private static MatrixPointer addScaleLatter(MatrixPointer A, MatrixPointer B, float scalar){
        MatrixPointer accumulator = MatrixPointer.zeroMatrix(A.getRows(), A.getCols()); // Create an accumulator.
        executeAddition(A,1F,B,scalar,accumulator);
        return accumulator;
    }
    private static MatrixPointer addScaleFormer(MatrixPointer A, MatrixPointer B, float scalar){
        MatrixPointer accumulator = MatrixPointer.zeroMatrix(A.getRows(), A.getCols()); // Create an accumulator.
        executeAddition(A,scalar,B,1F,accumulator);
        return accumulator;
    }

    /**
     * Evaluates the physical properties such as energy and momentum,
     * and also counts the number of particles beyond 10 earth radii from the origin for an approximation
     * of how many have "escaped" beyond the initial simulation volume.
     */
    public static void computeAndLogEvalStats(float mass, MatrixPointer densities, // Scalar property
                                               MatrixPointer positions, // Vector properties
                                               MatrixPointer velocities,
                                               MatrixPointer accelerations,int N_particles){

        // Define matrix pointers for the output.
        MatrixPointer particleCalculatedMomentum = new MatrixPointer(new float[N_particles]);
        MatrixPointer particleCalculatedKineticEnergies = new MatrixPointer(new float[N_particles]);

        Pointer kernelParameters = Pointer.to(
                Pointer.to(densities.getDevicePointer()),
                Pointer.to(positions.getDevicePointer()),
                Pointer.to(velocities.getDevicePointer()),
                Pointer.to(accelerations.getDevicePointer()),
                Pointer.to(particleCalculatedMomentum.getDevicePointer()),
                Pointer.to(particleCalculatedKineticEnergies.getDevicePointer()));
        executeKernel1P1T(calculate_properties_F,kernelParameters);


        int escaped = 0;
        for (float[] p : positions.peek()) {
            float x,y,z;
            x=p[0];y=p[1];z=p[2];
            float originDist = (float) Math.sqrt(x*x+y*y+z*z);
            if(originDist > 10){
                escaped++;
            }
        }
        System.out.println("ESCAPED: " + escaped);
        System.out.println("MASS OF ESCAPEES: " + (mass*escaped));

        // Read off the output.
        float[] momenta = particleCalculatedMomentum.unload()[0];
        float[] kineticEnergies = particleCalculatedKineticEnergies.unload()[0];

        float sumMomentum = 0;
        float sumKineticEnergy=0;
        for (int i = 0; i < N_particles; i++) {
            sumMomentum+=momenta[i];
            sumKineticEnergy+=kineticEnergies[i];
        }

        sumMomentum/=N_particles;
        sumKineticEnergy/=N_particles;

        System.out.println(sumMomentum + " " + sumKineticEnergy);
        System.out.println("NET MOMENTUM: " + sumMomentum);
        System.out.println("NET KINETIC ENERGY: " + sumKineticEnergy);
    }


    /**
     * Leap-frog integration (order 2).
     *
     * We have values for the local accelerations of each particle, but to evolve the system, we need to
     * translate these to velocities, and then those velocities to new positions after some time has passed.
     * This requires two levels of integration.
     *
     * Many calculations in the acceleration pipeline are dependent on the current velocity, which is in turn dependent on the
     * acceleration. This makes the equations implicit, and so to resolve this, we work out the velocity and acceleration at interleaved
     * "half" time steps. Thus the position of the particles is linked to the average of the velocity as derived from acceleration, and the velocity
     * worked out half a time-step before hand.
     *
     * In practice, this gives good results, and is reasonably straight forward.
     */
    public static void integration(float timestep,
            MatrixPointer r, MatrixPointer v, MatrixPointer a){
        float half_timestep = timestep/2;

        // "Firstly, predict the positions at time ti+1/2 in a manner analogous to equation 3.152 via
        // ri+1/2 = ri + (δt/2)*vi"
        MatrixPointer r_next_half = addScaleLatter(r,v, half_timestep);

        // "Secondly, use equation 3.152 to obtain the velocity at time ti+1/2,
        //ai+1/2."
        // Eqn. 3.152 = vi+(δt/2)*ai = vi+1/2 -> vi+1/2 = vi+(δt/2)*ai
        MatrixPointer v_next_half = addScaleLatter(v,a,half_timestep);

        MatrixPointer a_next_half = MatrixPointer.zeroMatrix(a.getRows(),a.getCols()); // Create an accumulator.
        SPH_GPU.extrapolate(v_next_half,r_next_half,a_next_half,timestep<0); //...and extrapolate other values (such as density, internal energy and gravitational potential) at the half timestep also. Hence calculate the acceleration at the half timestep

        // "Calculate the velocity at time ti+1 using vi+1 = vi+δt * (ai_+1/2)."
        MatrixPointer v_next = addScaleLatter(v,a_next_half,timestep);

        // "Now update the positions to timestep ti+1 using ri+1 = ri +(δt/2)*(vi + vi+1)."
        MatrixPointer v_sum = add(v,v_next);
        MatrixPointer r_next = addScaleLatter(r,v_sum,half_timestep);

        v.assign(v_next); // Frees old v and updates for the next cycle
        r.assign(r_next); // Frees old r and updates for the next cycle

        // Free intermediates.
        r_next_half.free();
        v_next_half.free();
        a_next_half.free();
        v_sum.free();

        // Return to the rendering pipeline
    }
}
