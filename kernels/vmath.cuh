#ifndef VMATH_CUH
#define VMATH_CUH

/**
    Out of place float3 vector operations.
*/
__device__ __forceinline__ float3 sub(float3 vecA, float3 vecB){
    return {vecA.x-vecB.x, vecA.y-vecB.y, vecA.z-vecB.z};
}

__device__ __forceinline__ float3 add(float3 vecA, float3 vecB){
    return {vecA.x+vecB.x, vecA.y+vecB.y, vecA.z+vecB.z};
}

__device__ __forceinline__ float3 addScaled(float3 vecA, float3 vecB, float scalarA, float scalarB){
    return {vecA.x*scalarA+vecB.x*scalarB,
            vecA.y*scalarA+vecB.y*scalarB,
            vecA.z*scalarA+vecB.z*scalarB};
}

__device__ __forceinline__ float3 mul(float3 vecA, float3 vecB){
    return {vecA.x*vecB.x, vecA.y*vecB.y, vecA.z*vecB.z};
}

// Sum squares.
__device__ __forceinline__ float norm1(float3 vector){
    return vector.x*vector.x+vector.y*vector.y+vector.z*vector.z;
}

// Euclidean norm.
__device__ __forceinline__ float norm2(float3 vector){
    return sqrt(vector.x*vector.x+vector.y*vector.y+vector.z*vector.z);
}

// Dot product.
__device__ __forceinline__ float dot(float3 vecA, float3 vecB){
    return vecA.x*vecB.x+vecA.y*vecB.y+vecA.z*vecB.z;
}

// Scalar multiplication.
__device__ __forceinline__ float3 scal(float3 vector, float scalar){
    return {vector.x*scalar, vector.y*scalar, vector.z*scalar};
}

__device__ __forceinline__ float3 unit(float3 vector, float length){
    return {vector.x/length, vector.y/length, vector.z/length};
}

/**
    In place float3 vector operations.
*/
__device__ __forceinline__ void setVector(float3 &vector, float3 xyz){
    vector.x = xyz.x;
    vector.y = xyz.y;
    vector.z = xyz.z;
}

__device__ __forceinline__ void addIP(float3 &vector, float3 xyz){
    vector.x += xyz.x;
    vector.y += xyz.y;
    vector.z += xyz.z;
}

__device__ __forceinline__ void IPsubXYZ(float3 &vector, float3 xyz){
    vector.x -= xyz.x;
    vector.y -= xyz.y;
    vector.z -= xyz.z;
}

__device__ __forceinline__ void IPscalXYZ(float3 &vector, float scalar){
    vector.x *= scalar;
    vector.y *= scalar;
    vector.z *= scalar;
}

#endif // VMATH_CUH