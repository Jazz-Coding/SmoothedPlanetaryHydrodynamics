// Cubic spline function.
__device__ inline float spEqn1(float x){
    float x2 = x*x;
    return ((3*x)/4 - 3/2)*x2 + 1;
}

__device__ inline float spEqn2(float x){
    float a = 2-x;
    float a3 = a*a*a;
    return a3/4;
}

__device__ inline float cubicSpline(float distance, float splineConst, float smoothingLength){
    // precomputedSplineConstant = 1/4*pi*h^3
    float x = distance/smoothingLength; // The number of "smoothing lengths" away this interaction occurred.
    if(x <= 1){
        return splineConst * spEqn1(x);
    } else if (x <= 2){
        return splineConst * spEqn2(x);
    } else {
        return 0;
    }
}

// Gradient of cubic spline function, derived explicitly.
__device__ inline float spGEqn1(float x){
    return x*(9*x/4-3*x);
}
__device__ inline float spGEqn2(float x){
    float a = 2-x;
    return (-3/4)*(a*a);
}
__device__ inline float cubicSplineGradient(float distance, float splineGradConstant, float smoothingLength){
    float x = distance/smoothingLength;

    if(x <= 1){
        return splineGradConstant * spGEqn1(x);
    } else if (x <= 2){
        return splineGradConstant * spGEqn2(x);
    } else {
        return 0;
    }
}

// Gravitation softening function.
// todo

// Gradient of gravitation softening function, derived explicitly.
__device__ inline float gravGradEqn1(float x){
        float x2 = x*x;
        return x*((x/2 - 6/5)*x2 + 4/3);
}
__device__ inline float gravGradEqn2(float x){
    // 8/3x -3x^2 +6/5x^3 -1/6x^4 -1/(15x^2)
    // -->
    // or
    // (x^3 (x ((36 - 5 x) x - 90) + 80) - 2)/ (30 x^2)
    float x2 = x*x;
    float x3 = x2*x;
    return (x3 * (x * ((36-5*x)*x-90)+80)-2) / (30*x2);
}

__device__ inline float gravGradSoften(float distance, float gravGradConstant, float smoothingLength){
    float x = distance/smoothingLength; // The number of "softeningLength lengths" away this interaction occurred.
    if(x <= 1){
        return gravGradConstant * gravGradEqn1(x);
    } else if (x <= 2){
        return gravGradConstant * gravGradEqn2(x);
    } else {
        return 1/(distance*distance);
    }
}