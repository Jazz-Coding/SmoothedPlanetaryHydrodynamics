package com.jazz;

/**
 * Smoothing kernels for approximating the derivative values of
 * scalar fields (i.e. pressure) over 3D space.
 *
 * The smoothing length is passed as a parameter to enable varying it dynamically,
 * a key strength of smoothed-particle hydrodynamics.
 */
public class SmoothingFunctions {
    // Pre-computed constants for neatness.
    private static final float PI = (float) Math.PI;
    private static final float SQRT_OF_PI = (float) Math.sqrt(Math.PI);

    /**
     * Floating point (32 bit) math functions, saves having to insert casts constantly in
     * the code.
     */
    private static float exp(float number){
        return (float) Math.exp(number);
    }
    private static float pow(float number, float exponent){
        return (float) Math.pow(number,exponent);
    }

    private static float log(float number){
        return (float) Math.log(number);
    }

    public static float[] splineConstants = new float[]{1F/6,5/(14*PI),1/(4*PI)};

    /**
     * The smoothing length changes at most once per timestep, which means recomputing it for every interaction is a wasteful endeavour,
     * and so the function accepts the precomputed spline constant as a parameter instead, which the caller can update only when needed.
     */
    private static float spEqn1(float x){
        // (2-x)^3 = (((6-x)*x-12)*x) + 8
        //return (((3*x-6)*x)*x)+4;
        float a = 2-x;
        float b = 1-x;
        return (a*a*a) - 4*(b*b*b);
        //return (((6-x)*x)-12)*x)+8; // 2 multiplications (down from 3)
    }

    private static float spEqn2(float x){
        float a = 2-x;
        return a*a*a;
        //return 8 + x*(-12 + (x*(6-x)));
    }


    public static float cubicSplinePC(float x, float precomputedSplineConstant){
        float y = x*2;
        if(y <= 1){
            return precomputedSplineConstant * spEqn1(y);
        } else if (y <= 2){
            return precomputedSplineConstant * spEqn2(y);
        }
        return 0;
    }

    public static float cubicSpline(float normalizedDistance, float smoothingLength, float precomputedSplineConstant){
        float x = normalizedDistance/smoothingLength; // The number of "smoothing lengths" away this interaction occurred.
        if(x <= 1){
            return precomputedSplineConstant * spEqn1(x);
        } else if (x <= 2){
            return precomputedSplineConstant * spEqn2(x);
        } else {
            return 0;
        }
    }

    private static float gravEqn1(float x, float x2){
        // (2/3)x^2 -(3/10)x^3 + (1/10)x^5 -7/5 // 12 multiplications
        // (1/10)x^5 -(3/10)x^3 + (2/3)x^2 -7/5
        // ((1/10)x^4 -(3/10)x^2 + (2/3)x)x -7/5
        // (((1/10)x^3 -(3/10)x + 2/3)x)x -7/5 //
        // ((((1/10)x^2 -3/10)x + 2/3)x)x -7/5 // 5 multiplications
        // Equation coefficients.
        float a = 1/10F;
        float b = -3/10F;
        float c = 2/3F;
        float d = -7/5F;
        //float x2 = x*x;
        return ((((a*x2 +b)*x + c)*x)*x) +d;
    }
    private static float gravEqn2(float x, float x2){
        // (4/3)x^2 - x^3 + (3/10)x^4 - (1/30)x^5 - (8/5) + (1/15x)
        // -(1/30)x^5 + (3/10)x^4 -x^3 + (4/3)x^2 -(8/5) + (1/15)x^-1
        // (-(1/30)x^4 + (3/10)x^3 -x^2 + (4/3)x)x -(8/5) + (1/15)x^-1
        // ((-(1/30)x^3 + (3/10)x^2 -x + (4/3))x)x -(8/5) + (1/15)x^-1
        // (((-(1/30)x^2 + (3/10)x -1)x + (4/3))x)x -(8/5) + (1/15)x^-1
        //float x2 = x*x;
        float x_ = 1/x;

        float a = -1/30F;
        float b = 3/10F;
        float c = 4/3F;
        float d = -8/5F;
        float e = 1/15F;

        return (((a*x2 + b*x -1)*x + c)*x)*x +d + e*x_;
    }

    public static float gravSoften(float normalizedDistance, float smoothingLength){
        float x = normalizedDistance/smoothingLength; // The number of "smoothing lengths" away this interaction occurred.
        float x2 = x*x;

        float c = 1/smoothingLength;

        if(x <= 1){
            return c * gravEqn1(x,x2);
        } else if (x <= 2){
            return c * gravEqn2(x,x2);
        } else {
            return -1/normalizedDistance;
        }
    }

    private static float gravGradEqn1(float x){
        // (4/3)x - (6/5)x^3 + (1/2)x^4
        // (1/2)x^4 - (6/5)x^3 + (4/3)x
        // x((1/2)x^3 - (6/5)x^2 + 4/3)
        // x(x((1/2)x^2 - (6/5)x) + 4/3)
        // ((((1/2)x - 6/5)x)x + 4/3)x
        float a = 1F/2;
        float b = -6F/5;
        float c=  4F/3;
        return (((a*x + b)*x)*x + c)*x;
    }
    private static float gravGradEqn2(float x){
        //(8/3)x - 3x^2 + (6/5)x^3 - (1/6)x^4 - (1/(15x^2))
        // a = 8/3, b = -3, c = 6/5, d = -1/6, e = -1/15
        // ax + bx^2 + cx^3 +dx^4 +ex^-2
        // dx^4 + cx^3 + bx^2 + ax + ex^-2
        // (dx^3 + cx^2 + bx + a)x + ex^-2
        // ((dx^2 + cx + b)x + a)x + ex^-2
        // (((dx + c)x + b)x + a)x + ex^-2
        float a = 8F/3, b = -3F, c = 6F/5, d = -1F/6, e = -1F/15;
        float x_2 = 1/(x*x);
        return (((d*x + c)*x + b)*x + a)*x + e*x_2;
    }
    public static float gravGradSoften(float normalizedDistance, float smoothingLength){
        float x = normalizedDistance/smoothingLength; // The number of "smoothing lengths" away this interaction occurred.
        float c = 1/(smoothingLength);
        if(x <= 1){
            return c * gravGradEqn1(x);
        } else if (x <= 2){
            return c * gravGradEqn2(x);
        } else {
            return 1/(normalizedDistance);
        }
    }

    private static float spGEqn1(float x){
        // (0.5x^4-3x^3+7.5x^2-9x+4) - (4x+6x^2-4x^3+x^4)
        // = -0.5x^4+x^3+1.5x^2-13x+4
        // a=-0.5, b=1, c=1.5,d=-13,e=4;
        // (ax^4+x^3+cx^2+d)x+e
        // ((ax^3+x^2+cx)x+d)x+e
        // (((ax^2+x+c)x)x+d)x+e
        // // ((((ax+1)x+c)x)x+d)x+e

        float a=-0.5F, c=1.5F,d=-13F,e=4F;
        return (((((a*x+1)*x+c)*x)*x+d)*x)+e;
    }
    private static float spGEqn2(float x){
        // -4+8x-6x^2+2x^3-(1/4)x^4
        // -(1/4)x^4 + 2x^3 -6x^2 +8x -4
        // a=-1/4, b=2, c=-6, d=8, e=-4
        // ax^4+bx^3+cx^2+dx+e
        // (ax^3+bx^2+cx+d)x+e
        // ((ax^2+bx+c)x+d)x+e
        // (((ax+b)x+c)x+d)x+e
        float a=-1F/4, b=2, c=-6, d=8, e=-4;
        return ((((a*x+b)*x+c)*x+d)*x)+e;
    }
    public static float cubicSplineGradient(float normalizedDistance, float smoothingLength, float precomputedSplineConstant){
        float x = normalizedDistance/smoothingLength;

        if(x <= 1){
            return precomputedSplineConstant * spGEqn1(x);
        } else if (x <= 2){
            return precomputedSplineConstant * spGEqn2(x);
        } else {
            return 0;
        }
    }
}
