package com.jazz.graphics.opengl;

import org.joml.Matrix4f;
import org.joml.Vector3f;
import org.joml.Vector4f;

import static com.jazz.utils.Constants.PI;
import static com.jazz.utils.Constants.TAU;

/**
 * Abstraction of the camera viewport.
 * Handles creative the perspective projection matrix, a component in the MVP matrix used for rendering 3D
 * environments.
 *
 * Also performs arcball camera motion, which permits orbiting the camera around a target point,
 * zooming in and out, and translating left/right/up/down relative to the camera.
 */
public class Camera {
    private float FOV;
    private float aspectRatio = 1920/1080F;
    private float nearClippingPlane = 0.01f;
    private float farClippingPlane = 1000.0f;

    private Matrix4f viewMatrix = new Matrix4f();
    private Matrix4f inverseViewMatrix = new Matrix4f();
    private Matrix4f perspectiveMatrix = new Matrix4f();

    private Vector3f targetPos = new Vector3f();
    private Vector3f cameraPos = new Vector3f();
    private Vector3f upVector = new Vector3f(0.0f, 1.0f, 0.0f); // The Y axis is up here.

    // Spherical coordinate copy for orbiting.
    float r;
    float theta;
    float phi;

    private void updateViewMatrix(){
        updateCartesianCoordinates();
        cameraPos.add(targetPos);

        viewMatrix = new Matrix4f().lookAt(cameraPos, targetPos, upVector);
        viewMatrix.invert(inverseViewMatrix);
    }
    private void updatePerspectiveMatrix(){
        perspectiveMatrix = new Matrix4f().perspective(FOV, aspectRatio, nearClippingPlane, farClippingPlane);
    }
    private void updateSphericalCoordinates(){
        float x,y,z;
        float epsilon = 0.001F;

        x=cameraPos.x;y=cameraPos.z;z=cameraPos.y;
        r = (float) Math.sqrt(x * x + y * y + z * z);
        theta = (float) Math.acos(z / (r + epsilon));
        phi = (float) (Math.signum(y) * Math.acos(x / (Math.sqrt(x * x + y * y)+epsilon)));
        if (phi < 0) {
            phi += 2 * Math.PI;
        }
    }

    private void updateCartesianCoordinates(){
        float x,y,z;
        x= (float) (r*Math.sin(theta)*Math.cos(phi));
        z= (float) (r*Math.sin(theta)*Math.sin(phi));
        y= (float) (r*Math.cos(theta));

        cameraPos = new Vector3f(x,y,z);
    }

    public void translate(float dx, float dy){
        Vector4f temp = new Vector4f(targetPos,0);
        inverseViewMatrix.transform(-dx,dy,0,0,temp); // Work in camera space so the translation occurs from the user's perspective.
        targetPos.add(temp.x,temp.y,temp.z);
    }

    private float epsilon = 1e-6F;

    private float dx=1;
    private float dy=1;

    private void changePhi(float change){
        phi+=change;
        if(phi < 0){
            // Wrap around.
            phi = TAU+phi;
        } else if (phi > TAU){
            phi = phi-TAU;
        }
    }


    private void changeTheta(float change){
        float lowerBound = 0+epsilon;
        float upperBound = PI-epsilon;
        theta+=change;

        // Handle wrap-arounds.
        if(theta < lowerBound){
            dy*=-1; // Flip the user controls (camera up direction and vertical drag direction) to make the transition seamless.
            theta = lowerBound+Math.abs(lowerBound-theta);
            upVector = new Vector3f(0,upVector.y*-1,0);
            changePhi(PI);
        } else if (theta>upperBound){
            dy*=-1;
            theta = upperBound-(theta-upperBound);
            upVector = new Vector3f(0,upVector.y*-1,0);
            changePhi(PI);
        }
    }

    public void orbitTarget(float longitudeChange, float latitudeChange){
        // Change the spherical coordinates.
        changeTheta(dy*(float) Math.toRadians(latitudeChange));
        changePhi(dx*(float) Math.toRadians(longitudeChange));
    }
    public void zoom(float zoomAmount){
        if(zoomAmount < r-epsilon){
            // Change the spherical coordinates.
            r-=zoomAmount;
        }
    }

    public Camera(Vector3f origin, float FOV) {
        this.targetPos = new Vector3f(0,0,0);
        this.cameraPos = origin;
        this.FOV = (float) Math.toRadians(FOV);

        updateSphericalCoordinates();
        updatePerspectiveMatrix();
    }

    public Matrix4f getViewMatrix() {
        updateViewMatrix();
        return viewMatrix;
    }

    public Matrix4f getPerspectiveMatrix() {
        updatePerspectiveMatrix();
        return perspectiveMatrix;
    }

    public Vector3f getTargetPos() {
        return targetPos;
    }

    public Vector3f getCameraPos() {
        return cameraPos;
    }

    public Vector3f getTargetToCamVector(float length){
        Vector3f temp = new Vector3f();
        targetPos.sub(cameraPos,temp);
        temp.normalize(length,temp);
        return temp;
    }
}
