package com.jazz.graphics.opengl.primitives.simple;

import java.awt.Color;

/**
 * The simplest unit of geometry, a single vertex, defined by an XYZ coordinate (along with a W coordinate
 * that has uses in calculating perspective projections), a color, and an optional normal vector that enables
 * shading the geometry.
 */
public class Vertex {
    float[] position; // XYZW
    float[] color; // RGBA
    float[] faceNormal;

    public Vertex(float x, float y, float z, float w, Color vertexColor){
        position = new float[]{x,y,z,w};
        this.color = new float[]{0,0,0,1}; // alpha = 1 typically
        vertexColor.getRGBColorComponents(this.color);
    }

    // We typically always use w=1.0.
    public Vertex(float x, float y, float z, Color vertexColor){
        this(x,y,z,1F,vertexColor);
    }

    public float[] getPosition(){
        return position;
    }
    public float[] getColor(){
        return color;
    }

    public void setColor(float[] color) {
        this.color = color;
    }

    public float[] getFaceNormal() {
        return faceNormal;
    }

    public void setFaceNormal(float[] faceNormal) {
        this.faceNormal = faceNormal;
    }
}
