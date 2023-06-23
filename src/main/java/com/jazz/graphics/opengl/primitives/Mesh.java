package com.jazz.graphics.opengl.primitives;

import org.joml.Matrix4f;
import org.joml.Vector3f;

import java.util.List;

import static org.lwjgl.opengl.GL15.*;
import static org.lwjgl.opengl.GL15.GL_ARRAY_BUFFER;
import static org.lwjgl.opengl.GL20.*;
import static org.lwjgl.opengl.GL30.glBindVertexArray;
import static org.lwjgl.opengl.GL30.glGenVertexArrays;
import static org.lwjgl.opengl.GL31.glDrawArraysInstanced;
import static org.lwjgl.opengl.GL33C.glVertexAttribDivisor;

/**
 * A mesh is a collection of vertices connected with triangular faces.
 */
public class Mesh {
    // Constructed with.
    private float[] vertexInformation;
    private List<Integer> componentSizes;

    // For rendering.
    private Matrix4f modelMatrix;
    private Vector3f originPosition;
    private Vector3f axisRotationAngles;
    private Vector3f scale;

    private int _VBO;
    private int _VAO;

    public Mesh(float[] vertexInformation, List<Integer> componentSizes, Vector3f originPosition, Vector3f axisRotationAngles, Vector3f scale) {
        this.vertexInformation = vertexInformation;
        this.componentSizes = componentSizes;

        this.originPosition = originPosition;
        this.axisRotationAngles = axisRotationAngles;
        this.scale = scale;
        calculateModelMatrix();

        // Load into VBO on the GPU.
        int VBO = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER,VBO);
        glBufferData(GL_ARRAY_BUFFER,this.vertexInformation,GL_STREAM_DRAW);
        glBindBuffer(GL_ARRAY_BUFFER,0);
        this._VBO = VBO;

        // Create VAO for rendering.
        int VAO = glGenVertexArrays();
        glBindVertexArray(VAO);
        glBindBuffer(GL_ARRAY_BUFFER, VBO);

        int meshVertexCount = componentSizes.get(0);
        int particleCount = componentSizes.get(2);
        int totalNormals = componentSizes.get(3);

        specifyVertexProperty(0,4,meshVertexCount); // Specify the vertex information.
        specifyVertexProperty(1,4,meshVertexCount); // Specify the colors (one per vertex, although typically these are all identical).

        specifyVertexProperty(2,4,particleCount); // Specify the position offsets per particle.
        glVertexAttribDivisor(2,1); // Also include the fact that this should be updated once per particle instead of per vertex.

        specifyVertexProperty(3, 4,totalNormals); // Specify the face normals.
        glBindVertexArray(0); // Reset back to default.

        this._VAO = VAO;
    }

    private int layoutOffset;
    private void specifyVertexProperty(int layoutIndex, int individualSize, int totalInstances){
        glEnableVertexAttribArray(layoutIndex);
        glVertexAttribPointer(layoutIndex,individualSize,GL_FLOAT,false,0,layoutOffset);
        layoutOffset += totalInstances*individualSize*4; // instances * floats per instance * sizeof(float) in bytes
    }

    private void calculateModelMatrix(){
        Matrix4f translationMat = new Matrix4f().translate(originPosition);
        Matrix4f rotationMat = new Matrix4f().rotateXYZ(this.axisRotationAngles);
        Matrix4f scaleMat = new Matrix4f().scale(scale);
        this.modelMatrix = scaleMat.mul(rotationMat.mul(translationMat));
    }

    public void draw(int modelMatrixUniformHandle){
        // Apply this mesh's model matrix.
        glUniformMatrix4fv(modelMatrixUniformHandle, false,this.modelMatrix.get(new float[16]));
        glBindVertexArray(_VAO);
        glDrawArrays(GL_TRIANGLES, 0, componentSizes.get(0));
        glBindVertexArray(0);
    }

    public void drawInstances(int modelMatrixUniformHandle, int count){
        // Apply this mesh's model matrix.
        glUniformMatrix4fv(modelMatrixUniformHandle, false,this.modelMatrix.get(new float[16]));
        glBindVertexArray(_VAO);
        glDrawArraysInstanced(GL_TRIANGLES,0,componentSizes.get(0),count);
        glBindVertexArray(0);
    }

    public int getVAO() {
        return _VAO;
    }

    public int get_VBO() {
        return _VBO;
    }
}
