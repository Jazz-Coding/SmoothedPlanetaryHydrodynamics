package com.jazz.graphics.opengl.primitives;

import com.jazz.graphics.opengl.primitives.simple.Vertex;
import org.joml.Vector3f;

import java.util.ArrayList;
import java.util.List;

/**
 * Helper for building a mesh given an input vertex list.
 */
public class MeshBuilder {
    private float[] vertexData;
    private List<Integer> componentSizes = new ArrayList<>();

    private Vector3f originPosition = new Vector3f(0,0,0);
    private Vector3f axisRotationAngles = new Vector3f(0,0,0);
    private Vector3f scale =  new Vector3f(1,1,1);

    public Mesh build(){
        return new Mesh(vertexData,componentSizes, originPosition, axisRotationAngles, scale);
    }

    public MeshBuilder collectVerticesWithInstanceOffsetsAndNormals(List<Vertex> vertices, List<float[]> offsets, List<float[]> vertexNormals){
        List<Float> values = new ArrayList<>();

        // Add all the positions.
        for (Vertex vertex : vertices) {
            float[] pos = vertex.getPosition();
            addArrayToList(pos,values);
        }
        componentSizes.add(vertices.size());

        // Add all the colors.
        for (Vertex vertex : vertices) {
            float[] col = vertex.getColor();
            addArrayToList(col,values);
        }
        componentSizes.add(vertices.size());

        // Add the offsets.
        for (float[] offset : offsets) {
            addArrayToList(offset,values);
        }
        componentSizes.add(offsets.size());

        for (float[] vertexNormal : vertexNormals) {
            addArrayToList(vertexNormal,values);
        }
        componentSizes.add(vertexNormals.size());

        // Convert back to float array.
        float[] array = new float[values.size()];
        for (int i = 0; i < values.size(); i++) {
            array[i] = values.get(i);
        }

        vertexData = array;
        return this;
    }
    public MeshBuilder collectVerticesWithInstanceOffsets(List<Vertex> vertices, List<float[]> offsets){
        List<Float> values = new ArrayList<>();

        // Add all the positions.
        for (Vertex vertex : vertices) {
            float[] pos = vertex.getPosition();
            addArrayToList(pos,values);
        }
        componentSizes.add(vertices.size());

        // Add all the colors.
        for (Vertex vertex : vertices) {
            float[] col = vertex.getColor();
            addArrayToList(col,values);
        }
        componentSizes.add(vertices.size());

        // Add the offsets.
        for (float[] offset : offsets) {
            addArrayToList(offset,values);
        }
        componentSizes.add(offsets.size());

        // Convert back to float array.
        float[] array = new float[values.size()];
        for (int i = 0; i < values.size(); i++) {
            array[i] = values.get(i);
        }

        vertexData = array;
        return this;
    }
    public MeshBuilder collectVertices(List<Vertex> vertices){
        List<Float> values = new ArrayList<>();

        // Add all the positions.
        for (Vertex vertex : vertices) {
            float[] pos = vertex.getPosition();
            addArrayToList(pos,values);
        }
        componentSizes.add(vertices.size());

        // Add all the colors.
        for (Vertex vertex : vertices) {
            float[] col = vertex.getColor();
            addArrayToList(col,values);
        }
        componentSizes.add(vertices.size());

        // Convert back to float array.
        float[] array = new float[values.size()];
        for (int i = 0; i < values.size(); i++) {
            array[i] = values.get(i);
        }

        vertexData = array;
        return this;
    }
    public MeshBuilder collectVerticesOnlyPositions(List<Vertex> vertices){
        List<Float> values = new ArrayList<>();

        // Add all the positions.
        for (Vertex vertex : vertices) {
            float[] pos = vertex.getPosition();
            addArrayToList(pos,values);
        }
        componentSizes.add(vertices.size());

        // Convert back to float array.
        float[] array = new float[values.size()];
        for (int i = 0; i < values.size(); i++) {
            array[i] = values.get(i);
        }

        vertexData = array;
        return this;
    }

    public MeshBuilder setOrigin(float x, float y, float z){
        originPosition = new Vector3f(x,y,z);
        return this;
    }
    public MeshBuilder setRotationAngles(float x, float y, float z){
        axisRotationAngles = new Vector3f(x,y,z);
        return this;
    }
    public MeshBuilder setScale(float x, float y, float z){
        scale = new Vector3f(x,y,z);
        return this;
    }

    private static void addArrayToList(float[] array, List<Float> list){
        for (float v : array) {
            list.add(v);
        }
    }
}
