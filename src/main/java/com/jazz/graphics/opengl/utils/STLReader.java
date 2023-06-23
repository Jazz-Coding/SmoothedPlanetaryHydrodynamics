package com.jazz.graphics.opengl.utils;

import com.jazz.graphics.opengl.primitives.simple.Vertex;

import java.awt.Color;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

/**
 * Helper for reading STL (stereolithography) files, a common file format for exporting meshes from other software.
 * We export the vertex data from an icosphere (a sphere with triangular faces) from Blender as our base mesh, but can
 * work with any valid STL file.
 */
public class STLReader {
    public static List<float[]> readSTLNormals(String stlFile){
        List<float[]> normals = new ArrayList<>();
        try {
            BufferedReader reader = new BufferedReader(new FileReader(stlFile));
            String line;

            while ((line = reader.readLine()) != null) {
                if(line.contains("facet normal")){
                    String[] spaced = line.split("facet normal ")[1].split(" ");
                    float normX = Float.parseFloat(spaced[0]);
                    float normY = Float.parseFloat(spaced[1]);
                    float normZ = Float.parseFloat(spaced[2]);

                    // Duplicate three times for each vertex in the triangle.
                    normals.add(new float[]{normX,normZ,normY,0});
                    normals.add(new float[]{normX,normZ,normY,0});
                    normals.add(new float[]{normX,normZ,normY,0});
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }


        return normals;
    }
    public static List<Vertex> readSTLVerts(String stlFile, Color assumedColor){
        List<Vertex> vertices = new ArrayList<>();

        try {
            BufferedReader reader = new BufferedReader(new FileReader(stlFile));
            String line;

            while ((line = reader.readLine()) != null) {
                if(line.contains("vertex")){
                    String[] spaced = line.split(" ");
                    float v1 = Float.parseFloat(spaced[1]);
                    float v2 = Float.parseFloat(spaced[2]);
                    float v3 = Float.parseFloat(spaced[3]);
                    vertices.add(new Vertex(v1, v2, v3, assumedColor));
                }
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }

        return vertices;
    }
}
