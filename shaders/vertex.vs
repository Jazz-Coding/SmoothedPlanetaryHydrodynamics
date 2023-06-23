#version 330

layout(location = 0) in vec4 in_position; // vertices defined by 4 floating point values
layout(location = 1) in vec4 color; // vertex color
layout(location = 2) in vec4 instanceOffset;
layout(location = 3) in vec3 vertex_normal;

out vec4 decided_color;
out vec3 directionToLight;
flat out vec3 interpolated_normal; // output normal to fragment shader

// Perspective transformation information.
uniform mat4 modelMatrix; // Transforms from model -> world space
uniform mat4 viewMatrix; // Transforms from world -> camera space
uniform mat4 perspectiveMatrix; // Transforms from camera -> clip space

// Lighting
uniform vec4 lightPos;


void main() {
    vec4 position = (modelMatrix*in_position)+instanceOffset; // Each instance receives a different offset value.
    //vec4 dirToLight = lightPos-position;
    directionToLight = normalize((lightPos-position).xyz);

    float d = length((lightPos-position).xyz); // Simplified distance to light shading.
    float reduced = pow(d,1.25);

    float intensity = max(15/(reduced),0.3);

    decided_color = color*intensity;

    // Compute normal, no interpolation yet (flat shading).
    interpolated_normal = normalize(vertex_normal.xyz);

    // Compute position in clip space.
    mat4 MVP = perspectiveMatrix*viewMatrix;
    vec4 position_clip = MVP*position;
    gl_Position = position_clip;
}