#version 330 core
in vec3 coord;
out float gl_ClipDistance[6];
flat out ivec3 intcolor;
//flat out int r;
uniform mat4 MVP, itMVP;
uniform vec3 vmin, vmax;
uniform vec2 viewport;
uniform float radius, scale;
const float lighting=0.75;
void main() {
    float l = min(viewport.x, viewport.y);
    vec4 res = MVP*(vec4(coord,1.0));
    for(int i =0; i<3; i++){
        gl_ClipDistance[i]=coord[i]- vmin[i];
        gl_ClipDistance[i+3]=vmax[i] -coord[i];
    }
	//int r = gl_InstanceID;
    int r = (gl_VertexID & 255)-128;
    int g = ((gl_VertexID >> 8 ) & 255)-128;
    int b = ((gl_VertexID >> 16) & 65535)-128;
    //int a = 128;
    intcolor = ivec3(r,g,b);
    vec3 center = (vmin +vmax)*0.5;
    res.z = ((MVP*vec4(center ,1.)).z -res.z)*scale/distance(vmax,vmin)*2;// in [-1:1]
    gl_PointSize = radius*l/scale;
    gl_Position = res;
}
