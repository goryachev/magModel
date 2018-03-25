#version 330 core
in vec3 coord, normal;
out float gl_ClipDistance[6];
flat out vec3 nrml;
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
    nrml = (MVP*vec4(normal,0.)).xyz*scale;
    //Viewport tansformation
    nrml *= vec3(viewport/l,1.);

    vec3 center = (vmin +vmax)*0.5;
    res.z = ((MVP*vec4(center ,1.)).z-res.z)*scale/distance(vmax,vmin)*2;// in [-1:1]
    gl_PointSize = radius*l/scale;
    gl_Position = res;
}
