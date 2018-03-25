#version 330 core
flat in vec3 nrml;
in float gl_ClipDistance[6];
out vec4 FragColor;
uniform mat4 MVP, itMVP;
uniform vec3 vmin, vmax;
uniform vec2 viewport;
uniform float radius, scale;
uniform sampler1D pal;
const float lighting=0.75;
void main() {
    // Вроде и без этого должно раьотать, но как-то на amd не очень
    //for(int i=0; i<6; i++){
    //    if (gl_ClipDistance[i]<=0) discard;
    //}
    const float camera_light = 0.95;
    const float vertical_light = 1.0;
    vec2 shift = gl_PointCoord.xy -vec2(0.5,0.5);
    float d = dot(shift,shift);
    if (d>= 0.25) discard;
    float dz = sqrt(1. - 4.*d);
    int tex_length = textureSize(pal,0);
    shift.y *= -1; // это не очень сильное колдунство, координаты по y повернуты вниз
    float col = ((dot(vec3(shift,dz*0.5), nrml)+.5)*(tex_length - 1.0)+0.5)/tex_length;
    gl_FragColor =texture(pal, col)*(1. - lighting*camera_light*(1.-dz));
    gl_FragDepth =gl_FragCoord.z-radius*dz/distance(vmax,vmin);
}
