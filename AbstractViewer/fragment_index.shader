#version 330 core
in float c;
//flat in ivec3 intcolor;
out ivec4 FragColor;
uniform sampler1D pal;
void main() {
    if (c <0. || c>1.) discard;
    int r = (gl_PrimitiveID & 255)-128;
    int g = ((gl_PrimitiveID >> 8 ) & 255)-128;
    int b = ((gl_PrimitiveID >> 16) & 65535)-128;
    FragColor = ivec4(r,g,b,127);
    gl_FragDepth =gl_FragCoord.z ;
}
