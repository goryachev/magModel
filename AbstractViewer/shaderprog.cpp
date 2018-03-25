#include "shaderprog.hpp"
#include <fstream>
//--------------------------------------------------------------------------------
//SHADERS
//--------------------------------------------------------------------------------

void ShaderProg::init(){
    sprog = ProgLink(vshader,fshader);
    //printf("link end\n");
    //AttachAttr(vattr, "coord", sprog);
    //AttachAttr(nattr, "normal", sprog);
    //AttachAttr(cattr, "color", sprog);
    //AttachUniform(mvp_loc, "MVP", sprog);
    //AttachUniform(it_mvp_loc, "itMVP", sprog);
    //AttachUniform(unif_minmax, "minmaxmul", sprog);
    //AttachUniform(tex_length, "tex_length",sprog);
    //AttachUniform(vmin, "vmin",sprog);
    //AttachUniform(vmax, "vmax",sprog);
    checkOpenGLerror();
    //printf("shaders end\n");
}
//--------------------------------------------------------------------------------
ShaderProg::ShaderProg(const char * vss, const char* fss){
    checkOpenGLerror();
    //printf("shaders begin\n");
    vshader = ShaderComp(GL_VERTEX_SHADER, vss);
    //printf("shaders v\n");
    fshader = ShaderComp(GL_FRAGMENT_SHADER, fss);
    //printf("shaders f\n");
    this->init();
}
//--------------------------------------------------------------------------------
ShaderProg::~ShaderProg(){
    glDeleteProgram(sprog);
}
//--------------------------------------------------------------------------------
void ShaderProg::extern_load(const char * vsf, const char* fsf){
    checkOpenGLerror();
    //printf("shaders begin\n");
    vshader = ShaderLoad(GL_VERTEX_SHADER, vsf);
    fshader = ShaderLoad(GL_FRAGMENT_SHADER, fsf);
    //printf("shaders init\n");
    this->init();
}
//--------------------------------------------------------------------------------
GLuint ShaderProg::ShaderLoad(GLenum shader_type, const char * shader_file){
    std::ifstream in(shader_file);
    std::string contents( (std::istreambuf_iterator<char>(in)),
            std::istreambuf_iterator<char>() );
    in.close();
    char * Source  = new char[contents.length() + 1];
    std::strcpy( Source, contents.c_str() );
    GLuint shade = ShaderComp(shader_type, Source);
    delete [] Source;
    return shade;
}
//--------------------------------------------------------------------------------
GLuint ShaderProg::ShaderComp(GLenum shader_type, const char * Source){
    //printf("shaders Comp\n");
    GLuint shade = glCreateShader(shader_type);
    //printf("shaders cre\n");
    glShaderSource(shade, 1, & Source, NULL);
    //printf("shaders comp\n");
    glCompileShader(shade);
    //if (shader_type == GL_VERTEX_SHADER) std::cout << "vertex shader \n";
    //else std::cout << "fragment shader \n";
    return shade;
}
//--------------------------------------------------------------------------------
// Шейдерная программа
GLuint ShaderProg::ProgLink(GLuint vShader, GLuint fShader){
    // Создаем программу и прикрепляем шейдеры к ней
    GLuint Program = glCreateProgram();
    glAttachShader(Program, vShader);
    glAttachShader(Program, fShader);
    // Линкуем шейдерную программу
    glLinkProgram(Program);
    // Проверяем статус сборки
    int link_ok;
    glGetProgramiv(Program, GL_LINK_STATUS, &link_ok);
    if(!link_ok)
    {
        std::cout << "error attach shaders \n";
        shaderLog(vShader);
        shaderLog(fShader);

    }
    return Program;
}
//--------------------------------------------------------------------------------
// Функция печати лога шейдера
void ShaderProg::shaderLog(unsigned int shader)
{
    int   infologLen   = 0;
    int   charsWritten = 0;
    char *infoLog;

    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infologLen);

    if(infologLen > 1)
    {
        infoLog = new char[infologLen];
        if(infoLog == NULL)
        {
            std::cout<<"ERROR: Could not allocate InfoLog buffer\n";
            exit(1);
        }
        glGetShaderInfoLog(shader, infologLen, &charsWritten, infoLog);
        std::cout<< "InfoLog: " << infoLog << "\n\n\n";
        delete[] infoLog;
    }
}
//--------------------------------------------------------------------------------
void ShaderProg::_AttachUniform(GLint & Unif, const char * name, GLuint Program) {
    Unif = glGetUniformLocation(Program, name);
    if(Unif == -1)
    {
        //std::cout << "could not bind uniform " << name << std::endl;
        return;
    }
}
//--------------------------------------------------------------------------------
// Прикрепление массива аттрибутов
void ShaderProg::_AttachAttr(GLint & Attr ,const char* attr_name,GLuint Program) {
    // Вытягиваем ID атрибута из собранной программы
    Attr = glGetAttribLocation(Program, attr_name);
    if(Attr == -1)
    {
       // std::cout << "could not bind attrib " << attr_name << std::endl;
        return;
    }
}
//--------------------------------------------------------------------------------
void ShaderProg::AttachUniform(GLint & Unif, const char * name){
    _AttachUniform(Unif, name, sprog);
}
void ShaderProg::AttachAttr(GLint & Attr ,const char* attr_name){
    _AttachAttr(Attr, attr_name, sprog);
}
//--------------------------------------------------------------------------------
/*template<int sur_size>*/ void ShaderProg::render(/*Surface<sur_size>*/ Plottable * surf, Viewer * view, Texture * tex ){
    glUseProgram(sprog);
    tex->use_texture(/*tex_length*/);
    view->plot(this);
    //load_mvp(view->calc_mvp(), view->calc_itmvp());
    //loadclip(view->get_vmin(), view->get_vmax());
    surf->plot(this);
    glUseProgram(0);
}
///*template<int sur_size>*/ void ShaderProg::render(/*Surface<sur_size>*/ Plottable * surf, Viewer * view){
//    glUseProgram(sprog);
//    //tex->use_texture(tex_length);
//    load_mvp(view->calc_mvp());
//    loadclip(view->get_vmin(), view->get_vmax());
//    surf->plot_index(vattr,nattr,cattr,unif_minmax);
//    glUseProgram(0);
//}
//template void ShaderProg::render<2>(Surface<2> * surf, Viewer * view, Texture * tex );
//template void ShaderProg::render<4>(Surface<4> * surf, Viewer * view, Texture * tex );
//--------------------------------------------------------------------------------
//void ShaderProg::loadclip(const glm::vec3 & vi , const glm::vec3 & va){
//    if (vmax != -1) glUniform3f( vmax, va.x,va.y,va.z);
//    if (vmin != -1) glUniform3f( vmin, vi.x,vi.y,vi.z);
//}
////--------------------------------------------------------------------------------
//void ShaderProg::load_mvp(const glm::mat4 & MVP, const glm::mat4& itMVP){
//    if (mvp_loc != -1) glUniformMatrix4fv( mvp_loc, 1, GL_FALSE, glm::value_ptr(MVP));
//    if (it_mvp_loc != -1) glUniformMatrix4fv( it_mvp_loc, 1, GL_FALSE, glm::value_ptr(itMVP));
//}

//--------------------------------------------------------------------------------
//TEXTURES
//--------------------------------------------------------------------------------
Texture::Texture(const float* pal, int length){
    tex_len = length;
    glGenTextures(1, &textureID);
    glGenSamplers(1, &samplerID);
    //glBindSampler(GL_TEXTURE0, samplerID);
    linear();
    //printf("gen tex sampl\n");
    checkOpenGLerror();
    glBindTexture(GL_TEXTURE_1D, textureID);
    //printf("bind tex sampl\n");
    checkOpenGLerror();
    //printf("pre load tex\n");
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RGB, tex_len, 0, GL_RGB, GL_FLOAT, pal);
    //printf("load tex\n");
    checkOpenGLerror();
    glBindTexture(GL_TEXTURE_1D, 0);
}
//--------------------------------------------------------------------------------
Texture::~Texture(){
    glDeleteTextures(1,&textureID);
    glDeleteSamplers(1,&samplerID);
}
//--------------------------------------------------------------------------------
void Texture::linear(){
    glSamplerParameteri(samplerID,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glSamplerParameteri(samplerID,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
}
//--------------------------------------------------------------------------------
void Texture::nearest(){
    glSamplerParameteri(samplerID,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glSamplerParameteri(samplerID,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
}
//--------------------------------------------------------------------------------
int Texture::get_length(){
    return tex_len;
}
//--------------------------------------------------------------------------------
//void Texture::load_texture(const float* pal, int length){
//    this->~Texture();
//    Texture(pal, length);
//
//}
//--------------------------------------------------------------------------------
void Texture::use_texture(/*GLint tli*/){
    //printf("texture use begin\n");
    //checkOpenGLerror();
    //glEnable(GL_TEXTURE_1D);
    //checkOpenGLerror();
    //printf("texture enabled\n");
    //glUniform1f(tli,(float)tex_len);
    glActiveTexture(GL_TEXTURE0);
    //printf("texture activated\n");
    glBindTexture(GL_TEXTURE_1D, textureID);
    //checkOpenGLerror();
   // printf("texture binded\n");
    glBindSampler(0, samplerID);
    //checkOpenGLerror();
    //printf("sampler activated\n");
    //checkOpenGLerror();
    //printf("texture use set\n");
}
//--------------------------------------------------------------------------------
// FRAMEBUFFERS
//--------------------------------------------------------------------------------
FrameBuffer::FrameBuffer(int w, int h, int _type){
	_width = w; _height = h;
	type= (GLenum)_type;
	glGenFramebuffers(1,&frame_buffer);
	glBindFramebuffer(GL_FRAMEBUFFER,frame_buffer);
	glGenRenderbuffers(1,&render_buffer);
	glBindRenderbuffer(GL_RENDERBUFFER, render_buffer);
	glRenderbufferStorage(GL_RENDERBUFFER, type, _width, _height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, render_buffer);
	//glBindRenderbuffer(GL_RENDERBUFFER, 0);
	glGenRenderbuffers(1,&depth_buffer);
	glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, _width, _height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer);
	GLint status = glCheckFramebufferStatus(GL_FRAMEBUFFER) ;
	//std::cout<<"fb init"<<std::endl;
	checkOpenGLerror();
	if(status != GL_FRAMEBUFFER_COMPLETE) {
		std::cout<<"failed to make complete framebuffer object "<< status<<std::endl;
	}
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	glBindFramebuffer(GL_FRAMEBUFFER,0);
}
//--------------------------------------------------------------------------------
FrameBuffer::~FrameBuffer(){
	glDeleteFramebuffers(1,&frame_buffer);
	glDeleteRenderbuffers(1,&render_buffer);
	glDeleteRenderbuffers(1,&depth_buffer);
}
//--------------------------------------------------------------------------------
void FrameBuffer::resize(int w, int h){
	_width = w; _height = h;
	glBindFramebuffer(GL_FRAMEBUFFER,frame_buffer);
	glBindRenderbuffer(GL_RENDERBUFFER,render_buffer);
	glRenderbufferStorage(GL_RENDERBUFFER, type, _width, _height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, render_buffer);
	glBindRenderbuffer(GL_RENDERBUFFER, depth_buffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, _width, _height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_buffer);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER,0);
}
//--------------------------------------------------------------------------------
void FrameBuffer::bind_draw(){
	glBindFramebuffer(GL_FRAMEBUFFER,frame_buffer);
	glBindRenderbuffer(GL_RENDERBUFFER,render_buffer);
}
//--------------------------------------------------------------------------------
void FrameBuffer::bind_read(){
	glReadBuffer(GL_COLOR_ATTACHMENT0);
}
//--------------------------------------------------------------------------------
void FrameBuffer::relax(){
	glBindFramebuffer(GL_FRAMEBUFFER,0);
}
//--------------------------------------------------------------------------------
