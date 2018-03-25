#include "plottable.hpp"
//--------------------------------------------------------------------------------
// Vertex array object (VAO)
//--------------------------------------------------------------------------------
VertexArray::VertexArray(){
	glGenVertexArrays(1, &AO);
	glGenBuffers(1, &EBO);
	this->bind();
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    this->release();
    checkOpenGLerror();
}
//--------------------------------------------------------------------------------
VertexArray::~VertexArray(){
	for(auto && BO : BOs) if(BO) glDeleteBuffers(1,&BO);
	if (EBO) glDeleteBuffers(1,&EBO);
	if (AO) glDeleteVertexArrays(1,&AO);
}
//--------------------------------------------------------------------------------
void VertexArray::add_buffer(){
	BOs.emplace_back();
	attrs.emplace_back();
	this->bind();
	glGenBuffers(1, &BOs.back());
	this->release();
    checkOpenGLerror();
}
//--------------------------------------------------------------------------------
void VertexArray::bind(){
	glBindVertexArray(AO);
}
//--------------------------------------------------------------------------------
void VertexArray::release(){
	glBindVertexArray(0);
}
//--------------------------------------------------------------------------------
void VertexArray::load_data(int pos, int size, const void * data){
	this->bind();
	glBindBuffer(GL_ARRAY_BUFFER, BOs[pos]);
	glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);
	this->release();
    checkOpenGLerror();
}
//--------------------------------------------------------------------------------
GLuint & VertexArray::get_BO(int pos){
	return BOs[pos];
}
//--------------------------------------------------------------------------------
void VertexArray::load_indices(int size, const void * data){
	this->bind();
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, size, data, GL_STATIC_DRAW );
	this->release();
    checkOpenGLerror();
}
//--------------------------------------------------------------------------------
GLint & VertexArray::get_attr(int pos){
	return attrs[pos];
}
//--------------------------------------------------------------------------------
void VertexArray::enable_attr(int pos, int num, GLenum type){
	if (attrs[pos]!=-1){
		glEnableVertexAttribArray(attrs[pos]);
		glBindBuffer(GL_ARRAY_BUFFER, BOs[pos]);
		glVertexAttribPointer(attrs[pos], num, type, GL_FALSE, 0, 0);
	}
    checkOpenGLerror();
}
//------------------------------------------------------------------------------
// Plottable
//------------------------------------------------------------------------------
//Plottable::Plottable(){
//    //glGenVertexArrays(1, &VAO);
//    //glBindVertexArray(VAO);
//    //glGenBuffers(4,&VBO[0]);
//    //glBindVertexArray(0);
//};
////------------------------------------------------------------------------------
//Plottable::~Plottable(){
//    if (VBO[0]){
//        //std::cout<<"delete buffers"<<std::endl;
//        glDeleteBuffers(4,VBO);
//    }
//    if (VAO) glDeleteVertexArrays(1, &VAO);
//}
//------------------------------------------------------------------------------
// Axis
//------------------------------------------------------------------------------
void Axis::load_on_device(){
    int NTR = 3;
    //glGenVertexArrays(1, &VAO);
    //glBindVertexArray(VAO);
    //glGenBuffers(4,&VBO[0]);
    const glm::vec3 tri[6] = {  glm::vec3(0,0,0), glm::vec3(1,0,0), glm::vec3(0,0,0),
        glm::vec3(0,1,0), glm::vec3(0,0,0), glm::vec3(0,0,1) };
    const glm::vec3 norm[6] = {  glm::vec3(1,1,1), glm::vec3(1,1,1), glm::vec3(1,1,1),
        glm::vec3(1,1,1), glm::vec3(1,1,1), glm::vec3(1,1,1) };
    float ap[6] = {0.f,0.f,1.f,1.f,2.f,2.f};
    unsigned int indices[6] = {0,1, 2,3, 4,5};//Мы не можем сделать по-другому : цвет различается значит все индексы разные
    VAO.load_data(POS, sizeof(glm::vec3) * NTR*2, tri);
    //glBindBuffer(GL_ARRAY_BUFFER, VBO[POS]);
    //glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * NTR*2, tri, GL_STATIC_DRAW);
    VAO.load_data(NRM, sizeof(glm::vec3) * NTR*2, norm);
    VAO.load_data(CLR, sizeof(float) * NTR*2, ap);
    //glBindBuffer(GL_ARRAY_BUFFER, VBO[NRM]);
    //glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec3) * NTR*2, norm, GL_STATIC_DRAW);
    //glBindBuffer(GL_ARRAY_BUFFER, VBO[CLR]);
    //glBufferData(GL_ARRAY_BUFFER, sizeof(float) * NTR*2, ap, GL_STATIC_DRAW);
    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBO[IND]);
    //glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6*sizeof(unsigned int), indices, GL_STATIC_DRAW );
    VAO.load_indices(6*sizeof(unsigned int), indices);
    VAO.release();
}
//------------------------------------------------------------------------------
void Axis::AttachToShader(ShaderProg * spr) {
    //glBindVertexArray(VAO);
    VAO.bind();
    spr->AttachUniform(unif_minmax,"minmaxmul");
    spr->AttachAttr(VAO.get_attr(NRM),"normal");
    spr->AttachAttr(VAO.get_attr(CLR),"color");
    spr->AttachAttr(VAO.get_attr(POS),"coord");
    if (unif_minmax != -1) glUniform3f(unif_minmax, 0.f, 2.f, 0.5f);
    VAO.enable_attr(POS, 3, GL_FLOAT);
    VAO.enable_attr(NRM, 3, GL_FLOAT);
    VAO.enable_attr(CLR, 1, GL_FLOAT);
    //if (vattr != -1){
    //    glEnableVertexAttribArray(vattr);
    //    glBindBuffer(GL_ARRAY_BUFFER, VBO[POS]);
    //    glVertexAttribPointer(vattr, 3, GL_FLOAT, GL_FALSE, 0, 0);
    //}
    //if (nattr != -1){
    //    glEnableVertexAttribArray(nattr);
    //    glBindBuffer(GL_ARRAY_BUFFER, VBO[NRM]);
    //    glVertexAttribPointer(nattr, 3, GL_FLOAT, GL_FALSE, 0, 0);
    //}
    //if (cattr != -1) {
    //    glEnableVertexAttribArray(cattr);
    //    glBindBuffer(GL_ARRAY_BUFFER, VBO[CLR]);
    //    glVertexAttribPointer(cattr, 1, GL_FLOAT, GL_FALSE, 0, 0);
    //}
    VAO.release();
    //glBindVertexArray(0);
}
//------------------------------------------------------------------------------
void Axis::plot(ShaderProg * spr) {
    int Ntr = 3;
    AttachToShader(spr);
    //glBindVertexArray(VAO);
    VAO.bind();
    //glDrawElements(GL_LINES,Ntr*2, GL_UNSIGNED_INT, (void *)0);
    glDrawElementsInstanced(GL_LINES,Ntr*2, GL_UNSIGNED_INT, (void *)0,1);
    //glBindVertexArray(0);
    VAO.release();
    //glDisableVertexAttribArray(vattr);
    //glDisableVertexAttribArray(cattr);
    //glDisableVertexAttribArray(nattr);
    //glBindBuffer(GL_ARRAY_BUFFER,0);
}
//------------------------------------------------------------------------------
// SurfTemplate
//------------------------------------------------------------------------------
void SurfTemplate::plot(ShaderProg * spr) {
    int Ntr = select.size();
    attach_shader(spr); // Возможно оверкил делать это каждый раз
    //glBindVertexArray(VAO);
    VAO.bind();
    //glDrawElements(GL_TRIANGLES, Ntr, GL_UNSIGNED_INT, (void *)0);
    glDrawElementsInstanced(GL_TRIANGLES, Ntr, GL_UNSIGNED_INT, (void *)0, 1);
    //glBindVertexArray(0);
    VAO.release();
    //glDisableVertexAttribArray(vattr);
    //glDisableVertexAttribArray(cattr);
    //glDisableVertexAttribArray(nattr);
    //glBindBuffer(GL_ARRAY_BUFFER,0);
}
//------------------------------------------------------------------------------
float SurfTemplate::min(){
    return minmaxmul[0];
}
//------------------------------------------------------------------------------
float SurfTemplate::max(){
    return minmaxmul[1];
}
//------------------------------------------------------------------------------
void SurfTemplate::rangemove(float shift,bool l){
    float &min = minmaxmul[0];
    float &max = minmaxmul[1];
    float len = max - min;
    if (l) min -= shift * len;
    else max += shift * len;
}
//------------------------------------------------------------------------------
void SurfTemplate::extendrange(float factor){
    float &min = minmaxmul[0];
    float &max = minmaxmul[1];
    float &mul = minmaxmul[2];
    float len_2 = (max - min)*0.5f,
          center = (min + max)* 0.5f;
    min = center - len_2*factor;
    max = center + len_2*factor;
    mul = (min != max)? 1.f/(max  - min): 1.f;
}
//------------------------------------------------------------------------------
void SurfTemplate::refill_select(){
    int NTR = get_cells_size();
    select.clear();
    select.resize(NTR*3);
    for(unsigned int i=0; i < select.size(); i++) select[i] = i;
}
//------------------------------------------------------------------------------
