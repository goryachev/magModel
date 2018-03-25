//#define GL_GLEXT_PROTOTYPES
#include "AbstractViewer/plottable.hpp"
//#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include "err.h"
#include "tablet_interface.hpp"
#define MOM 1
#define POS 0
//typedef cudaGraphicsResource_t cudaGraphicsResource ;
TabletInterface::~TabletInterface(){
	//if(res_pos && CHECK_ERROR(cudaGraphicsUnmapResources(1, (cudaGraphicsResource_t * ) &res_pos, NULL))) throw(-1);
	if(res_pos && CHECK_ERROR(cudaGraphicsUnregisterResource(*(cudaGraphicsResource_t  *) res_pos ))) throw(-1);
	if(res_mom && CHECK_ERROR(cudaGraphicsUnregisterResource(*(cudaGraphicsResource_t  *) res_mom ))) throw(-1);
	//if(CHECK_ERROR(cudaGraphicsUnmapResources(1, &res_mom, NULL))) throw(-1);
};
TabletInterface::TabletInterface(): Plottable(), radius(1.){ 
	VAO.add_buffer();
	VAO.add_buffer();
	//size_t size;
	//if(CHECK_ERROR(cudaGraphicsMapResources(1, &res_mom, NULL))) throw(-1);
}
void TabletInterface::map_resources(){
	//int Npt=TabletIF::Nmoms;
	//VAO.load_data(MOM,  sizeof( float ) *3* Npt, NULL);
	//VAO.bind();
	size_t size;
	//float * coords ;
	if(CHECK_ERROR(cudaGraphicsMapResources(1, (cudaGraphicsResource **) &res_pos, NULL))) throw(-1);
	if(CHECK_ERROR(cudaGraphicsResourceGetMappedPointer( (void**) &coords, &size, (cudaGraphicsResource *) res_pos))) throw(-1);
	//float * moments;
	if(CHECK_ERROR(cudaGraphicsMapResources(1, (cudaGraphicsResource **) &res_mom, NULL))) throw(-1);
	if(CHECK_ERROR(cudaGraphicsResourceGetMappedPointer( (void**) &moments , &size, (cudaGraphicsResource *) res_mom))) throw(-1);
}
void TabletInterface::unmap_resources(){
	if(CHECK_ERROR(cudaGraphicsUnmapResources(1, (cudaGraphicsResource **) &res_pos, NULL))) throw(-1);
	if(CHECK_ERROR(cudaGraphicsUnmapResources(1, (cudaGraphicsResource **) &res_mom, NULL))) throw(-1);
	// анмапить res_pos  не обязательно
}
void TabletInterface::load_on_device(){
	//Выделяем память на девайсе
	int Npt=TabletIF::set();//552960; //TabletIF::Nmoms; // ВРЕМЕННЫЙ КОСТЫЛЬ!
	printf("%d\n", Npt);
	VAO.load_data(POS,sizeof( float ) *3* Npt, NULL); // Выделяем память под вершины.
	VAO.bind(); // Хак, после load_data буфер на вершине стека - POS
	//glBindBuffer(GL_ARRAY_BUFFER, VAO.get_BO(POS));
	printf("BO %d\n", VAO.get_BO(POS));
	//VAO.bind()
	if(CHECK_ERROR(cudaGraphicsGLRegisterBuffer(( cudaGraphicsResource ** ) &res_pos, VAO.get_BO(POS), cudaGraphicsMapFlagsNone))) throw(-1);
	//if(CHECK_ERROR(cudaGraphicsMapResources(1, (cudaGraphicsResource **)&res_pos, NULL))) throw(-1);
	VAO.load_data(MOM,sizeof( float ) *3* Npt,NULL); // Выделяем  память под моменты.
	VAO.bind(); 
	if(CHECK_ERROR(cudaGraphicsGLRegisterBuffer((cudaGraphicsResource **) &res_mom, VAO.get_BO(MOM), cudaGraphicsMapFlagsNone))) throw(-1);
	VAO.release();
	reload_select();
	map_resources();
	TabletIF::init(moments, coords);
	unmap_resources();
}
//void reload_normals(){
//    int Npt=TabletIF::Nmoms();
//    VAO.load_data(MOM,  sizeof( floatT3 ) * Npt, NULL);
//    VAO.bind();
//	if(CHECK_ERROR(cudaGraphicsGLRegisterBuffer(&, bufferObj, cudaGraphicsMapFlagsNone))) throw(-1);
//	//if(CHECK_ERROR(cudaGraphicsResourceGetMappedPointer((void**) &moments, &size, resource))) throw(-1);
//    //glBindVertexArray(VAO);
//    //glBindBuffer(GL_ARRAY_BUFFER, VBO[MOM]);
//    //glBufferData(GL_ARRAY_BUFFER, sizeof( aiv::vctr<3, float> ) * Npt,
//    //        reinterpret_cast<float *>(& (normals[0]) ), GL_STATIC_DRAW);
//    //glBindVertexArray(0);
//}
void TabletInterface::reload_select(){
	//glBindVertexArray(VAO);
	VAO.load_indices(select.size()*sizeof(unsigned int),
			reinterpret_cast<unsigned int *>( & (select[0]) ));
	//glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, VBO[IND]);
	//glBufferData(GL_ELEMENT_ARRAY_BUFFER, select.size()*sizeof(unsigned int),
	//        reinterpret_cast<unsigned int *>( & (select[0]) ), GL_STATIC_DRAW );
	//glBindVertexArray(0);
}
void TabletInterface::set_select(int * ids, int len){
	select.clear();
	select.resize(len);
	for(unsigned int i =0; i< select.size(); i++) select[i] = ids[i];
	reload_select();
}
void TabletInterface::select_all(){
	select.resize(TabletIF::Nmoms);
	//select.resize(std::min(MViewFormat::size(),1000000));
	//WOUT(select.size());
	for(unsigned int i=0; i< select.size(); i++){
		select[i] = i;
	}
	reload_select();
}
void TabletInterface::get_auto_box(aiv::vctr<3, float> & bb_min, aiv::vctr<3, float> & bb_max){ 
	float* bmin = this->TabletIF::MinBox;
	float* bmax = this->TabletIF::MaxBox;
	bb_min = Vctf(bmin[0],bmin[1], bmin[2]);
	bb_max = Vctf(bmax[0],bmax[1], bmax[2]);
	//if (!select.empty()){
	//    bb_min = MViewFormat::get_coord(select[0]);
	//    bb_max = MViewFormat::get_coord(select[0]);
	//    for(auto i:select){
	//        auto v = MViewFormat::get_coord(i);
	//        bb_min <<= v;
	//        bb_max >>= v;
	//    }
	//    bb_min += aiv::vctr<3, float>(-radius);
	//    bb_max += aiv::vctr<3, float>(radius);
	//    if (bb_min ==bb_max){
	//        bb_min += aiv::vctr<3, float>(-1.f);
	//        bb_max += aiv::vctr<3, float>(1.f);
	//    }
	//} else{
	//    bb_min = aiv::vctr<3, float>(-radius);
	//    bb_max = aiv::vctr<3, float>(radius);
	//    if (bb_min ==bb_max){
	//        bb_min += aiv::vctr<3, float>(-1.f);
	//        bb_max += aiv::vctr<3, float>(1.f);
	//    }
	//}
}
void TabletInterface::attach_shader(ShaderProg * spr) {
	//glBindVertexArray(VAO);
	VAO.bind();
	spr->AttachUniform(unif_radius, "radius");
	spr->AttachAttr(VAO.get_attr(POS), "coord");
	spr->AttachAttr(VAO.get_attr(MOM), "normal");
	VAO.enable_attr(POS,3, GL_FLOAT);
	VAO.enable_attr(MOM, 3, GL_FLOAT);
	//if (vattr != -1){
	//    glEnableVertexAttribArray(vattr);
	//    glBindBuffer(GL_ARRAY_BUFFER, VBO[POS]);
	//    glVertexAttribPointer(vattr, 3, GL_FLOAT, GL_FALSE, 0, 0);
	//}
	//if (nattr != -1){
	//    glEnableVertexAttribArray(nattr);
	//    glBindBuffer(GL_ARRAY_BUFFER, VBO[MOM]);
	//    glVertexAttribPointer(nattr, 3, GL_FLOAT, GL_FALSE, 0, 0);
	//}
	if (unif_radius != -1) glUniform1f(unif_radius, radius);
	//glBindVertexArray(0);
	VAO.release();
}
void TabletInterface::plot(ShaderProg * spr){
	int Npt = select.size();
	attach_shader(spr);
	//glBindVertexArray(VAO);
	VAO.bind();
	glDrawElementsInstanced(GL_POINTS, Npt, GL_UNSIGNED_INT, (void *)0, 1);
	VAO.release();
	//glBindVertexArray(0);
}

