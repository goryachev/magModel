/*
Author :: Goryachev Ivan
E-mail :: kvant87@gmail.com
Compile:: nvcc -ccbin g++ -O3 -DDATA_VECTOR_SZ=3 -Xptxas="-v" -gencode arch=compute_50,code=\"sm_50,compute_50\" rot.cu -o rot
*/

#include "cuda_math.h"
#include <cstdio>
#include <ctime>
#include <cstdlib>
#include <string>
#include "heap.hpp"

//const int Nw = 1;//32;		// warp size
const int Nv = 1;//32*Nw;		// CUDA block size
//const int NS = 1;
const int Nsm = 1;//14;		// Amount of stream processors
const int dptMax = 2;//1<<12;	// Maximum data per thread 
const int N = Nsm * dptMax;
//const short unsigned int Nat=2;
const long int iR = 100;

//#include "struct.h"
typedef float3 float3NT[Nv];
struct rot_pars {
   float3NT* rot_vec;
   float3NT* H;
   float3NT* res;
}; /* struct rot_pars */
__constant__ rot_pars rotDev; 

__global__ void __launch_bounds__(Nv,1) rotM(const int dpt){
	register float3 v; 
	register float3 H; 
#pragma unroll 1
	for (int i=blockIdx.x*dpt; i<(blockIdx.x+1)*dpt; i++){
	  v = rotDev.rot_vec[i][threadIdx.x];
          H = rotDev.H[i][threadIdx.x];
//          for(int ir=0; ir<iR; ir++) rotDev.res[i][threadIdx.x]+= rotateM(v , H);
          for(int ir=0; ir<iR; ir++) v += rotateM(v+ir , H+ir);
	  rotDev.res[i][threadIdx.x]+= v;
	}
};/* rotM */

__global__ void __launch_bounds__(Nv,1) rotQ(const int dpt){
	register float3 v; 
	register float3 H; 
#pragma unroll 1
	for (int i=blockIdx.x*dpt; i<(blockIdx.x+1)*dpt; i++){
	  v = rotDev.rot_vec[i][threadIdx.x];
          H = rotDev.H[i][threadIdx.x];
          for(int ir=0; ir<iR; ir++) v += rotateQ(v+ir , H+ir);
	  rotDev.res[i][threadIdx.x]+= v;
	}
};/* rotQ */

struct rot_pars_host: public rot_pars{
  void reset(){ 
    if(	
		checkErr( cudaMalloc((void**) (&rot_vec), N*sizeof(float3NT)) )	||
		checkErr( cudaMalloc((void**) (&H), N*sizeof(float3NT)) )	||
		checkErr( cudaMalloc((void**) (&res), N*sizeof(float3NT)) )
      ) exit(EXIT_FAILURE) ;  
  };
  void clean(){
    if(
	checkErr( cudaFree(rot_vec) )	|| 
	checkErr( cudaFree(H) )		|| 
	checkErr( cudaFree(res) )	 
      ) exit(EXIT_FAILURE) ;  
  };
} rotHost;

__global__ void __launch_bounds__(1024,1) initDev(){
#pragma unroll 1
	for (int i=blockIdx.x*dptMax; i<(blockIdx.x+1)*dptMax; i++){
	  rotDev.rot_vec[i][threadIdx.x] = make_float3(1.f, 2.f, 1.f);
          rotDev.H[i][threadIdx.x] = make_float3(2.f, 1.f, 2.f);
          rotDev.res[i][threadIdx.x] = make_float3(0.f, 0.f, 0.f);
	}
};

#include "FCC.h"
#include "stencil.cuh"
#define init(x, v) for (int i=0; i<N; i++){ x[i] = v; }

int main(int argc, char *argv[]){
if(analysisCommandLine(argc, argv)) return 0;
/*************************
	HOST EXECUTION
**************************/
/*
  clock_t timer;
  float3* v0  = new float3[N*Nv];
  float3* H   = new float3[N*Nv];
  float3* res = new float3[N*Nv];
//-------------------------------------------------------------
  init (v0, make_float3(1.f, 1.f, 1.f));
  init(H, make_float3(2.f, 1.f, 2.f));
  for (int dpt = Nv; dpt <= N*Nv ; dpt+=Nv){
	timer = clock();
	for (int i=0; i<dpt; i++) res[i] = rotateM(v0[i], H[i]);
  	timer = clock() - timer;
  	printf( "%d\t%lf\n", dpt, ((float) timer / CLOCKS_PER_SEC) );
  }
//  	printf("%f, %f, %f\n\n", res[0].x, res[0].y, res[0].z);
  printf("\n");
//-------------------------------------------------------------
  init (v0, make_float3(1.f, 1.f, 1.f));
  init(H, make_float3(2.f, 1.f, 2.f));
  for (int dpt = Nv; dpt <= N*Nv ; dpt+=Nv){
	timer = clock();
	for (int i=0; i<dpt; i++) res[i] = rotateQ(v0[i], H[i]);
	timer = clock() - timer;
  	printf( "%d\t%lf\n", dpt, ((float) timer / CLOCKS_PER_SEC) );
  }
//  printf("%f, %f, %f\n\n", res[0].x, res[0].y, res[0].z);
*/
/*************************
	GPUs EXECUTION
*************************/
#define kernel(str, krn) 											\
  printf(str);													\
  printf("\n");													\
  for(long int dpt = 1; dpt<dptMax; dpt*=2 ){									\
        krn<<<Nsm, Nv>>>(dpt); 											\
	cudaDeviceSynchronize();										\
	tt = tm.tock();												\
	printf("N :: %ld\tGPU time :: %10.2f  |  %10g GLU/sec\n", Nsm*dpt*Nv, tt,  Nsm*dpt*Nv*iR/tt*1e-6 );	\
  }														\
  printf("\n");				
  checkErr( cudaSetDevice(0) );	
  rotHost.reset();		
  cudaMemcpyToSymbol(rotDev, &rotHost, sizeof(rotDev)); 
  cuClock tm; float tt;
  normRand Htherm; Htherm.initRand(); 
  init_pos();
  initDev<<<Nsm, Nv>>>(); 
  cudaDeviceSynchronize();
  checkMem();
  cudaDeviceSynchronize();
  tm.tick();


  kernel (">>> quarternion rotate calculation",	rotQ );
  kernel (">>> matrix rotate calculation", rotM );
  kernel (">>> stencil with Hext", stHext );
  kernel (">>> stencil with Hani", stHani );
  printf (">>> stencil with Htherm\n");													
  for(long int dpt = 1; dpt<dptMax; dpt*=2 ){									
        stHtherm<<<Nsm, Nv>>>(dpt, Htherm); 										
	cudaDeviceSynchronize();										
	tt = tm.tock();												
	printf("N :: %ld\tGPU time :: %10.2f  |  %10g GLU/sec\n", Nsm*dpt*Nv, tt,  Nsm*dpt*Nv*iR/tt*1e-6 );	
  }														
  printf("\n");				
  kernel (">>> stencil with Hexch", stHexch);

  for(long int dpt = 1; dpt<dptMax; dpt*=2 ){									
        stHall<<<Nsm, Nv>>>(dpt, Htherm); 										
	cudaDeviceSynchronize();										
	tt = tm.tock();												
	printf("N :: %ld\tGPU time :: %10.2f  |  %10g GLU/sec\n", Nsm*dpt*Nv, tt,  Nsm*dpt*Nv*iR/tt*1e-6 );	
  }														
  printf("\n");				
  tm.clean(); Htherm.clean();
  return 0;
}
