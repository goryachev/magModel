#ifndef HEAP_H_
#define HEAP_H_
#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <unistd.h>
#include <stdlib.h>
#include "cuda_math.h"

#define pow(x) x*x
#define pow2(x) x*x

bool checkErr(cudaError_t cuerr, std::string& s)
{
  if (cuerr!=cudaSuccess){
      printf("%s :: %s\n", s.c_str(), cudaGetErrorString(cuerr) );
      return true;
  };
  return false;
};

bool checkErr(cudaError_t cuerr){
  if (cuerr!=cudaSuccess){
      printf("ERROR :: %s\n", cudaGetErrorString(cuerr) );
      return true;
  };
  return false;
};

struct cuClock {
  cudaEvent_t start, stop;
  float totalTime;
  std::string log;
void init (){
  totalTime = 0;
  log = "Create CUDA start point";
  checkErr(cudaEventCreate(&start), log);
  log = "Create CUDA stop point";
  checkErr(cudaEventCreate(&stop), log);
} /* init() */
void clean() {
  log = "Destroy CUDA start point";
  checkErr(cudaEventDestroy(start), log);
  log = "Destroy CUDA stop point";
  checkErr(cudaEventDestroy(stop), log);
}; /* clean() */
float tick() {
  float t = totalTime;
  init();
  log = "Record CUDA start point";
  checkErr(cudaEventRecord(start, 0), log); 
  return t;
}; /* tick() */
float tock() {
  log = "Record CUDA stop point";
  checkErr(cudaEventRecord(stop, 0), log);
  log = "Synchronize CUDA events";
  checkErr(cudaEventSynchronize(stop), log);
  log = "Calculate Elapsed time";
  checkErr(cudaEventElapsedTime(&totalTime, start, stop), log);
  clean();
  return tick();
}; /* tock() */
}; /* struct cuClock */

__forceinline__ __host__ __device__ float3 rotateQ(float3 v, float3 H){
	register float lH = length(H);
	register float a = 0.5*lH;
	if(a < 1e-15f) return v;
        float3 nH = H*(1.0f/lH);
	#if defined(__CUDA_ARCH__)
        float sina, cosa; __sincosf(a, &sina, &cosa);
	#else
        float sina, cosa; sincosf(a, &sina, &cosa);
	#endif
	register float cos2a  = cosa*cosa;
	register float sin2a  = sina*sina;
	register float sincos = sina*cosa;
//	register float3 u = sina*nH;
//	register float  s = cosa;
/* Quaternion formula :: (2.0f*dot(u, v))*u + (s*s-dot(u, u))*v + (2.0f*s)*cross(u,v); */
	return make_float3(
(     nH.x*nH.x*v.x + 2.0f*nH.x*nH.y*v.y + 2.0f*nH.x*nH.z*v.z - nH.y*nH.y*v.x-nH.z*nH.z*v.x)*sin2a + cos2a*v.x + 2.0f*sincos*(nH.y*v.z-nH.z*v.y),
(2.0f*nH.y*nH.x*v.x +      nH.y*nH.y*v.y + 2.0f*nH.y*nH.z*v.z - nH.x*nH.x*v.y-nH.z*nH.z*v.y)*sin2a + cos2a*v.y + 2.0f*sincos*(nH.z*v.x-nH.x*v.z),
(2.0f*nH.z*nH.x*v.x + 2.0f*nH.z*nH.y*v.y +      nH.z*nH.z*v.z - nH.x*nH.x*v.z-nH.y*nH.y*v.z)*sin2a + cos2a*v.z + 2.0f*sincos*(nH.x*v.y-nH.y*v.x)
	);
};/* rotateQ */

__forceinline__ __host__ __device__ float3 rotateM(float3 v, float3 H){
	register float a = length(H);
	if(a < 1e-15f) return v;
	register float3 u = H*(1.0f/a);
	#if defined(__CUDA_ARCH__)
        float sina, cosa; __sincosf( a , &sina , &cosa );
	#else
        float sina, cosa; sincosf(a, &sina, &cosa);
	#endif
	register float ucos = 1-cosa;
	float rotM[3][3] = { 
		{cosa + pow(u.x)*ucos   ,	u.x*u.y*ucos - u.z*sina,	u.x*u.z*ucos + u.y*sina},
		{u.y*u.x*ucos + u.z*sina,	cosa + pow(u.y)*ucos   , 	u.y*u.z*ucos - u.x*sina},
		{u.z*u.x*ucos - u.y*sina, 	u.z*u.y*ucos + u.x*sina,	cosa + pow(u.z)*ucos   }
	};
	return make_float3(v.x*rotM[0][0] + v.y*rotM[0][1] + v.z*rotM[0][2],
			   v.x*rotM[1][0] + v.y*rotM[1][1] + v.z*rotM[1][2],
 			   v.x*rotM[2][0] + v.y*rotM[2][1] + v.z*rotM[2][2]);
};/* rotateM */

__forceinline__ __host__ __device__ float3 taylor(float3  v, float3 H )
{
	return v - cross(v, H) + 0.5*cross(cross(v, H ), H) - (1.f/6.f)*cross(cross(cross(v, H), H ), H);// + 1.f/24.f*cross(cross(cross(cross(v, H), H), H) ,H);
};/* taylor */

void deviceDiagnostics()
{
  int deviceCount;
  checkErr( cudaGetDeviceCount(&deviceCount) );  
  printf("GPU devices :: %d \n", deviceCount);
  cudaDeviceProp devProp[deviceCount];
  for(int i = 0; i < deviceCount; ++i) {
    printf("*********** CUDA Device #%d ***********\n", i);
    checkErr( cudaGetDeviceProperties(&devProp[i], i) );
    printf("Name: %s\n", devProp[i].name);
    printf("Major revision number: %d\n", devProp[i].major);
    printf("Minor revision number: %d\n", devProp[i].minor);
    printf("Number of multiprocessors: %d\n", devProp[i].multiProcessorCount);
    printf("Number of cores: %d\n", devProp[i].multiProcessorCount);
    printf("Total global memory: %zu\n", devProp[i].totalGlobalMem);
    printf("Total shared memory per block: %zu\n", devProp[i].sharedMemPerBlock);
    printf("Total registers per block: %d\n", devProp[i].regsPerBlock);
    printf("Warp size: %d\n", devProp[i].warpSize);
    printf("Maximum memory pitch: %lu\n", devProp[i].memPitch);
    printf("Maximum threads per block: %d\n", devProp[i].maxThreadsPerBlock);
    printf("Clock rate: %d\n", devProp[i].clockRate);
    printf("Total constant memory: %lu\n", devProp[i].totalConstMem);
    printf("Texture alignment: %lu\n", devProp[i].textureAlignment);
    printf("Concurrent copy and execution: %s\n", (devProp[i].deviceOverlap ? "Yes" : "No"));
    printf("Kernel execution timeout: %s\n", (devProp[i].kernelExecTimeoutEnabled ? "Yes" : "No"));
  }
}

bool analysisCommandLine(int argc, char *argv[])//, utilParameters& uP)
{
  for(int id = 1; id < argc; id ++){ 
	if(!strcmp(argv[id], "-d" )) deviceDiagnostics();
	if(!strcmp(argv[id], "-d!")){deviceDiagnostics(); return true;}
//	if(!strcmp(argv[id], "-d=")) uP.NumDev = atoi(argv[id+1]);
//	if((!strcmp(argv[id],"--test"))||(!strcmp(argv[id],"-t"))) uP.test_mode = true;
  };
  return false;
};

void checkMem() {
	size_t free, total;
 	checkErr( cudaMemGetInfo(&free, &total) );
	printf("Total GPU Mem Size :: %16zu\t(%8ld MB)\n", total, total/1024/1024);
	printf("Busy  GPU Mem Size :: %16zu\t(%8ld MB)\n", total-free, (total-free)/1024/1024);
	printf("Free  GPU Mem Size :: %16zu\t(%8ld MB)\n", free, free/1024/1024);
}
#endif /* HEAP_H_ */
