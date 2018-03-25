#include "err.h"
#include <cstdio>
#include <vector_types.h>
#include <vector>
#include <algorithm>
#include <functional>
#include <iostream>
#include "cuda_math.h"
#include "tablet.hpp"

//#define USE_DOUBLE

#ifdef USE_DOUBLE
typedef double floatT;
typedef double3 floatT3;
#define make_floatT3 make_double3
#else//use float
typedef float floatT;
typedef float3 floatT3;
#define make_floatT3 make_float3
#endif

struct TriHexagon;

struct Tablet {//Базовая структра хранения и доступа к данным таблетки
  floatT3* moments;//массив моментов
  float3* coords;//массив координат
  float2* grid_xy;
  float dt, Rtab;
  int3 Nrays;//число линеек в сетке по каждой координате
  TriHexagon* ids; // массив индексов узлов сетки в трёх координатных системах
  float3 Hext;//Внешнее поле
  float3 Aani[3];//Оси анизотропии
  float Kani, zero;//коэффициент анизотропии
  float Jexch;//интеграл обменного взаимодействия
  float alpha, gamma; //константы прецессии и диссипации
  int Nlayer3, Nmesh, Nmoment;//число троек слоёв в таблетке, а также число узлов сетки и число моментов в три-слое

  int initPentagon(int na, int kerns[], int Ndl=1);
  int initMoments();
  void set();
  void clear();
  int getNmoms() { return 3*Nlayer3*Nmoment; }
  float3 getMinBox();
  float3 getMaxBox();
};

void PrintLastError(const char *file, int line) {
  cudaError_t err=cudaGetLastError();
  if(err!=cudaSuccess) fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
}
bool CheckError(cudaError_t err, const char *file, int line) {
  if(err==cudaSuccess) return false;
  fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
  return true;
}

void deviceDiagnostics(){
  int deviceCount;
  CHECK_ERROR( cudaGetDeviceCount(&deviceCount) );  
  printf("GPU devices :: %d \n", deviceCount);
  cudaDeviceProp devProp[deviceCount];
  for(int i = 0; i < deviceCount; ++i) {
    printf("*** CUDA Device #%d ***", i);
    CHECK_ERROR( cudaGetDeviceProperties(&devProp[i], i) );
    printf("%s ***\n", devProp[i].name);
    printf("\t%d.%d compute capability\n", devProp[i].major, devProp[i].minor);
    printf("\t%d multiprocessors\n", devProp[i].multiProcessorCount);
    printf("\t%.2fGB max mem pitch of %.2fGB global memory\n", devProp[i].memPitch/(1024.*1024.*1024), devProp[i].totalGlobalMem/(1024.*1024.*1024));
    printf("\t%.2fKB total shared memory per block\n", devProp[i].sharedMemPerBlock/1024.);
    printf("\t%.2fKB total constant memory\n", devProp[i].totalConstMem/1024.);
    printf("\t%.2fK registers per block\n", devProp[i].regsPerBlock/1024.);
    printf("\t%d/%d threads per Warp/block\n", devProp[i].warpSize, devProp[i].maxThreadsPerBlock);
    printf("\tClock rate: %.2fGHz\n", devProp[i].clockRate*1e-6);
    printf("\tTexture alignment: %luB\n", devProp[i].textureAlignment);
    printf("\tConcurrent copy and execution: %s\n", (devProp[i].deviceOverlap ? "Yes" : "No"));
    printf("\tKernel execution timeout: %s\n", (devProp[i].kernelExecTimeoutEnabled ? "Yes" : "No"));
  }
}

//#include "heap.hpp"

__forceinline__ __host__ __device__ floatT3 rotateQ(floatT3 v, floatT3 H) {
	register floatT lH = length(H);
	register floatT a = 0.5*lH;
	if(a < floatT(1e-15)) return v;
        floatT3 nH = H*(floatT(1.0)/lH);
#ifdef USE_DOUBLE
        double sina, cosa; sincos(a, &sina, &cosa);
#else//use float
	#if defined(__CUDA_ARCH__)
        float sina, cosa; __sincosf(a, &sina, &cosa);
	#else
        float sina, cosa; sincosf(a, &sina, &cosa);
	#endif
#endif
	//register floatT3 u = sina*nH;
/* Quaternion formula :: (floatT(2.0)*dot(u, v))*u + (s*s-dot(u, u))*v + (floatT(2.0)*s)*cross(u,v); */
	return make_floatT3( (     nH.x*nH.x*v.x + floatT(2.0)*nH.x*nH.y*v.y + floatT(2.0)*nH.x*nH.z*v.z - v.x - nH.y*nH.y*v.x-nH.z*nH.z*v.x)*sina*sina + v.x + floatT(2.0)*cosa*sina*(nH.y*v.z-nH.z*v.y),
                       (floatT(2.0)*nH.x*nH.y*v.x +      nH.y*nH.y*v.y + floatT(2.0)*nH.y*nH.z*v.z - v.y - nH.x*nH.x*v.y-nH.z*nH.z*v.y)*sina*sina + v.y + floatT(2.0)*cosa*sina*(nH.z*v.x-nH.x*v.z),
                       (floatT(2.0)*nH.x*nH.z*v.x + floatT(2.0)*nH.y*nH.z*v.y +      nH.z*nH.z*v.z - v.z - nH.x*nH.x*v.z-nH.y*nH.y*v.z)*sina*sina + v.z + floatT(2.0)*cosa*sina*(nH.x*v.y-nH.y*v.x) );
};/* rotateQ */

__forceinline__ __host__ __device__ floatT3 rotateM(floatT3 v, floatT3 H){
	register floatT a = length(H);
	if(a < 1e-15f) return v;
	register floatT3 u = H*(floatT(1.0)/a);
#ifdef USE_DOUBLE
        double sina, cosa; sincos(a, &sina, &cosa);
#else//use float
	#if defined(__CUDA_ARCH__)
        float sina, cosa; __sincosf( a , &sina , &cosa );
	#else
        float sina, cosa; sincosf(a, &sina, &cosa);
	#endif
#endif
	register floatT ucos = 1-cosa;
	floatT rotM[3][3] = { 
		{cosa + u.x*u.x*ucos   ,	u.x*u.y*ucos - u.z*sina,	u.x*u.z*ucos + u.y*sina},
		{u.y*u.x*ucos + u.z*sina,	cosa + u.y*u.y*ucos   , 	u.y*u.z*ucos - u.x*sina},
		{u.z*u.x*ucos - u.y*sina, 	u.z*u.y*ucos + u.x*sina,	cosa + u.z*u.z*ucos   }
	};
	return make_floatT3(v.x*rotM[0][0] + v.y*rotM[0][1] + v.z*rotM[0][2],
			   v.x*rotM[1][0] + v.y*rotM[1][1] + v.z*rotM[1][2],
 			   v.x*rotM[2][0] + v.y*rotM[2][1] + v.z*rotM[2][2]);
};/* rotateM */

int rot_test(double tMax, double tDrop) {
  floatT3 H={0,1,1}, m1={1,0,0}, m2={1,0,0};
  double t=0.0, dt=1.e-1;
  while(t<tMax) {
    printf("%g\t%g\t%g\t%g\t%g\t%g\t%g\n",t, m1.x,m1.y,m1.z, m2.x,m2.y,m2.z);
    for(double tN=t+tDrop; t<tN; t+=dt) {
      floatT3 tm1=rotateM(m1,(H+0*m2)*dt);
      floatT3 tm2=rotateQ(m2,(H+0*m1)*dt);
      m1 = tm1; m2 = tm2;
    }
  }
  return m1.x+m2.x;
}

__constant__ Tablet tab;
/*
struct sortT {
  unsigned int sort; // 3 * 5 * 2bit
  __device__ __host__ inline void set(int il, int ia, int s) { sort &= ~(3<<(2*(il*5+ia))); sort |= s<<(2*(il*5+ia)); }
};*/

struct TriHexagon {
  ushort4 idx;
  __device__ __host__ inline bool present(int il, int ia) { return 1&(idx.w>>(il*5+ia)); }
  __device__ __host__ inline void set(int il, int ia) { idx.w |= 1<<(il*5+ia); }
  __device__ __host__ inline void clear(int il, int ia) { idx.w &= ~(1<<(il*5+ia)); }
  __device__ __host__ inline int get_id(int i) { return ((unsigned short*)&idx)[i]; }
  __device__ __host__ inline void set_id(int i, unsigned short v) { ((unsigned short*)&idx)[i] = v; }

  __device__ void set() { idx = tab.ids[threadIdx.x].idx; }
  void reset() { idx = make_ushort4(0,0,0,0x7FFF); }
};

__global__ void test_rotate(int Nt) {
  ushort4 id=tab.ids[threadIdx.x].idx; floatT3* pM=&tab.moments[3*blockIdx.x*tab.Nmoment];//Red
  floatT3 m[3][3], zero={0,0,0};
  for(int il=0; il<3; il++) {
    m[il][0] = pM[id.x];
    m[il][1] = (id.w&2)?pM[  tab.Nmesh+id.x]:zero;
    m[il][2] = (id.w&8)?pM[2*tab.Nmesh+id.z]:zero;
    id = make_ushort4(id.y,id.z,id.x,((id.w&31)<<10)|(id.w>>5));
    pM += 3*tab.Nmesh;
  }
#pragma unroll 1
  for(int it=0; it<Nt; it++) {
    for(int il=0; il<3; il++) {
      for(int i=0; i<3; i++) {
        floatT3 mij=m[il][i];
        floatT3 Heff=tab.Hext+tab.zero*mij;
        m[il][i] = rotateQ(mij, Heff*tab.dt);
      }
    }
  }
  id=tab.ids[threadIdx.x].idx; pM=&tab.moments[3*blockIdx.x*tab.Nmoment];//Red
  for(int il=0; il<3; il++) {
    pM[id.x] = m[il][0];
    if(id.w&2) pM[  tab.Nmesh+id.x] = m[il][1];
    if(id.w&8) pM[2*tab.Nmesh+id.z] = m[il][2];
    id = make_ushort4(id.y,id.z,id.x,((id.w&31)<<10)|(id.w>>5));
    pM += 3*tab.Nmesh;
  }
}
__global__ void test_ani(int Nt) {
  ushort4 id=tab.ids[threadIdx.x].idx; floatT3* pM=&tab.moments[3*blockIdx.x*tab.Nmoment];//Red
  floatT3 m[3][3], zero={0,0,0};
  for(int il=0; il<3; il++) {
    m[il][0] = pM[id.x];
    m[il][1] = (id.w&2)?pM[  tab.Nmesh+id.x]:zero;
    m[il][2] = (id.w&16)?pM[2*tab.Nmesh+id.z+1]:zero;
    id = make_ushort4(id.y,id.z,id.x,((id.w&31)<<10)|(id.w>>5));
    pM += 3*tab.Nmesh;
  }
#pragma unroll 1
  for(int it=0; it<Nt; it++) {
#pragma unroll 3
    for(int il=0; il<3; il++) {
#pragma unroll 3
      for(int i=0; i<3; i++) {
        const int subl=(il+i)%3;
        floatT3 mij=m[il][i];
        floatT3 Heff=tab.Hext+tab.Kani*dot(tab.Aani[subl],mij)*tab.Aani[subl];
        m[il][i] = rotateQ(mij, Heff*tab.dt);
      }
    }
  }
  id=tab.ids[threadIdx.x].idx; pM=&tab.moments[3*blockIdx.x*tab.Nmoment];//Red
  for(int il=0; il<3; il++) {
    pM[id.x] = m[il][0];
    if(id.w&2) pM[  tab.Nmesh+id.x] = m[il][1];
    if(id.w&16) pM[2*tab.Nmesh+id.z+1] = m[il][2];
    id = make_ushort4(id.y,id.z,id.x,((id.w&31)<<10)|(id.w>>5));
    pM += 3*tab.Nmesh;
  }
}
/*=======================================
	Runge-Kutta stencil 2nd order
=========================================
  0:	m1 = m0 (rotate)     h*S(m0, H0)
	m2 = m0 (rotate) 0.5*h*S(m0, H0)
	H1 = H1(m1)
h/2:	m0 = m2 (rotate) 0.5*h*S(m2, H1)
where	S(m_i, H_j) = gamma*H_j + alpha * cross(m_i, H_j)  
=======================================*/
__global__ void stencil(int Nt) {
  ushort4 id=tab.ids[threadIdx.x].idx; floatT3* pM=&tab.moments[3*blockIdx.x*tab.Nmoment];//Red
  floatT3 m[3][3], zero={0,0,0};
  for(int il=0; il<3; il++) {
    m[il][0] = pM[id.x];
    m[il][1] = (id.w&2)?pM[  tab.Nmesh+id.x]:zero;
    m[il][2] = (id.w&16)?pM[2*tab.Nmesh+id.z+1]:zero;
    id = make_ushort4(id.y,id.z,id.x,((id.w&31)<<10)|(id.w>>5));
    pM += 3*tab.Nmesh;
  }
  float3 m1, m2;
#pragma unroll 1
  for(int it=0; it<Nt; it++) {
#pragma unroll 3
    for(int il=0; il<3; il++) {
#pragma unroll 3
      for(int i=0; i<3; i++) {
	      const int subl=(il+i)%3;
        floatT3 m0=m[il][i];
        floatT3 Heff=tab.Hext+tab.Kani*dot(tab.Aani[subl],m0)*tab.Aani[subl];
        m1 = rotateQ(m0, tab.dt*(tab.gamma*Heff + tab.alpha*cross(m0,Heff)));
        m2 = 0.5*m1;
        Heff=tab.Hext+tab.Kani*dot(tab.Aani[subl],m2)*tab.Aani[subl];
        m0 = rotateQ(m2, 0.5*tab.dt*(tab.gamma*Heff + tab.alpha*cross(m2,Heff)));
        m[il][i] = normalize(m0);
      }
    }
    //__syncthreads();
  }
  id=tab.ids[threadIdx.x].idx; pM=&tab.moments[3*blockIdx.x*tab.Nmoment];//Red
  for(int il=0; il<3; il++) {
    pM[id.x] = m[il][0];
    if(id.w&2) pM[  tab.Nmesh+id.x] = m[il][1];
    if(id.w&16) pM[2*tab.Nmesh+id.z+1] = m[il][2];
    id = make_ushort4(id.y,id.z,id.x,((id.w&31)<<10)|(id.w>>5));
    pM += 3*tab.Nmesh;
  }
}
/*
__global__ void run(int Nt) {
  __shared__ floatT3 shM[2][512], shH[2][512];
  ushort4 id=tab.ids[threadIdx.x].idx; floatT3* pM=&tab.moments[3*blockIdx.x*tab.Nmoment];//Red
  floatT3 m[3][3]={pM[id.x],pM[id.x+tab.Nmoment],pM[id.x+2*tab.Nmoment]}, zero={0,0,0};
  for(int il=0; il<3; il++) {
    const int ish=il*tab.Nmoment;
    m[il][0] = pM[id.x+ish];
    m[il][1] = (id.w&2)?pM[ish+  tab.Nmesh+id.x]:zero;
    m[il][2] = (id.w&16)?pM[ish+2*tab.Nmesh+id.z+1]:zero;
    id = make_ushort4(id.y,id.z,id.x,((id.w&31)<<10)|(id.w>>5));
  }
#pragma unroll 1
  for(int it=0; it<Nt; it++) {
    floatT3 Hexch[3] = {zero,zero,zero};
#pragma unroll 3
    for(int il=0; il<3; il++) {
      shM[0][id.x  ] = m[il][1]; shH[0][id.x  ] = H[1];
      shM[1][id.z+1] = m[il][2]; shH[1][id.z+1] = H[2];
      __syncthreads();
      m[il][1] = shM[0][id.x  ]; H[1] = shH[0][id.x  ];
      m[il][2] = shM[1][id.z+1]; H[2] = shH[1][id.z+1];
#pragma unroll 3
      for(int i=0; i<3; i++) {
        const int subl=(il+i)%3;
        floatT3 mij=m[il][i];
        floatT3 Hexch = tab.Jexch*(m[il][(i+1)%3]+m[il][(i+2)%3]);
        floatT3 Heff=tab.Hext+tab.Kani*dot(tab.Aani[subl],mij)*tab.Aani[subl];
        m[il][i] = rotateQ(mij, Heff*tab.dt);
      }
      pM += 3*tab.Nmesh;
    }
    pM -= 9*tab.Nmesh;
  }
  for(int il=0; il<3; il++) {
    pM[id.x] = m[il][0];
    if(id.w&2) pM[  tab.Nmesh+id.x] = m[il][1];
    if(id.w&16) pM[2*tab.Nmesh+id.z+1] = m[il][2];
    id = make_ushort4(id.y,id.z,id.x,((id.w&31)<<10)|(id.w>>5));
    pM += tab.Nmoment;
  }
  pM[id.x]=m0[0]; pM[id.x+tab.Nmoment]=m0[1]; pM[id.x+2*tab.Nmoment]=m0[2];
}
*/
__global__ void initMoments() {
  //floatT3 mx={1.0, 0.0, 0.0}, my={0.0, 1.0, 0.0}, mz={0.0, 0.0, 1.0}, zero={0,0,0}, out={0,0,-1};
  floatT3 mx=tab.Aani[0], my=tab.Aani[1], mz=tab.Aani[2], zero={0,0,0}, out={0,0,-1};
  TriHexagon th; th.set(); int i;
  floatT3* pM=&tab.moments[3*blockIdx.x*tab.Nmoment];//Red
  float z=blockIdx.x; float2 xy=tab.grid_xy[threadIdx.x];
  const float H=0.5, A=2*H/sqrt(3.), h=A/2, a=2*H/3.;
  float3* pC=&tab.coords[3*blockIdx.x*tab.Nmoment];
  pM[th.idx.x] = mx; pC[th.idx.x] = {xy.x-h,xy.y,z};
  i=  tab.Nmesh+th.idx.x; if(th.present(0,1)) { pM[i] = my; pC[i] = {xy.x-A,xy.y-H,z}; } else { pM[i] = zero; pC[i] = out; }//{xy.x-A,xy.y-H,z-1}; }
  i=2*tab.Nmesh+th.idx.z; if(th.present(0,3)) { pM[i] = mz; pC[i] = {xy.x-A,xy.y+H,z}; } else { pM[i] = zero; pC[i] = out; }//{xy.x-A,xy.y+H,z-1}; }
  pM += tab.Nmoment; pC += tab.Nmoment;//Green
  z += 1.0/3.0;
  pM[th.idx.y] = my; pC[th.idx.y] = {xy.x,xy.y-a/2,z};
  i=  tab.Nmesh+th.idx.y; if(th.present(1,1)) { pM[i] = mz; pC[i] = {xy.x+A,xy.y-a/2  ,z}; } else { pM[i] = zero; pC[i] = out; }//{xy.x+A,xy.y-a/2  ,z-1}; }
  i=2*tab.Nmesh+th.idx.x; if(th.present(1,3)) { pM[i] = mx; pC[i] = {xy.x-h,xy.y-a/2-H,z}; } else { pM[i] = zero; pC[i] = out; }//{xy.x-h,xy.y-a/2-H,z-1}; }
  pM += tab.Nmoment;//Blue
  pC += tab.Nmoment;//Green
  z += 1.0/3.0;
  pM[th.idx.z] = mz; pC[th.idx.z] = {xy.x,xy.y+a/2,z};
  i=  tab.Nmesh+th.idx.z; if(th.present(2,1)) { pM[i] = mx; pC[i] = {xy.x-h,xy.y+a/2+H,z}; } else { pM[i] = zero; pC[i] = out; }//{xy.x-h,xy.y+a/2+H,z-1}; }
  i=2*tab.Nmesh+th.idx.y; if(th.present(2,3)) { pM[i] = my; pC[i] = {xy.x+A,xy.y+a/2  ,z}; } else { pM[i] = zero; pC[i] = out; }//{xy.x+A,xy.y+a/2  ,z-1}; }
}

// Вспомогательная структура для инициализации таблетки
struct ITH {//Симметричная относительно базовой точки структура моментов (три гексагона | три X | шесть тригонов)
  int3 ray;  // координаты базовой точки в тройной сетке (номер линии сетки)
  TriHexagon pid;
  bool use; // признак использования структуры
  void set(int3 r) { use = true; ray = r; pid.reset(); }
  int& get_ray(int i) { return ((int*)&ray)[i]; }
};
void setITH(int ic, std::vector<ITH*>& trigs, int num) {
  int ip=(ic+1)%3;//, im=(ic+2)%3;
  for(int i=0, _r=trigs[0]->get_ray(ic)-1; i<num; i++) {
    ITH& th=*trigs[i];
    th.pid.set_id(ic,i);
    int r=th.get_ray(ic);
    //for(int j=0; j<5; j++) th.pid.set(ic,j);
    if(_r != r) {
      th.pid.clear(ic,1); th.pid.clear(ip,3);
      if(i>0) { ITH& _th=*trigs[i-1]; _th.pid.clear(ic,2); _th.pid.clear(ip,4); }
    }
    _r = r;
  }
  ITH& _th=*trigs[num-1]; _th.pid.clear(ic,2); _th.pid.clear(ip,4);
}
float3 Tablet::getMinBox() { return make_float3(-Rtab,-Rtab,0.); }
float3 Tablet::getMaxBox() { return make_float3(Rtab,Rtab,Nlayer3); }
int Tablet::initPentagon(int na, int kerns[], int Ndl) {
  //Забиваем полный шестиугольник тригексагонами с номерами линий, двигаясь по спирали от центра
  int num6=3*na*(na-1)+1; // число узлов сетки в шестиугольнике со стороной na
  ITH* pgon = new ITH[num6],* p=pgon;
  int3 ray=make_int3(na-1), ray0; p->set(ray); // стартовая точка спирали --- центр шестиугольника
  for(int ia=1; ia<na; ia++) {
    ray.x--; ray.y++;
    for(int i=0; i<ia; i++) { p++; p->set(ray); ray.y--; ray.z++; }
    for(int i=0; i<ia; i++) { p++; p->set(ray); ray.y--; ray.x++; }
    for(int i=0; i<ia; i++) { p++; p->set(ray); ray.z--; ray.x++; }
    for(int i=0; i<ia; i++) { p++; p->set(ray); ray.y++; ray.z--; }
    for(int i=0; i<ia; i++) { p++; p->set(ray); ray.y++; ray.x--; }
    for(int i=0; i<ia; i++) { p++; p->set(ray); ray.z++; ray.x--; }
  }
  //Удаляем уголки и считаем сколько тригексагонов осталось
  int num=num6, nal=na-1, jNN=num6-1;
  for(int i=0; i<6*Ndl; i++) {
    int jM=-((kerns[i]+1)/2), jP=kerns[i]+jM, jN=jNN-(i%6)*nal;
    for(int j=0; j<jP; j++) pgon[jN-j].use = false;
    if(i%6==0) jN -= 6*nal;
    for(int j=jM; j<0; j++) pgon[jN-j].use = false;
    if((i+1)%6==0) { jNN -= 6*nal; nal--; }
    num -= kerns[i];
  }
  Nmesh = num;
  //Создаём вектор тригексагонов для последующей сортировки
  p = pgon;
  std::vector<ITH*> trigs;
  for(int i=0; i<num6; i++) {
    if(p->use) trigs.push_back(p);
    p++;
  }
  printf("#Nmesh: %d(theor) =?= %d(real) of %d in hexagon\n", num, int(trigs.size()), num6);
  //Сортируем тригексагоны по координатам (номер линии, тригексагон в линии) по каждой из осей
  std::sort(trigs.begin(), trigs.end(), [](ITH* a, ITH* b) { if(b->ray.x == a->ray.x) return b->ray.y < a->ray.y; return b->ray.x > a->ray.x; });
  setITH(0, trigs, num); ray0.x = trigs[0]->ray.x; Nrays.x = trigs[num-1]->ray.x-ray0.x+1;
  std::sort(trigs.begin(), trigs.end(), [](ITH* a, ITH* b) { if(b->ray.y == a->ray.y) return b->ray.z < a->ray.z; return b->ray.y > a->ray.y; });
  setITH(1, trigs, num); ray0.y = trigs[0]->ray.y; Nrays.y = trigs[num-1]->ray.y-ray0.y+1;
  std::sort(trigs.begin(), trigs.end(), [](ITH* a, ITH* b) { if(b->ray.z == a->ray.z) return b->ray.x < a->ray.x; return b->ray.z > a->ray.z; });
  setITH(2, trigs, num); ray0.z = trigs[0]->ray.z; Nrays.z = trigs[num-1]->ray.z-ray0.z+1;
  char fn[128];
  sprintf(fn, "tab%d.dat",na);
  FILE* fd=fopen(fn, "w");
  fprintf(fd, "#Nrays: %d,%d,%d;\tray0: %d,%d,%d\n", Nrays.x, Nrays.y, Nrays.z, ray0.x, ray0.y, ray0.z);
  p = pgon;
  //Создаём массив индексов и Выводим диагностику
  TriHexagon* pids=new TriHexagon[num];
  float2* xy=new float2[num];
  Rtab = sqrt(Nmesh)*5.0/8.0+1;
  for(int i=0; i<num; i++) {
    while(!p->use) p++;
    pids[i].idx = p->pid.idx;
    xy[i] = make_float2((p->ray.z-p->ray.x)/sqrt(3),na-1.0-p->ray.y); 
    fprintf(fd, "%d\t%.4g\t%.4g\t%d\t%d\t%d\t%x\n", i, xy[i].x,xy[i].y, p->pid.idx.x,p->pid.idx.y,p->pid.idx.z,p->pid.idx.w);
    p++;
  }
  Nmoment = 3*Nmesh;//-2*(Nrays.x+Nrays.y+Nrays.z);
  fprintf(fd, "# %d Moments\n", Nmoment);
  size_t sz=Nmesh*sizeof(TriHexagon);
  if(CHECK_ERROR(cudaMalloc((void**) &ids, sz))) throw(-1);
  if(CHECK_ERROR(cudaMemcpy(ids, pids, sz, cudaMemcpyHostToDevice))) throw(-1);
  sz = num*sizeof(float2);
  if(CHECK_ERROR(cudaMalloc((void**) &grid_xy, sz))) throw(-1);
  if(CHECK_ERROR(cudaMemcpy(grid_xy, xy, sz, cudaMemcpyHostToDevice))) throw(-1);
  fclose(fd);
  delete[] pids;
  delete[] pgon;
  delete[] xy;
  return num;
}
void Tablet::set() {
  size_t sz=3*Nlayer3*Nmoment*sizeof(floatT3);
  printf("#Moments memory: %g M\n", sz*1e-6);
  if(CHECK_ERROR(cudaMalloc((void**) &moments, sz))) throw(-1);
  if(CHECK_ERROR(cudaMalloc((void**) &coords, sz))) throw(-1);
}
int Tablet::initMoments() {
  Hext = make_float3(1,0,0);
  Kani = 0.002;
  Jexch=-1.0;
  alpha = 0.1;
  gamma = 1.0;
  Aani[0] = make_float3(     0.     ,   -sqrt(2./3.),sqrt(1./3.));
  Aani[1] = make_float3( sqrt(1./2.),0.5*sqrt(2./3.),sqrt(1./3.));
  Aani[2] = make_float3(-sqrt(1./2.),0.5*sqrt(2./3.),sqrt(1./3.));
  dt =0.1;
  if(CHECK_ERROR(cudaMemcpyToSymbol(tab, this, sizeof(Tablet)))) throw(-1);
  ::initMoments<<<Nlayer3,Nmesh>>>();
  return 0;
}
//===========================
int TabletIF::run(int Nt) {
  cudaTimer tm; tm.start();
  //test_rotate<<<tablet->Nlayer3,tablet->Nmesh>>>(Nt);
  //test_ani<<<tablet->Nlayer3,tablet->Nmesh>>>(Nt);
  stencil<<<tablet->Nlayer3,tablet->Nmesh>>>(Nt);
  //::run<<<tablet->Nlayer3,tablet->Nmesh>>>(Nt);
  double t=tm.stop()*1e-3; if(t==0.0) t=1;
  printf("#Test Time: %g sec, rate=%g Gmom/sec\n", t, 1e-9*Nt*Nmoms/t);
  return 1;
}
void Tablet::clear() {
  if(CHECK_ERROR(cudaFree(ids))) throw(-1);
  if(CHECK_ERROR(cudaFree(coords))) throw(-1);
  if(CHECK_ERROR(cudaFree(moments))) throw(-1);
}
void TabletIF::clear() {
  delete tablet;
}
int TabletIF::set() {
  tablet = new Tablet();
  int rs=0;
  //{int kerns[]={0,0,0,0,0,0}; rs=tablet->initPentagon(2, kerns);}
  //{int kerns[]={0,1,1,1,1,1}; rs=tablet->initPentagon(4, kerns);}
  //{int kerns[]={0,3,3,3,3,3}; rs=tablet->initPentagon(10, kerns);}
  //{int kerns[]={10,9,10,9,10,9, 3,3,3,3,3,3}; rs=tablet->initPentagon(11, kerns, 2);}//256
  {int kerns[]={5,5,5,5,5,5, 0,1,1,1,1,1}; rs=tablet->initPentagon(14, kerns, 2);}//512
  //{int kerns[]={0,1,0,1,0,1}; rs=tablet->initPentagon(19, kerns);}//1024
  //{int kerns[]={13,13,13,13,13,13, 5,7,5,7,5,7, 0,1,0,1,0,1}; rs=tablet->initPentagon(20, kerns,3);}//1024
  //{int kerns[]={13,13,13,13,13,13, 5,6,5,6,5,6, 1,1,1,1,1,1}; rs=tablet->initPentagon(20, kerns,3);}//1024
  tablet->Nlayer3 = 30;
  Nmoms = 3*tablet->Nlayer3*tablet->Nmoment;
  printf("%d moms in layer / %d total\n", rs, Nmoms);
  return Nmoms;
}
int TabletIF::init(float* m, float* c) {
  if(c) tablet->coords = (float3*)c;
  if(m) tablet->moments = (floatT3*)m;
  int rs=0;
  rs=tablet->initMoments();
  //rs=rot_test(1e7,1e4);
  printf("Nmoms %d\n", Nmoms);
  MinBox[0]=MinBox[1]=-tablet->Rtab; MinBox[2]=0;
  MaxBox[0]=MaxBox[1]= tablet->Rtab; MaxBox[2]=1*tablet->Nlayer3;
  return rs;
}

int main(int argc, char *argv[]){
  int rs=0;
  cudaTimer tm; tm.start();
  TabletIF tab;
  tab.set();
  tab.tablet->set();
  tab.init();
  printf("#Init Time: %g sec, res=%d\n", tm.stop()*1e-3, rs);
  tm.start();
  int Nloop=10, Nt=1000;
  for(int it=0; it<Nloop; it++) {
    rs=tab.run(Nt);
    //run<<<tab.Nlayer3,tab.Nmesh>>>(Nt);
  }
  double t=tm.stop()*1e-3; if(t==0.0) t=1;
  printf("#Test Time: %g sec, rate=%g Gmom/sec\n", t, 1e-9*Nt*Nloop*tab.Nmoms/t);
  tab.tablet->clear();
  tab.clear();
  return rs;
};
