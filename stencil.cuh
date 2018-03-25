/*
	Runge-Kutta 4 order scheme
*/
#ifndef STENCIL_H_
#define STENCIL_H_
const float alpha = 0.1;
const float J = 0.1;
const float h = 0.001;
const float Th = 0.1;
const float Hx = 1.f;
const float Hy = 1.f;
const float Hz = 1.f;

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { printf("Error at %s:%d\n",__FILE__,__LINE__); throw -1;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { printf("Error at %s:%d\n",__FILE__,__LINE__); throw -1;}} while(0)
#include <curand.h>
#include <curand_kernel.h>
#define rotate rotateQ

__global__ void init_rand(curandState *states){
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  curand_init(blockIdx.x, threadIdx.x, 0, &states[tid]); 
  //curand_init(1234, tid, 0, &states[tid]); 
};

struct normRand{
  curandState *states;
  float Th;
  void initRand(){
    CUDA_CALL(cudaMalloc((void **)&states, Nsm*Nv*sizeof(curandState)));
    CUDA_CALL(cudaThreadSynchronize());
    init_rand<<< Nsm , Nv>>>(states);
    CUDA_CALL(cudaThreadSynchronize());
  }
  float3 __device__ __forceinline__ get_rand() {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    return make_float3(curand_normal(&states[tid]),
		       curand_normal(&states[tid]),
	  	       curand_normal(&states[tid]));
  }
  void clean(){
   CUDA_CALL(cudaFree(states));
  };
};

  float3 __device__ __forceinline__ get_rand(curandState *state) {
    float3 res=make_float3(curand_normal(state), curand_normal(state), curand_normal(state));
    return res;
  }

/*****************************
	Simple stencil 
*****************************/
__global__ void __launch_bounds__(Nv, 1+(Nv<=128)) stHext(const int dpt){
  float3 Hext = make_float3(Hx, Hy, Hz);
  float3 regM0, regM, regMres, H;
#pragma unroll 1
  for (int i=blockIdx.x*dpt; i<(blockIdx.x+1)*dpt; i++){
    regMres = make_float3(0.f); 
    regM0 = rotDev.rot_vec[i][threadIdx.x]; /* copy global -> register */
    float RKcoeff[4] = {1./6., 1./3., 1./3., 1./6.};
    float step[4] = {1./2., 1./2., 1., 1. };
#pragma unroll 1
    for(int ir=0; ir<iR; ir++){
      regM = regM0;
#pragma unroll 4
      for(int stRK=0; stRK<4; stRK++) {
        __syncthreads();
        H = Hext; 
          H = h*(H + alpha * cross(regM, H));
          regM = rotate(regM0, step[stRK]*H);
          if(stRK<2) regMres += RKcoeff[stRK]*rotate(regM0, H);
          else       regMres += RKcoeff[stRK]*regM;
      }
      regMres = normalize(regMres);
    }
    rotDev.rot_vec[i][threadIdx.x] = regMres; /* copy register -> global */
  }
}
/*****************************
	Stencil with Htherm term 
*****************************/
__global__ void __launch_bounds__(Nv, 1+(Nv<=128)) stHtherm(const int dpt, normRand Hth){
  float3 Hext = make_float3(Hx, Hy, Hz);
  float3 regM0, regM, regMres, H;
#pragma unroll 1
  for (int i=blockIdx.x*dpt; i<(blockIdx.x+1)*dpt; i++){
    regMres = make_float3(0.f); 
    regM0 = rotDev.rot_vec[i][threadIdx.x]; /* copy global -> register */
    float RKcoeff[4] = {1./6., 1./3., 1./3., 1./6.};
    float step[4] = {1./2., 1./2., 1., 1. };
    float noise = 2.0*sqrt(alpha*Th*h);
    curandState state=Hth.states[threadIdx.x + blockDim.x * blockIdx.x];
#pragma unroll 1
    for(int ir=0; ir<iR; ir++){
      regM = regM0; 
#pragma unroll 1
      for(int stRK=0; stRK<4; stRK++) {
        __syncthreads();
        H = Hext; 
          H = h*(H + alpha * cross(regM, H));
          regM = rotate(regM0, step[stRK]*H);
          if(stRK<2) regMres += RKcoeff[stRK]*rotate(regM0, H);
          else       regMres += RKcoeff[stRK]*regM;
      }
      regMres+= noise*cross(regM0, get_rand(&state));
      regMres = normalize(regMres);
    }
    rotDev.rot_vec[i][threadIdx.x] = regMres; /* copy register -> global */
    Hth.states[threadIdx.x + blockDim.x * blockIdx.x] = state;
  }
}
/*****************************
	Stencil with Hanis 
*****************************/
const int Nani=3;
__constant__ float4 ani_pars[Nani];
void init_pos() {
  float4 ani[Nani];
  for(int j=0; j<3; j++) {
    ani[j].x = ani[j].y = ani[j].z = 0.0; ani[j].w = 0.0;
  }
  ani[0].x = ani[1].y = ani[2].z = 1.0;
  ani[0].w = 2.0;
  cudaMemcpyToSymbol(ani_pars, ani, sizeof(ani_pars));
  cudaDeviceSynchronize();
}
__forceinline__ __device__ float3 Hani(float3& M) {
  float3 H={0,0,0};
  for(int i=0; i<3; i++) {
    float K=ani_pars[i].w;
    float3 c=make_float3(ani_pars[i]);
		H += (K*dot(M, c))*c;
  }
  return H;
}

__global__ void __launch_bounds__(Nv, 1+(Nv<=128)) stHani(const int dpt){
  float3 Hext = make_float3(Hx, Hy, Hz);
  float3 regM0, regM, regMres, H;
#pragma unroll 1
  for (int i=blockIdx.x*dpt; i<(blockIdx.x+1)*dpt; i++){
    regMres = make_float3(0.f); 
    regM0 = rotDev.rot_vec[i][threadIdx.x]; /* copy global -> register */
    float RKcoeff[4] = {1./6., 1./3., 1./3., 1./6.};
    float step[4] = {1./2., 1./2., 1., 1. };
#pragma unroll 1
    for(int ir=0; ir<iR; ir++){
      regM = regM0; 
#pragma unroll 1
      for(int stRK=0; stRK<4; stRK++) {
        __syncthreads();
	H = Hani(regM) + Hext;
        H = h*(H + alpha * cross(regM, H));
        regM = rotate(regM0, step[stRK]*H);
        if(stRK<2) regMres += RKcoeff[stRK]*rotate(regM0, H);
        else       regMres += RKcoeff[stRK]*regM;
      }
      regMres = normalize(regMres);
    }
    rotDev.rot_vec[i][threadIdx.x] = regMres; /* copy register -> global */
  }
}
/*****************************
	Stencil with Hexch 
*****************************/
const unsigned int Nnbr = 8;

__global__ void __launch_bounds__(Nv, 1+(Nv<=128)) stHexch(const int dpt){
  float3 Hext = make_float3(Hx, Hy, Hz);
  float3 regM0, regM, regMres, H;
#pragma unroll 1
  for (int i=blockIdx.x*dpt; i<(blockIdx.x+1)*dpt; i++){
    regMres = make_float3(0.f); 
    regM0 = rotDev.rot_vec[i][threadIdx.x]; /* copy global -> register */
    float RKcoeff[4] = {1./6., 1./3., 1./3., 1./6.};
    float step[4] = {1./2., 1./2., 1., 1. };
    __shared__ float3 shM[Nv];
    short unsigned int nbr[Nnbr];
    for ( short int sign=-1, id=0; sign<2; sign+=2 ) 
      for ( short unsigned int inbr=0; inbr<Nnbr/2; ++inbr, id++ )
        nbr[id] = ((threadIdx.x + sign*inbr+Nv)%Nv);
#pragma unroll 1
    for(int ir=0; ir<iR; ir++){
      regM = regM0; 
#pragma unroll 4
      for(int stRK=0; stRK<4; stRK++) {
        __syncthreads();
        shM[threadIdx.x] = regM;
        __syncthreads();
	H = Hext;
        for(unsigned int id=0; id<Nnbr; ++id) H += J*shM[nbr[id]];
        H = h*(H + alpha * cross(regM, H));
        regM = rotate(regM0+ir, step[stRK]*H+ir);
        if(stRK<2) regMres += RKcoeff[stRK]*rotate(regM0+ir, H+ir);
        else       regMres += RKcoeff[stRK]*regM;
      }
      regMres = normalize(regMres);
    }
    rotDev.rot_vec[i][threadIdx.x] = regMres; /* copy register -> global */
  }
}


/*****************************
	Stencil with all 
*****************************/
__global__ void __launch_bounds__(Nv, 1+(Nv<=128)) stHall(const int dpt, normRand Hth){
  float3 Hext = make_float3(Hx, Hy, Hz);
  float3 regM0, regM, regMres, H;
#pragma unroll 1
  for (int i=blockIdx.x*dpt; i<(blockIdx.x+1)*dpt; i++){
    regMres = make_float3(0.f); 
    regM0 = rotDev.rot_vec[i][threadIdx.x]; /* copy global -> register */
    float RKcoeff[4] = {1./6., 1./3., 1./3., 1./6.};
    float step[4] = {1./2., 1./2., 1., 1. };
    float noise = 2.0*sqrt(alpha*Th*h);
    curandState state=Hth.states[threadIdx.x + blockDim.x * blockIdx.x];
    __shared__ float3 shM[Nv];
    short unsigned int nbr[Nnbr];
    for ( short int sign=-1, id=0; sign<2; sign+=2 ) 
      for ( short unsigned int inbr=0; inbr<Nnbr/2; ++inbr, id++ )
        nbr[id] = ((threadIdx.x + sign*inbr+Nv)%Nv);
#pragma unroll 1
    for(int ir=0; ir<iR; ir++){
      regM = regM0; 
#pragma unroll 4
      for(int stRK=0; stRK<4; stRK++) {
        __syncthreads();
        shM[threadIdx.x] = regM;
        __syncthreads();
        H = Hani(regM) + Hext; 
        for(unsigned int id=0; id<Nnbr; ++id) H += J*shM[nbr[id]];
        H = h*(H + alpha * cross(regM, H));
        regM = rotate(regM0+ir, step[stRK]*H+ir);
        if(stRK<2) regMres += RKcoeff[stRK]*rotate(regM0+ir, H+ir);
        else       regMres += RKcoeff[stRK]*regM;
      }
      regMres+= noise*cross(regM0, get_rand(&state));
      regMres = normalize(regMres);
    }
    rotDev.rot_vec[i][threadIdx.x] = regMres; /* copy register -> global */
    Hth.states[threadIdx.x + blockDim.x * blockIdx.x] = state;
  }
}


#endif /* HEAP_H_ */
