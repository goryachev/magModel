#include "cuda_math.h"

#define XBIT (Nv/8)
#define YBIT (Nv/64)
#define ZBIT 1

typedef float floatT;
typedef float2 floatT2;
typedef float3 floatT3;
typedef float4 floatT4;

const int K4gn  = 8;  //Число kern-ов в Grain-е и Grape-е, число Grain-ов в Grape-е
const int K4gp  = Nv;
const int gn4gp = K4gp/K4gn;
const int Ngrapes= Na*Nv; 

__device__ __forceinline__ int gnId(int i) { return (i&(Nv-2*XBIT))/8+(i&(XBIT-2*YBIT))/4+(i&(YBIT-2))/2; }
__device__ __forceinline__ int knId(int i) { return (i&XBIT)/(XBIT/4)+(i&YBIT)/(YBIT/2)+(i&ZBIT); }
__device__ __forceinline__ int gnId() { return gnId(threadIdx.x); }
__device__ __forceinline__ int knId() { return knId(threadIdx.x); }

typedef float3 float3NT[Nv];
struct rot_pars
{
   float3NT* rot_vec;
   float3NT* H;
   float3NT* res;
}; /* struct rot_pars */

template <int Nat>
struct kern{  float3 m[Nat];
};/* struct kern */

template <int Nat>
struct Grape {
  struct Grain { floatT4 m[K4gn*3/(4/Nat)]; } grains[gn4gp];
  __device__ __forceinline__ floatT4& ref_m1() { 
	return grains[gnId()].m[knId()]; 
} /* ref_m1 */
  __device__ __forceinline__ void set_m2(floatT mx, floatT my) {
    floatT4 m=make_float4(mx,my,__shfl_down(mx,1),__shfl_down(my,1));
    if((threadIdx.x%2)==0) grains[gnId()].m[K4gn+knId()/2] = m;
} /* set_m2 */
  __device__ __forceinline__ void set_mom(float3 m1, float3 m2) { 
	ref_m1() = make_float4(m1.x,m1.y,m1.z,m2.z); 
	set_m2(m2.x,m2.y); 
} /* set_mom */
  __device__ __forceinline__ floatT2 get_m2() {
    float4 m;
    if(threadIdx.x%2==0) m=grains[gnId()].m[K4gn+(knId())/2];
    float2 m2=make_float2(__shfl_up(m.z,1),__shfl_up(m.w,1));
    if(threadIdx.x%2==0) return make_float2(m.x,m.y);
    else return m2;
} /* get_m2 */
  __device__ __forceinline__ void get_mom(float3& rm1, float3& rm2) { 
	float4 m1=ref_m1(); float2 m2=get_m2(); 
	rm1 = make_float3(m1.x,m1.y,m1.z); rm2 = make_float3(m2.x,m2.y,m1.w); 
} /* get_mom */ 
};/* struct Grape */

//template <int Nat>
struct Wine {
  Grape<Nat>* grapes;
  int3 MeshSz, MeshSh;
  __host__ __device__ int get_grapeID(int iS, int iA) { return Na*iS+iA; }
  __host__ __device__ Grape<Nat>& get_grape(int iS, int iA) { return grapes[get_grapeID(iS, iA)]; }
};/* struct Wine */

__constant__ rot_pars rotDev; 
__constant__ Wine wine;

template <int Nat>
struct wine_pars_host: public Wine{
  wine_pars_host(){
	std::string log;
	size_t sz = Ngrapes*sizeof(Grape<Nat>);//, sz2=Ngrapes*sizeof(int3);
	checkErr(cudaMalloc((void**) (&grapes), sz), (log="malloc"));
	checkErr(cudaMemset((void**) (&grapes), 0, sz), (log="memset"));
	checkErr(cudaMemcpyToSymbol(wine, this, sizeof(Wine)), (log="memcpy"));
  };
  void clean(){
	cudaFree(grapes); 
  };
};

wine_pars_host<Nat> wineHost;

