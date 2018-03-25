#ifndef FCC_H_
#define FCC_H_
#include "cuda_math.h"
#include "stencil.cuh"

const int Nx = 1<<5;
const int Ny = 1<<5;
const int Nz = 1<<5;

const float _J = 0.1;

const int _nbr[][3]={{-1,-1, 0}, {-1, 0,-1}, {-1, 0, 0},
		    {-1, 0, 1}, {-1, 1, 0}, { 0,-1,-1},
		    { 0,-1, 0}, { 0,-1, 1}, { 0, 0,-1},
		    { 0, 0, 0}, { 0, 0, 1}, { 0, 1,-1},
		    { 0, 1, 0}, { 0, 1, 1}, { 1,-1, 0},
		    { 1, 0,-1}, { 1, 0, 0}, { 1, 0, 1},
		    { 1, 1, 0}	
};


#define xB1(ib) for(int i=0;i<xRank;++i) ib+=1<<i

struct knFCC{
  float3 m[4];
  __forceinline__ void exch(knFCC& _kn){
    for(short int ia=0;ia<4;ia++)  
    for(short int _ia=ia++; _ia!=ia; ia=(ia+1)%4) m[ia]+=J*_kn.m[_ia];
  }
};

/*****************************
	FCC StenciL 
*****************************/
__global__ void __launch_bounds__(Nv, 1+(Nv<=128)) FCC(const int dpt) {
  float3 Hext = make_float3(Hx, Hy, Hz);
//  curandState state=Hth.states[threadIdx.x + blockDim.x * blockIdx.x];
    knFCC regM0, regM, regMres;
    /* copy global -> register */
    regM0.m[0] = make_float3(1., 1., 1.);
    regM0.m[1] = make_float3(1., 1., 1.);
    regM0.m[2] = make_float3(1., 1., 1.);
    regM0.m[3] = make_float3(1., 1., 1.);
   
    float RKcoeff[4] = {1./6., 1./3., 1./3., 1./6.};
    float step[4] = {1./2., 1./2., 1., 1. };
    regMres.m[0] = regMres.m[1] = regMres.m[2] = regMres.m[3] = make_float3(0.f); 
    regM.m[0] = regM0.m[0]; regM.m[1] = regM0.m[1]; regM.m[2] = regM0.m[2]; regM.m[3] = regM0.m[3];
    float3 Hexch[4];
    for(int stRK=0; stRK<4; stRK++) {
      __syncthreads();
      Hexch[0] = Hexch[1] = Hexch[2] = Hexch[3] = Hext;
      if(true) Hexch[0] += make_float3(1);
      if(true) Hexch[1] += make_float3(1);
      if(true) Hexch[2] += make_float3(1);
      if(true) Hexch[3] += make_float3(1);
      __syncthreads();
      for(int il=0; il<12; il++) {   }
      for(int iA=0; iA<4; iA++) {
        Hexch[iA] = h*(Hexch[iA] + alpha * cross(regM.m[iA], Hexch[iA]));
        regM.m[iA] = rotate(regM0.m[iA], step[stRK]*Hexch[iA]);
        if(stRK<2) regMres.m[iA] += RKcoeff[stRK]*rotate(regM0.m[iA], Hexch[iA]);
        else       regMres.m[iA] += RKcoeff[stRK]*regM.m[iA];
      }

    }
/*    float noise = 2.0*sqrt(alpha*Th*h);
    regMres.m[0] += noise*cross(regM0.m[0], get_rand(&state));
    regMres.m[1] += noise*cross(regM0.m[1], get_rand(&state));
    regMres.m[0] = normalize(regMres.m[0]);
    regMres.m[1] = normalize(regMres.m[1]);
  *//* copy register -> global */
//  Hth.states[threadIdx.x + blockDim.x * blockIdx.x] = state;
};

#endif /* HEAP_H_ */
