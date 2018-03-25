#include <cstdio>
#include <cmath>

#define lp4grid(x) for(int i=-2; i<3; i++) for(int j=-2; j<3; j++) for(int k=-2; k<3; k++){ x;} 

struct float3{
  float x;
  float y;
  float z;
  float3(){ };
  float3(float _x, float _y, float _z): x(_x), y(_y), z(_z) { };
  inline float3 make_float3(float x, float y, float z){ float3 v; v.x=x; v.y=y; v.z=z; return v;}
  float3 operator+(float3 v){return make_float3(x+v.x, y+v.y, z+v.z);}
  float3 operator-(float3 v){return make_float3(x-v.x, y-v.y, z-v.z);}
  float3 operator=(float3 v){ x=v.x, y=v.y, z=v.z;}
  float3 operator+=(float3 v){ x=x+v.x, y=y+v.y, z=z+v.z;}
  float3 operator-=(float3 v){ x=x-v.x, y=y-v.y, z=z-v.z;}
  bool operator==(float3 v){ if( x==v.x && y==v.y && z==v.z) return true; return false;}
};

inline float3 make_float3(float x, float y, float z){ float3 v; v.x=x; v.y=y; v.z=z; return v;}

struct kern{
  float3 at[4];
  bool on;
  kern():on(false){ }
  void set_coord(float x, float y, float z){
	at[0].x=x    ; at[0].y=y    ; at[0].z=z    ;
	at[1].x=x+0.5; at[1].y=y+0.5; at[1].z=z    ;
	at[2].x=x+0.5; at[2].y=y    ; at[2].z=z+0.5;
	at[3].x=x    ; at[3].y=y+0.5; at[3].z=z+0.5;
  }
  kern(float x, float y, float z):on(false){
    set_coord(x, y, z); 
  }
  void isON(int& N){ if(on) N++;}
  void print(){ 
    printf("%3.1f %3.1f %3.1f\t", at[0].x, at[0].y, at[0].z );
    printf("%3.1f %3.1f %3.1f\t", at[1].x, at[1].y, at[1].z );
    printf("%3.1f %3.1f %3.1f\t", at[2].x, at[2].y, at[2].z );
    printf("%3.1f %3.1f %3.1f\n", at[3].x, at[3].y, at[3].z );
  }
  void check(float3 v){ if( v==at[0] || v==at[1] || v==at[2] || v==at[3] ) on=true; }
};


int main(){
  kern grid[5][5][5];
  lp4grid( grid[i+2][j+2][k+2].set_coord(float(i), float(j), float(k)) )
 
  kern k0(0., 0., 0.);
  for(int ia=0; ia<4; ia++){ 
    lp4grid( grid[i+2][j+2][k+2].check(k0.at[ia]+make_float3( 0.5,  0.5, 0.) ) )
    lp4grid( grid[i+2][j+2][k+2].check(k0.at[ia]+make_float3( 0.5, -0.5, 0.) ) )
    lp4grid( grid[i+2][j+2][k+2].check(k0.at[ia]+make_float3(-0.5,  0.5, 0.) ) )
    lp4grid( grid[i+2][j+2][k+2].check(k0.at[ia]+make_float3(-0.5, -0.5, 0.) ) )

    lp4grid( grid[i+2][j+2][k+2].check(k0.at[ia]+make_float3(0., 0.5,  0.5) ) )
    lp4grid( grid[i+2][j+2][k+2].check(k0.at[ia]+make_float3(0., 0.5, -0.5) ) )
    lp4grid( grid[i+2][j+2][k+2].check(k0.at[ia]+make_float3(0.,-0.5,  0.5) ) )
    lp4grid( grid[i+2][j+2][k+2].check(k0.at[ia]+make_float3(0.,-0.5, -0.5) ) )
    
    lp4grid( grid[i+2][j+2][k+2].check(k0.at[ia]+make_float3( 0.5, 0.,  0.5) ) )
    lp4grid( grid[i+2][j+2][k+2].check(k0.at[ia]+make_float3( 0.5, 0., -0.5) ) )
    lp4grid( grid[i+2][j+2][k+2].check(k0.at[ia]+make_float3(-0.5, 0.,  0.5) ) )
    lp4grid( grid[i+2][j+2][k+2].check(k0.at[ia]+make_float3(-0.5, 0., -0.5) ) )
  }
//  lp4grid( grid[i+2][j+2][k+2].print() )
  int N=-1;
  lp4grid( grid[i+2][j+2][k+2].isON(N) )
  lp4grid( if(grid[i+2][j+2][k+2].on) grid[i+2][j+2][k+2].print() )
  printf("%d\n", N);
  return 0;
}
