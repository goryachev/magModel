#ifndef TABLET_HPP
#define TABLET_HPP
//#include "wo_cuda_math.h"
#ifndef SWIG
struct Tablet;
#endif //SWIG

struct TabletIF {//Базовая структра интерфейса
  float MinBox[3], MaxBox[3];
  int Nmoms;
  Tablet* tablet;
  float * coords;
  float * moments;
// public:
  void clear();
  int run(int Nt=1);
  int init(float* pm=0, float* pc=0);
  int set();
};
#endif //TABLET_HPP
