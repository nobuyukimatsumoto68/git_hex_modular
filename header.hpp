#pragma once

#include <random>
#include <stack>
#include <cassert>
#include <cmath>

// #include <sstream>
#include <fstream>

#include <string>
#include <vector>
#include <array>
#include <functional>
// #include <any>


using Idx = std::size_t;
// using Idx = unsigned long int;
// using Idx = int;
using Double = long double;
// using int = short;

using namespace std;

constexpr int TWO = 2;
constexpr int THREE = 3;
constexpr int SIX = 6;

int mult = 8;
Idx Lx = 3*2*mult; // 12
Idx Ly = 3*1*mult;

// constexpr Idx Lx = 6*4; // 12
// constexpr Idx Ly = 2*Lx;
constexpr int nparallel = 12; //12


constexpr int nu = 1; // PP, PA, AA, AP

constexpr Double tt = 0.1;
constexpr Double kappa[3] = {
  2.0 - 4.0/3.0 * std::exp(-tt),
  2.0/3.0 * std::exp(-tt),
  2.0/3.0 * std::exp(-tt)
};
const Double cos6 = std::cos(M_PI/6.0);

const Double Betac[3] = {
  std::atanh( cos6 * kappa[0] ),
  std::atanh( cos6 * kappa[1] ),
  std::atanh( cos6 * kappa[2] )
  // 0.5 * std::log(2.0 + std::sqrt(3.0)),
  // 0.5 * std::log(2.0 + std::sqrt(3.0)),
  // 0.5 * std::log(2.0 + std::sqrt(3.0))
};

const Double B[3] = {
  cos6 / (1.0 - kappa[0]*kappa[0] * cos6 * cos6),
  cos6 / (1.0 - kappa[1]*kappa[1] * cos6 * cos6),
  cos6 / (1.0 - kappa[2]*kappa[2] * cos6 * cos6)
};
//
// constexpr Double alat = 1.0/Lx;
// constexpr Double xipsi = std::sqrt( 1.5*std::sqrt(3.0)*alat / (2.0*M_PI) );

#ifndef _OPENMP
// int omp_get_thread_num() { return 0; }
#endif



std::mt19937 gen;
std::uniform_real_distribution<Double> d01D(0.0, 1.0); // (1, 6);
std::uniform_int_distribution<int> d01I(0, 1); // (1, 6);
std::uniform_int_distribution<Idx> d0N(0, Lx*Ly-1); // (1, 6);
void set_gen( const int seed ) {
  std::mt19937 gen0( seed );
  gen.seed( gen0() );
}
Double dist01(){ return d01D(gen); }
Idx dist0N(){ return d0N(gen); }
int distpm1(){ return 2*d01I(gen)-1; }


// ---------------
// GLOBAL FUNCTIONS
// ---------------

Idx idx(const Idx x, const Idx y)  { return x + Lx*y; }

void get_xy(Idx& x, Idx& y, const Idx i)  {
  x = (i+Lx)%Lx;
  y = (i-x)/Lx;
}

inline int get_char( const Idx x, const Idx y) { return (x-y+Lx*Ly)%3; }

bool is_site(const Idx x, const Idx y)  {
  const Idx c = get_char(x,y);
  bool res = true;
  if(c==1) res = false; // e.g., (1,0)
  return res;
}

bool is_site(const Idx i)  {
  Idx x,y;
  get_xy(x, y, i);
  return is_site(x,y);
}


bool is_link(const Idx x, const Idx y, const int mu)  {
  const Idx c = get_char(x,y);
  bool res = false;
  if(c==0 && mu<3) res = true; // e.g., (0,0)
  else if(c==2 && mu>=3) res = true; // e.g., (0,1)
  return res;
}

bool is_link(const Idx i, const int mu)  {
  Idx x,y;
  get_xy(x, y, i);
  return is_link(x,y,mu);
}


void cshift(Idx& xp, Idx& yp, const Idx x, const Idx y, const int mu)  {
  int dx = -(mu+2)%3+1;
  int dy = -(mu+1)%3+1;
  if(mu>=3){
    dx *= -1;
    dy *= -1;
  }
  xp = (x+dx+Lx)%Lx;
  yp = (y+dy+Ly)%Ly;
}

void cshift(Idx& ip, const Idx i, const int mu)  {
  Idx x, y, xp, yp;
  get_xy(x, y, i);
  cshift(xp, yp, x, y, mu);
  ip = idx(xp, yp);
}



// ----------------
// CLASS DEFINITIONS
// ----------------


struct Spin {
  Idx N;
  std::vector<int> s;

  inline int& operator()(const Idx x, const Idx y) { return s[idx(x,y)]; }
  inline int operator()(const Idx x, const Idx y) const { return s[idx(x,y)]; }

  int& operator[](const Idx i) { return s[i]; }
  int operator[](const Idx i) const { return s[i]; }

  Spin() = delete;

  Spin( const int N_ )
    : N(N_)
    , s( N )
  {
    set1();
  }

  void set1() {
    for(Idx i=0; i<Lx*Ly; i++){
      if( is_site(i) ) s[i] = 1;
      else s[i] = 0;
    }
  }

  void random() {
    for(Idx i=0; i<Lx*Ly; i++){
      if( is_site(i) ) s[i] = distpm1();
      else s[i] = 0;
    }
  }

  int ss( const Idx x, const Idx y, const int mu ) const {
    Idx xp, yp;
    cshift(xp, yp, x, y, mu);
    return (*this)(x,y) * (*this)(xp,yp);
  }

  Double ss_even( const int mu ) const {
    Double tot = 0.0;
    Idx counter = 0;

    for(Idx x=0; x<Lx; x++){
      for(Idx y=0; y<Ly; y++){
        if( !is_link(x,y,mu) ) continue;
        tot += ss(x,y,mu);
        counter++;
      }
    }
    return tot/counter;
  }

  Double ss_corr( const Idx dx, const Idx dy ) const {
    assert(0<=dx && dx<Lx);
    assert(0<=dy && dy<Ly);

    Double res = 0.0;
    int counter = 0;

    for(Idx x=0; x<Lx; x++){
      for(Idx y=0; y<Ly; y++){
        if( !is_site(x,y) ) continue;
        const Idx xp = (x+dx)%Lx;
        const Idx yp = (y+dy)%Ly;
        if( !is_site(xp,yp) ) continue;

        res += (*this)(x,y) * (*this)(xp,yp);
        counter++;
      }}

    res /= counter;
    return res;
  }


  std::vector<Double> ss_corr() const {
    std::vector<Double> corr(N, 0.0);

#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
    // #pragma omp parallel for num_threads(nparallel) schedule(static)
#endif
    for(Idx dx=0; dx<Lx; dx++){
      for(Idx dy=0; dy<Ly; dy++){
        corr[idx(dx,dy)] = ss_corr( dx, dy );
      }}

    return corr;
  }


  Double eps( const Idx x, const Idx y ) const {
    assert(0<=x && x<Lx);
    assert(0<=y && y<Ly);
    assert( is_site(x,y) );

    Double res = 0.0;
    for(int mu=0; mu<SIX; mu++){
      if( !is_link(x,y,mu) ) continue;
      Idx xp, yp;
      cshift( xp, yp, x, y, mu );
      res += (*this)(x,y) * (*this)(xp,yp);
      res *= 0.5 * kappa[mu%3] * B[mu%3];
    }
    res -= 1.0;

    return res;
  }


  Double eps_1pt() const {
    Double res = 0.0;

// #ifdef _OPENMP
// #pragma omp parallel for num_threads(nparallel)
// #endif
    int counter = 0;
    for(Idx x=0; x<Lx; x++){
      for(Idx y=0; y<Ly; y++){
        if( !is_site(x,y) ) continue;
        res += eps( x, y );
        counter++;
      }}

    res /= counter;
    return res;
  }


  Double epseps_corr( const Idx dx, const Idx dy ) const {
    assert(0<=dx && dx<Lx);
    assert(0<=dy && dy<Ly);

    Double res = 0.0;
    int counter = 0;

    for(Idx x=0; x<Lx; x++){
      for(Idx y=0; y<Ly; y++){
        if( !is_site(x,y) ) continue;
        const Idx xp = (x+dx)%Lx;
        const Idx yp = (y+dy)%Ly;
        if( !is_site(xp,yp) ) continue;

        res += eps(x,y) * eps(xp,yp);
        counter++;
      }}

    res /= counter;
    return res;
  }


  std::vector<Double> epseps_corr() const {
    std::vector<Double> corr(N, 0.0);

#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
    // #pragma omp parallel for num_threads(nparallel) schedule(static)
#endif
    for(Idx dx=0; dx<Lx; dx++){
      for(Idx dy=0; dy<Ly; dy++){
        corr[idx(dx,dy)] = epseps_corr( dx, dy );
      }}

    return corr;
  }


  Double K( const Idx x, const Idx y, const int mu ) const {
    assert(0<=x && x<Lx);
    assert(0<=y && y<Ly);
    // assert(0<=mu && mu<3);
    assert( is_link(x,y,mu) );

    Idx xp, yp;
    cshift( xp, yp, x, y, mu );

    Double res = (*this)(x,y)*(*this)(xp,yp);
    res -= kappa[mu%3]*cos6;
    res *= B[mu%3];
    return res;
  }


  // Double K_1pt( const int mu ) const {
  //   // std::vector<Double> tmp(nparallel, 0.0);
  //   // std::vector<int> counter(nparallel, 0);
  //   int counter = 0;
  //   Double res = 0.0;

  //   // #ifdef _OPENMP
  //   // #pragma omp parallel for num_threads(nparallel) schedule(static)
  //   // #endif
  //   for(Idx x=0; x<Lx; x++){
  //     for(Idx y=0; y<Ly; y++){
  //       if( !is_link(x,y,mu) ) continue;
  //       res += K( x, y, mu );
  //       counter++;
  //       // tmp[omp_get_thread_num()] += K( x, y, mu );
  //       // counter[omp_get_thread_num()]++;
  //     }}

  //   // Double res = 0.0;
  //   // int tot = 0;
  //   // for(int i=0; i<nparallel; i++) {
  //   //   res += tmp[i];
  //   //   tot += counter[i];
  //   // }
  //   res /= counter;
  //   // res /= tot;
  //   return res;
  // }


  Double Txx_even( const Idx x, const Idx y ) const {
    assert(0<=x && x<Lx);
    assert(0<=y && y<Ly);
    assert( get_char(x,y)==0 );

    Idx xp, yp;
    Double res = 0.0;
    int mu;

    mu = 0;
    cshift( xp, yp, x, y, mu );
    res += 2.0*K(x, y, mu);
    res -= ( eps(x,y)+eps(xp,yp) ); // mu deriv

    mu = 1;
    cshift( xp, yp, x, y, mu );
    res -= K(x, y, mu);
    res += 0.5 * ( eps(x,y)+eps(xp,yp) ); // mu deriv

    mu = 2;
    cshift( xp, yp, x, y, mu );
    res -= K(x, y, mu);
    res += 0.5 * ( eps(x,y)+eps(xp,yp) ); // mu deriv

    res /= 3.0;

    return res;
  }

  Double Txx_odd( const Idx x, const Idx y ) const {
    Idx xp, yp;
    Double res = 0.0;
    int mu;

    mu = 3;
    cshift( xp, yp, x, y, mu );
    res += 2.0*K(x, y, mu);
    res -= ( eps(x,y)+eps(xp,yp) ); // mu deriv

    mu = 4;
    cshift( xp, yp, x, y, mu );
    res -= K(x, y, mu);
    res += 0.5 * ( eps(x,y)+eps(xp,yp) ); // mu deriv

    mu = 5;
    cshift( xp, yp, x, y, mu );
    res -= K(x, y, mu);
    res += 0.5 * ( eps(x,y)+eps(xp,yp) ); // mu deriv

    res /= 3.0;

    return res;
  }

  Double Txx( const Idx x, const Idx y ) const {
    assert(0<=x && x<Lx);
    assert(0<=y && y<Ly);
    const int c = get_char(x,y);
    Double res = 0.0;
    if(c==0) res = Txx_even(x,y);
    else if(c==2) res = Txx_odd(x,y);
    else assert(false);
    return res;
  }


  Double Txy_even( const Idx x, const Idx y ) const {
    assert( get_char(x,y)==0 );

    Idx xp, yp;
    Double res = 0.0;
    int mu;

    mu = 2;
    cshift( xp, yp, x, y, mu );
    res += K(x, y, mu);
    res -= 0.5 * ( eps(x,y)+eps(xp,yp) ); // mu deriv

    mu = 1;
    cshift( xp, yp, x, y, mu );
    res -= K(x, y, mu);
    res += 0.5 * ( eps(x,y)+eps(xp,yp) ); // mu deriv

    res /= std::sqrt(3.0);

    return res;
  }


  Double Txy_odd( const Idx x, const Idx y ) const {
    assert( get_char(x,y)==2 );

    Idx xp, yp;
    Double res = 0.0;
    int mu;

    mu = 5;
    cshift( xp, yp, x, y, mu );
    res += K(x, y, mu);
    res -= 0.5 * ( eps(x,y)+eps(xp,yp) ); // mu deriv

    mu = 4;
    cshift( xp, yp, x, y, mu );
    res -= K(x, y, mu);
    res += 0.5 * ( eps(x,y)+eps(xp,yp) ); // mu deriv

    res /= std::sqrt(3.0);

    return res;
  }

  Double Txy( const Idx x, const Idx y ) const {
    assert(0<=x && x<Lx);
    assert(0<=y && y<Ly);
    const int c = get_char(x,y);
    Double res = 0.0;
    if(c==0) res = Txy_even(x,y);
    else if(c==2) res = Txy_odd(x,y);
    else assert(false);
    return res;
  }

  Double TxxN( const Idx x, const Idx y ) const {
    assert(0<=x && x<Lx);
    assert(0<=y && y<Ly);

    Idx xp, yp;
    Double res = 0.0;
    int mu=0;

    int c = get_char(x,y);
    if(c==2) mu+=3;
    else if(c==1) assert(false);

    cshift( xp, yp, x, y, mu );
    res += K(x, y, mu);
    res -= 0.5*( eps(x,y)+eps(xp,yp) ); // mu deriv

    res -= 0.25;

    return res;
  }

  Double TxyN( const Idx x, const Idx y ) const {
    assert(0<=x && x<Lx);
    assert(0<=y && y<Ly);

    Idx xp, yp;
    Double res = 0.0;

    int c = get_char(x,y);
    if(c==1) assert(false);

    int mu=2;
    if(c==2) mu+=3;

    cshift( xp, yp, x, y, mu );
    res += K(x, y, mu);
    res -= 0.5*( eps(x,y)+eps(xp,yp) ); // mu deriv

    mu=1;
    if(c==2) mu+=3;

    cshift( xp, yp, x, y, mu );
    res -= K(x, y, mu);
    res += 0.5*( eps(x,y)+eps(xp,yp) ); // mu deriv

    res /= std::sqrt(3.0);

    return res;
  }

  Double TyyN( const Idx x, const Idx y ) const {
    assert(0<=x && x<Lx);
    assert(0<=y && y<Ly);

    Idx xp, yp;
    Double res = 0.0;

    int c = get_char(x,y);
    if(c==1) assert(false);

    int mu=2;
    if(c==2) mu+=3;

    cshift( xp, yp, x, y, mu );
    res += K(x, y, mu);
    res -= 0.5*( eps(x,y)+eps(xp,yp) ); // mu deriv

    mu=1;
    if(c==2) mu+=3;

    cshift( xp, yp, x, y, mu );
    res += K(x, y, mu);
    res -= 0.5*( eps(x,y)+eps(xp,yp) ); // mu deriv

    mu=0;
    if(c==2) mu+=3;

    cshift( xp, yp, x, y, mu );
    res -= 0.5*K(x, y, mu);
    res += 0.25*( eps(x,y)+eps(xp,yp) ); // mu deriv

    res *= 2.0/3.0;

    res -= 0.25;

    return res;
  }

  Double TxxF( const Idx x, const Idx y ) const {
    const Double txx = TxxN(x,y);
    const Double tyy = TyyN(x,y);

    return txx - 0.5*(txx+tyy);
  }

  Double TxyF( const Idx x, const Idx y ) const {
    return TxyN(x,y);
  }

  Double TyyF( const Idx x, const Idx y ) const {
    const Double txx = TxxN(x,y);
    const Double tyy = TyyN(x,y);

    return tyy - 0.5*(txx+tyy);
  }


  Double Txx_1pt( ) const {
    Double res = 0.0;
    int counter = 0;

    for(Idx x=0; x<Lx; x++){
      for(Idx y=0; y<Ly; y++){
        if( !is_site(x,y) ) continue;
        res += TxxF( x, y );
        counter++;
      }}

    res /= counter;
    return res;
  }


  Double TxxN_1pt( ) const {
    Double res = 0.0;
    int counter = 0;

    for(Idx x=0; x<Lx; x++){
      for(Idx y=0; y<Ly; y++){
        if( !is_site(x,y) ) continue;
        res += TxxN( x, y );
        counter++;
      }}

    res /= counter;
    return res;
  }

  Double Txy_1pt( ) const {
    Double res = 0.0;
    int counter = 0;

    for(Idx x=0; x<Lx; x++){
      for(Idx y=0; y<Ly; y++){
        if( !is_site(x,y) ) continue;
        res += TxyF( x, y );
        counter++;
      }}

    res /= counter;
    return res;
  }

  Double Tyy_1pt( ) const {
    Double res = 0.0;
    int counter = 0;

    for(Idx x=0; x<Lx; x++){
      for(Idx y=0; y<Ly; y++){
        if( !is_site(x,y) ) continue;
        res += TyyF( x, y );
        counter++;
      }}

    res /= counter;
    return res;
  }

  Double TyyN_1pt( ) const {
    Double res = 0.0;
    int counter = 0;

    for(Idx x=0; x<Lx; x++){
      for(Idx y=0; y<Ly; y++){
        if( !is_site(x,y) ) continue;
        res += TyyN( x, y );
        counter++;
      }}

    res /= counter;
    return res;
  }


  Double TxxTxx_corr( const Idx dx, const Idx dy ) const {
    assert(0<=dx && dx<Lx);
    assert(0<=dy && dy<Ly);

    Double res = 0.0;
    int counter = 0;

    for(Idx x=0; x<Lx; x++){
      for(Idx y=0; y<Ly; y++){
        if( !is_site(x,y) ) continue;
        const Idx xp = (x+dx)%Lx;
        const Idx yp = (y+dy)%Ly;
        if( !is_site(xp,yp) ) continue;
        res += Txx(x,y) * Txx(xp,yp);
        counter++;
      }}

    res /= counter;
    return res;
  }

  Double TxxTxy_corr( const Idx dx, const Idx dy ) const {
    assert(0<=dx && dx<Lx);
    assert(0<=dy && dy<Ly);

    Double res = 0.0;
    int counter = 0;

    for(Idx x=0; x<Lx; x++){
      for(Idx y=0; y<Ly; y++){
        if( !is_site(x,y) ) continue;
        const Idx xp = (x+dx)%Lx;
        const Idx yp = (y+dy)%Ly;
        if( !is_site(xp,yp) ) continue;
        res += Txx(x,y) * Txy(xp,yp);
        counter++;
      }}

    res /= counter;
    return res;
  }


  std::vector<Double> TxxTxx_corr() const {
    std::vector<Double> corr(N, 0.0);

#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
    // #pragma omp parallel for num_threads(nparallel) schedule(static)
#endif
    for(Idx dx=0; dx<Lx; dx++){
      for(Idx dy=0; dy<Ly; dy++){
        corr[idx(dx,dy)] = TxxTxx_corr( dx, dy );
      }}
    return corr;
  }


  std::vector<Double> TxxTxy_corr() const {
    std::vector<Double> corr(N, 0.0);

#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
    // #pragma omp parallel for num_threads(nparallel) schedule(static)
#endif
    for(Idx dx=0; dx<Lx; dx++){
      for(Idx dy=0; dy<Ly; dy++){
        corr[idx(dx,dy)] = TxxTxy_corr( dx, dy );
      }}
    return corr;
  }


  //  Double Txx_eps( const Idx dx1, const Idx dy1 ) const {
//     assert(0<=dx1 && dx1<Lx);
//     assert(0<=dy1 && dy1<Ly);

//     Double res = 0.0;
//     int counter = 0;

//     for(Idx x=0; x<Lx; x++){
//       for(Idx y=0; y<Ly; y++){
//         if( !is_site(x,y) ) continue;
//         const Idx x1 = (x+dx1)%Lx;
//         const Idx y1 = (y+dy1)%Ly;
//         if( !is_site(x1,y1) ) continue;
//         res += Txx(x,y) * eps(x1,y1);
//         counter++;
//       }}

//     res /= counter;
//     return res;
//   }


//   std::vector<Double> Txx_eps_corr() const {
//     std::vector<Double> corr(N, 0.0);

// #ifdef _OPENMP
// #pragma omp parallel for num_threads(nparallel)
//     // #pragma omp parallel for num_threads(nparallel) schedule(static)
// #endif
//     for(Idx dx1=0; dx1<Lx; dx1++){
//       for(Idx dy1=0; dy1<Ly; dy1++){
//         corr[idx(dx1,dy1)] = Txx_eps( dx1, dy1 );
//       }}
//     return corr;
//   }


//   Double Txy_eps( const Idx dx1, const Idx dy1 ) const {
//     assert(0<=dx1 && dx1<Lx);
//     assert(0<=dy1 && dy1<Ly);

//     Double res = 0.0;
//     int counter = 0;

//     for(Idx x=0; x<Lx; x++){
//       for(Idx y=0; y<Ly; y++){
//         if( !is_site(x,y) ) continue;
//         const Idx x1 = (x+dx1)%Lx;
//         const Idx y1 = (y+dy1)%Ly;
//         if( !is_site(x1,y1) ) continue;
//         res += Txy(x,y) * eps(x1,y1);
//         counter++;
//       }}

//     res /= counter;
//     return res;
//   }


//   std::vector<Double> Txy_eps_corr() const {
//     std::vector<Double> corr(N, 0.0);

// #ifdef _OPENMP
// #pragma omp parallel for num_threads(nparallel)
//     // #pragma omp parallel for num_threads(nparallel) schedule(static)
// #endif
//     for(Idx dx1=0; dx1<Lx; dx1++){
//       for(Idx dy1=0; dy1<Ly; dy1++){
//         corr[idx(dx1,dy1)] = Txy_eps( dx1, dy1 );
//       }}
//     return corr;
//   }


//   Double Txx_ss( const Idx dx1, const Idx dy1, const Idx dx2, const Idx dy2 ) const {
//     assert(0<=dx1 && dx1<Lx);
//     assert(0<=dy1 && dy1<Ly);
//     assert(0<=dx2 && dx2<Lx);
//     assert(0<=dy2 && dy2<Ly);

//     Double res = 0.0;
//     int counter = 0;

//     for(Idx x=0; x<Lx; x++){
//       for(Idx y=0; y<Ly; y++){
//         if( !is_site(x,y) ) continue;
//         const Idx x1 = (x+dx1)%Lx;
//         const Idx y1 = (y+dy1)%Ly;
//         const Idx x2 = (x+dx2)%Lx;
//         const Idx y2 = (y+dy2)%Ly;
//         if( !is_site(x1,y1) || !is_site(x2,y2) ) continue;
//         res += Txx(x,y) * (*this)(x1,y1) * (*this)(x2,y2);
//         counter++;
//       }}

//     res /= counter;
//     return res;
//   }


//   std::vector<Double> Txx_ss_corr( const Idx dx1, const Idx dy1 ) const {
//     std::vector<Double> corr(N, 0.0);

// #ifdef _OPENMP
// #pragma omp parallel for num_threads(nparallel)
//     // #pragma omp parallel for num_threads(nparallel) schedule(static)
// #endif
//     for(Idx dx2=0; dx2<Lx; dx2++){
//       for(Idx dy2=0; dy2<Ly; dy2++){
//         corr[idx(dx2,dy2)] = Txx_ss( dx1, dy1, dx2, dy2 );
//       }}
//     return corr;
//   }



//   Double Txy_ss( const Idx dx1, const Idx dy1, const Idx dx2, const Idx dy2 ) const {
//     assert(0<=dx1 && dx1<Lx);
//     assert(0<=dy1 && dy1<Ly);
//     assert(0<=dx2 && dx2<Lx);
//     assert(0<=dy2 && dy2<Ly);

//     Double res = 0.0;
//     int counter = 0;

//     for(Idx x=0; x<Lx; x++){
//       for(Idx y=0; y<Ly; y++){
//         if( !is_site(x,y) ) continue;
//         const Idx x1 = (x+dx1)%Lx;
//         const Idx y1 = (y+dy1)%Ly;
//         const Idx x2 = (x+dx2)%Lx;
//         const Idx y2 = (y+dy2)%Ly;
//         if( !is_site(x1,y1) || !is_site(x2,y2) ) continue;
//         res += Txy(x,y) * (*this)(x1,y1) * (*this)(x2,y2);
//         counter++;
//       }}

//     res /= counter;
//     return res;
//   }


//   std::vector<Double> Txy_ss_corr( const Idx dx1, const Idx dy1 ) const {
//     std::vector<Double> corr(N, 0.0);

// #ifdef _OPENMP
// #pragma omp parallel for num_threads(nparallel)
//     // #pragma omp parallel for num_threads(nparallel) schedule(static)
// #endif
//     for(Idx dx2=0; dx2<Lx; dx2++){
//       for(Idx dy2=0; dy2<Ly; dy2++){
//         corr[idx(dx2,dy2)] = Txy_ss( dx1, dy1, dx2, dy2 );
//       }}
//     return corr;
//   }


  Double Txx_ss( const Idx x0, const Idx y0,
                 const Idx x1, const Idx y1,
                 const Idx x2, const Idx y2 ) const {
    assert(0<=x0 && x0<Lx);
    assert(0<=y0 && y0<Ly);
    assert(0<=x1 && x1<Lx);
    assert(0<=y1 && y1<Ly);
    assert(0<=x2 && x2<Lx);
    assert(0<=y2 && y2<Ly);

    Double res = 0.0;
    int counter = 0;

    for(Idx dx=0; dx<Lx; dx++){
      for(Idx dy=0; dy<Ly; dy++){
        // if( !is_site(x,y) ) continue;
        const Idx x0p = (dx+x0)%Lx;
        const Idx y0p = (dy+y0)%Ly;
        const Idx x1p = (dx+x1)%Lx;
        const Idx y1p = (dy+y1)%Ly;
        const Idx x2p = (dx+x2)%Lx;
        const Idx y2p = (dy+y2)%Ly;
        if( !is_site(x0p,y0p) || !is_site(x1p,y1p) || !is_site(x2p,y2p) ) continue;
        res += TxxF(x0p,y0p) * (*this)(x1p,y1p) * (*this)(x2p,y2p);
        counter++;
      }}

    res /= counter;
    return res;
  }


  std::vector<Double> Txx_ss_corr( const Idx x1, const Idx y1,
                                   const Idx x2, const Idx y2 ) const {
    std::vector<Double> corr(N, 0.0);

#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
    // #pragma omp parallel for num_threads(nparallel) schedule(static)
#endif
    for(Idx x0=0; x0<Lx; x0++){
      for(Idx y0=0; y0<Ly; y0++){
        corr[idx(x0,y0)] = Txx_ss( x0, y0, x1, y1, x2, y2 );
      }}
    return corr;
  }


  Double Txy_ss( const Idx x0, const Idx y0,
                 const Idx x1, const Idx y1,
                 const Idx x2, const Idx y2 ) const {
    assert(0<=x0 && x0<Lx);
    assert(0<=y0 && y0<Ly);
    assert(0<=x1 && x1<Lx);
    assert(0<=y1 && y1<Ly);
    assert(0<=x2 && x2<Lx);
    assert(0<=y2 && y2<Ly);

    Double res = 0.0;
    int counter = 0;

    for(Idx dx=0; dx<Lx; dx++){
      for(Idx dy=0; dy<Ly; dy++){
        // if( !is_site(x,y) ) continue;
        const Idx x0p = (dx+x0)%Lx;
        const Idx y0p = (dy+y0)%Ly;
        const Idx x1p = (dx+x1)%Lx;
        const Idx y1p = (dy+y1)%Ly;
        const Idx x2p = (dx+x2)%Lx;
        const Idx y2p = (dy+y2)%Ly;
        if( !is_site(x0p,y0p) || !is_site(x1p,y1p) || !is_site(x2p,y2p) ) continue;
        // if( !is_site(x1p,y1p) || !is_site(x2p,y2p) ) continue;
        res += TxyF(x0p,y0p) * (*this)(x1p,y1p) * (*this)(x2p,y2p);
        counter++;
      }}

    res /= counter;
    return res;
  }


  std::vector<Double> Txy_ss_corr( const Idx x1, const Idx y1,
                                   const Idx x2, const Idx y2 ) const {
    std::vector<Double> corr(N, 0.0);

#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
    // #pragma omp parallel for num_threads(nparallel) schedule(static)
#endif
    for(Idx x0=0; x0<Lx; x0++){
      for(Idx y0=0; y0<Ly; y0++){
        corr[idx(x0,y0)] = Txy_ss( x0, y0, x1, y1, x2, y2 );
      }}
    return corr;
  }

  Double Tyy_ss( const Idx x0, const Idx y0,
                 const Idx x1, const Idx y1,
                 const Idx x2, const Idx y2 ) const {
    assert(0<=x0 && x0<Lx);
    assert(0<=y0 && y0<Ly);
    assert(0<=x1 && x1<Lx);
    assert(0<=y1 && y1<Ly);
    assert(0<=x2 && x2<Lx);
    assert(0<=y2 && y2<Ly);

    Double res = 0.0;
    int counter = 0;

    for(Idx dx=0; dx<Lx; dx++){
      for(Idx dy=0; dy<Ly; dy++){
        // if( !is_site(x,y) ) continue;
        const Idx x0p = (dx+x0)%Lx;
        const Idx y0p = (dy+y0)%Ly;
        const Idx x1p = (dx+x1)%Lx;
        const Idx y1p = (dy+y1)%Ly;
        const Idx x2p = (dx+x2)%Lx;
        const Idx y2p = (dy+y2)%Ly;
        if( !is_site(x0p,y0p) || !is_site(x1p,y1p) || !is_site(x2p,y2p) ) continue;
        // if( !is_site(x1p,y1p) || !is_site(x2p,y2p) ) continue;
        res += TyyF(x0p,y0p) * (*this)(x1p,y1p) * (*this)(x2p,y2p);
        counter++;
      }}

    res /= counter;
    return res;
  }


  std::vector<Double> Tyy_ss_corr( const Idx x1, const Idx y1,
                                   const Idx x2, const Idx y2 ) const {
    std::vector<Double> corr(N, 0.0);

#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
    // #pragma omp parallel for num_threads(nparallel) schedule(static)
#endif
    for(Idx x0=0; x0<Lx; x0++){
      for(Idx y0=0; y0<Ly; y0++){
        corr[idx(x0,y0)] = Tyy_ss( x0, y0, x1, y1, x2, y2 );
      }}
    return corr;
  }



  Double Txx_epseps( const Idx x0, const Idx y0,
                     const Idx x1, const Idx y1,
                     const Idx x2, const Idx y2 ) const {
    assert(0<=x0 && x0<Lx);
    assert(0<=y0 && y0<Ly);
    assert(0<=x1 && x1<Lx);
    assert(0<=y1 && y1<Ly);
    assert(0<=x2 && x2<Lx);
    assert(0<=y2 && y2<Ly);

    Double res = 0.0;
    int counter = 0;

    for(Idx dx=0; dx<Lx; dx++){
      for(Idx dy=0; dy<Ly; dy++){
        // if( !is_site(x,y) ) continue;
        const Idx x0p = (dx+x0)%Lx;
        const Idx y0p = (dy+y0)%Ly;
        const Idx x1p = (dx+x1)%Lx;
        const Idx y1p = (dy+y1)%Ly;
        const Idx x2p = (dx+x2)%Lx;
        const Idx y2p = (dy+y2)%Ly;
        if( !is_site(x0p,y0p) || !is_site(x1p,y1p) || !is_site(x2p,y2p) ) continue;
        // if( !is_site(x1p,y1p) || !is_site(x2p,y2p) ) continue;
        res += Txx(x0p,y0p) * eps(x1p,y1p) * eps(x2p,y2p);
        counter++;
      }}

    res /= counter;
    return res;
  }


  std::vector<Double> Txx_epseps_corr( const Idx x1, const Idx y1,
                                       const Idx x2, const Idx y2 ) const {
    std::vector<Double> corr(N, 0.0);

#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
    // #pragma omp parallel for num_threads(nparallel) schedule(static)
#endif
    for(Idx x0=0; x0<Lx; x0++){
      for(Idx y0=0; y0<Ly; y0++){
        corr[idx(x0,y0)] = Txx_epseps( x0, y0, x1, y1, x2, y2 );
      }}
    return corr;
  }


  Double Txy_epseps( const Idx x0, const Idx y0,
                       const Idx x1, const Idx y1,
                       const Idx x2, const Idx y2 ) const {
    assert(0<=x0 && x0<Lx);
    assert(0<=y0 && y0<Ly);
    assert(0<=x1 && x1<Lx);
    assert(0<=y1 && y1<Ly);
    assert(0<=x2 && x2<Lx);
    assert(0<=y2 && y2<Ly);

    Double res = 0.0;
    int counter = 0;

    for(Idx dx=0; dx<Lx; dx++){
      for(Idx dy=0; dy<Ly; dy++){
        // if( !is_site(x,y) ) continue;
        const Idx x0p = (dx+x0)%Lx;
        const Idx y0p = (dy+y0)%Ly;
        const Idx x1p = (dx+x1)%Lx;
        const Idx y1p = (dy+y1)%Ly;
        const Idx x2p = (dx+x2)%Lx;
        const Idx y2p = (dy+y2)%Ly;
        if( !is_site(x0p,y0p) || !is_site(x1p,y1p) || !is_site(x2p,y2p) ) continue;
        // if( !is_site(x1p,y1p) || !is_site(x2p,y2p) ) continue;
        res += Txy(x0p,y0p) * eps(x1p,y1p) * eps(x2p,y2p);
        counter++;
      }}

    res /= counter;
    return res;
  }


  std::vector<Double> Txy_epseps_corr( const Idx x1, const Idx y1,
                                       const Idx x2, const Idx y2 ) const {
    std::vector<Double> corr(N, 0.0);

#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
    // #pragma omp parallel for num_threads(nparallel) schedule(static)
#endif
    for(Idx x0=0; x0<Lx; x0++){
      for(Idx y0=0; y0<Ly; y0++){
        corr[idx(x0,y0)] = Txy_epseps( x0, y0, x1, y1, x2, y2 );
      }}
    return corr;
  }



//   Double Txy_epseps( const Idx dx1, const Idx dy1, const Idx dx2, const Idx dy2 ) const {
//     assert(0<=dx1 && dx1<Lx);
//     assert(0<=dy1 && dy1<Ly);
//     assert(0<=dx2 && dx2<Lx);
//     assert(0<=dy2 && dy2<Ly);

//     Double res = 0.0;
//     int counter = 0;

//     for(Idx x=0; x<Lx; x++){
//       for(Idx y=0; y<Ly; y++){
//         if( !is_site(x,y) ) continue;
//         const Idx x1 = (x+dx1)%Lx;
//         const Idx y1 = (y+dy1)%Ly;
//         const Idx x2 = (x+dx2)%Lx;
//         const Idx y2 = (y+dy2)%Ly;
//         if( !is_site(x1,y1) || !is_site(x2,y2) ) continue;
//         res += Txy(x,y) * eps(x1,y1) * eps(x2,y2);
//         counter++;
//       }}

//     res /= counter;
//     return res;
//   }


//   std::vector<Double> Txy_epseps_corr( const Idx dx1, const Idx dy1 ) const {
//     std::vector<Double> corr(N, 0.0);

// #ifdef _OPENMP
// #pragma omp parallel for num_threads(nparallel)
//     // #pragma omp parallel for num_threads(nparallel) schedule(static)
// #endif
//     for(Idx dx2=0; dx2<Lx; dx2++){
//       for(Idx dy2=0; dy2<Ly; dy2++){
//         corr[idx(dx2,dy2)] = Txy_epseps( dx1, dy1, dx2, dy2 );
//       }}
//     return corr;
//   }




  std::string print() const {
    std::stringstream ss;
    for(Idx y=Ly-1; y>=0; y--){
      for(Idx x= 0; x<Lx; x++) {
        ss << std::setw(5) << (*this)(x, y);
      }
      ss << std::endl;
    }
    return ss.str();
  }



  void ckpoint( const std::string& str ) const {
    std::ofstream of( str, std::ios::out | std::ios::binary | std::ios::trunc);
    if(!of) assert(false);

    int tmp = 0.0;
    for(Idx i=0; i<Lx*Ly; i++){
      tmp = (*this)[i];
      of.write( (char*) &tmp, sizeof(int) );
    }
    of.close();
  }

  void read( const std::string& str ) {
    std::ifstream ifs( str, std::ios::in | std::ios::binary );
    if(!ifs) assert(false);

    int tmp;
    for(Idx i=0; i<Lx*Ly; ++i){
      ifs.read((char*) &tmp, sizeof(int) );
      (*this)[i] = tmp;
    }
    ifs.close();
  }



};


void heatbath( Spin& s ){
  // omp not allowed
  for(Idx i=0; i<Lx*Ly; i++){
    if( !is_site(i) ) continue;

    double henv = 0.0;
    for(int mu=0; mu<SIX; mu++){
      if( !is_link(i,mu) ) continue;
      Idx j;
      cshift( j, i, mu );
      henv += Betac[mu%3] * s[j];
    }

    // const Double p = std::exp(2.0*beta_c*senv);
    const Double p = std::exp(2.0 * henv);
    const Double r = dist01();
    if( r<p/(1.0+p) ) s[i] = 1;
    else s[i] = -1;
  }
}


void wolff( Spin& s ){
  std::vector<int> is_cluster(Lx*Ly, 0);
  std::stack<Idx> stack_idx;

  Idx init = dist0N();
  while( !is_site(init) ) init = dist0N();

  is_cluster[init] = 1;
  stack_idx.push(init);

  while( stack_idx.size() != 0 ){

    const Idx p = stack_idx.top();
    stack_idx.pop();
    s[p] = -s[p]; // flip when visited

    for(int mu = 0; mu < SIX; mu++){
      if( !is_link(p,mu) ) continue;
      Idx q;
      cshift(q, p, mu);
      if( s[q] == s[p] || is_cluster[q]==1 ) continue; // s[x]*sR[y]<0 or y in c

      const Double r = dist01();
      if( r < std::exp(-2.0 * Betac[mu%3]) ) continue; // reject
      // if( r < std::exp(-2.0 * beta_c) ) continue; // reject

      is_cluster[q] = 1;
      stack_idx.push(q);
    }
  }
}



struct Scalar {
  Double v;

  Scalar()
    : v(0.0)
  {}

  Scalar( const Double v_ )
    : v(v_)
  {}

  Scalar( const Scalar& other )
    : v(other.v)
  {}

  void clear(){ v = 0.0; }

  Scalar& operator+=(const Scalar& rhs){
    v += rhs.v;
    return *this;
  }

  Scalar& operator+=(const Double& rhs){
    v += rhs;
    return *this;
  }

  Scalar& operator/=(const Double& rhs){
    v /= rhs;
    return *this;
  }

  std::string print() const {
    std::stringstream ss;
    ss << std::scientific << std::setprecision(15);
    ss << v;
    return ss.str();
  }

  void print(std::FILE* stream) const {
    fprintf( stream, "%0.15Le\t", v );
  }



};



struct Corr {
  std::vector<Double> v;

  Corr()
    : v(Lx*Ly, 0.0)
  {}

  Corr( const std::vector<Double> v_ )
    : v(v_)
  {}

  Corr( const Corr& other )
    : v(other.v)
  {}

  Double& operator()(const Idx x, const Idx y) { return v[idx(x,y)]; }
  Double operator()(const Idx x, const Idx y) const { return v[idx(x,y)]; }

  void clear(){ for(Idx i=0; i<Lx*Ly; i++) v[i] = 0.0; }

  Corr& operator+=(const Corr& rhs)
  {
    assert( rhs.v.size()==Lx*Ly );
    for(Idx i=0; i<Lx*Ly; i++) v[i] += rhs.v[i];
    return *this;
  }

  Corr& operator+=(const std::vector<Double>& rhs)
  {
    assert( rhs.size()==Lx*Ly );
    for(Idx i=0; i<Lx*Ly; i++) v[i] += rhs[i];
    return *this;
  }

  Corr& operator/=(const Double& rhs)
  {
    for(Idx i=0; i<Lx*Ly; i++) v[i] /= rhs;
    return *this;
  }


  // void print() const {
  //   for(int y=0; y<Ly; y++){
  //     for(int x=0; x<Lx; x++) {
  //       printf( "%0.15e\t", (*this)(x, y) );
  //     }
  //     printf("\n");
  //   }
  // }

  void print(std::FILE* stream) const {
    for(Idx y=0; y<Ly; y++){
      for(Idx x=0; x<Lx; x++) {
        fprintf( stream, "%0.15Le\t", (*this)(x, y) );
      }
      fprintf( stream, "\n");
    }
  }

  std::string print() const {
    std::stringstream ss;
    ss << std::scientific << std::setprecision(15);
    for(Idx y=0; y<Ly; y++){
      for(Idx x=0; x<Lx; x++) {
        ss << (*this)(x, y) << " ";
      }
      ss << std::endl;
    }
    return ss.str();
  }

};


template<typename T> // T needs to have: .clear, +=, /= defined
struct Obs {
  std::string description;
  int N;
  std::function<T(const Spin&)> f;

  T sum;
  int counter;

  // Obs() = delete;

  Obs
  (
   const std::string& description_,
   const int N_,
   const std::function<T(const Spin&)>& f_
   )
    : description(description_)
    , N(N_)
    , f(f_)
    , sum()
    , counter(0)
  {}

  void clear(){
    sum.clear();
    counter = 0;
  }

  void meas( const Spin& s ) {
    sum += f(s);
    counter++;
  }

  // T mean() const {
  //   T tmp(sum);
  //   tmp /= counter;
  //   return mean;
  // }

  void write_and_clear( const std::string& dir, const int label ){
    const std::string filename = dir + description + "_" + std::to_string(label) + ".dat";

    // std::ofstream of( filename, std::ios::out | std::ios::trunc );
    // if(!of) assert(false);
    // of << std::scientific << std::setprecision(15);
    // of << sum.print();
    // of.close();

    FILE *stream = fopen(filename.c_str(), "w");
    if (stream == NULL) assert(false);
    std::ofstream of( filename );
    sum /= counter;
    sum.print( stream );
    fclose( stream );

    clear();
  }

};




inline Corr ss_corr_wrapper( const Spin& s ){ return Corr( s.ss_corr() ); }
inline Scalar eps_1pt_wrapper( const Spin& s ){ return Scalar( s.eps_1pt() ); }
inline Corr epseps_corr_wrapper( const Spin& s ){ return Corr( s.epseps_corr() ); }



// struct ObsList {
//   std::vector<std::any> list;

//   // T sum;
//   int counter;

//   ObsList
//   ()
//     : counter(0)
//   {}

//   void meas(){
//     Obs tmp = list[0];
//     tmp.meas();
//   }
// };


