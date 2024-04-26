#pragma once

#include <random>
#include <stack>
#include <cassert>
#include <cmath>

#include <fstream>

#include <string>
#include <vector>
#include <array>
#include <functional>


using Idx = std::size_t;
using Double = long double;

constexpr Double m_pi() {return std::atan(1.0l)*4.0l;}

constexpr int TWO = 2;
constexpr int THREE = 3;
constexpr int SIX = 6;

constexpr int mult = 16;
constexpr Idx Lx = 3*2*mult; // 12
constexpr Idx Ly = 3*1*mult;

constexpr int nparallel = 12; //12


constexpr int nu = 1; // PP, PA, AA, AP

constexpr Double tt = 0.1; // 48.0l/(Lx*Ly*2.0l/3.0l);
// constexpr Double kappa[3] = {
//   2.0 - 4.0/3.0 * std::exp(-tt),
//   2.0/3.0 * std::exp(-tt),
//   2.0/3.0 * std::exp(-tt)
// };
constexpr Double kappa[3] = {
  2.0l/3.0l + 2.0l/3.0l*tt,
  2.0l/3.0l - 1.0l/3.0l*tt,
  2.0l/3.0l - 1.0l/3.0l*tt
};
const Double cos6 = std::cos(m_pi()/6.0l);

const Double Betac[3] = {
  std::atanh( cos6 * kappa[0] ),
  std::atanh( cos6 * kappa[1] ),
  std::atanh( cos6 * kappa[2] )
};

const Double B[3] = {
  cos6 / (1.0l - kappa[0]*kappa[0] * cos6 * cos6),
  cos6 / (1.0l - kappa[1]*kappa[1] * cos6 * cos6),
  cos6 / (1.0l - kappa[2]*kappa[2] * cos6 * cos6)
};
//

#ifndef _OPENMP
int omp_get_thread_num() { return 0; }
#endif



std::mt19937 gen;
std::uniform_real_distribution<Double> d01D(0.0l, 1.0l); // (1, 6);
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
    Double tot = 0.0l;
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

    Double res = 0.0l;
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
    std::vector<Double> corr(N, 0.0l);

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

    Double res = 0.0l;
    for(int mu=0; mu<SIX; mu++){
      if( !is_link(x,y,mu) ) continue;
      Idx xp, yp;
      cshift( xp, yp, x, y, mu );
      Double tmp = (*this)(x,y) * (*this)(xp,yp);
      tmp -= kappa[mu%3]*cos6;
      res += 0.5l*kappa[mu%3]*B[mu%3] * tmp;
    }
    res -= 0.25l;

    return res;
  }


  Double eps_1pt() const {
    Double res = 0.0l;

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

    Double res = 0.0l;
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
    std::vector<Double> corr(N, 0.0l);

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

    res -= 0.25l;

    return res;
  }


  Double TA( const Idx x, const Idx y ) const {
    assert(0<=x && x<Lx);
    assert(0<=y && y<Ly);

    Double res = 0.0l;
    int mu=0;

    int c = get_char(x,y);
    if(c==2) mu+=3;
    else if(c==1) assert(false);

    Idx xp, yp;
    cshift( xp, yp, x, y, mu );

    res += K(x, y, mu);
    res -= 0.5*( eps(x,y)+eps(xp,yp) ); // mu deriv

    return res;
  }

  Double TB( const Idx x, const Idx y ) const {
    assert(0<=x && x<Lx);
    assert(0<=y && y<Ly);

    Double res = 0.0l;
    int mu=1;

    int c = get_char(x,y);
    if(c==2) mu+=3;
    else if(c==1) assert(false);

    Idx xp, yp;
    cshift( xp, yp, x, y, mu );

    res += K(x, y, mu);
    res -= 0.5l*( eps(x,y)+eps(xp,yp) ); // mu deriv

    return res;
  }


  Double TC( const Idx x, const Idx y ) const {
    assert(0<=x && x<Lx);
    assert(0<=y && y<Ly);

    Double res = 0.0l;
    int mu=2;

    int c = get_char(x,y);
    if(c==2) mu+=3;
    else if(c==1) assert(false);

    Idx xp, yp;
    cshift( xp, yp, x, y, mu );

    res += K(x, y, mu);
    res -= 0.5l*( eps(x,y)+eps(xp,yp) );

    return res;
  }

  Double Txx( const Idx x, const Idx y ) const {
    const Double tA = TA(x,y);
    const Double tB = TB(x,y);
    const Double tC = TC(x,y);
    return 1.0l/3.0l * ( 2.0l*tA - tB - tC );
  }

  Double Txy( const Idx x, const Idx y ) const {
    const Double tB = TB(x,y);
    const Double tC = TC(x,y);
    return 1.0l/std::sqrt(3.0l) * ( tC - tB );
  }

  Double Tyy( const Idx x, const Idx y ) const {
    return -Txx(x,y);
  }

  Double Txx_1pt( ) const {
    Double res = 0.0l;
    int counter = 0;

    for(Idx x=0; x<Lx; x++){
      for(Idx y=0; y<Ly; y++){
        if( !is_site(x,y) ) continue;
        res += Txx( x, y );
        counter++;
      }}

    res /= counter;
    return res;
  }

  Double Txy_1pt( ) const {
    Double res = 0.0l;
    int counter = 0;

    for(Idx x=0; x<Lx; x++){
      for(Idx y=0; y<Ly; y++){
        if( !is_site(x,y) ) continue;
        res += Txy( x, y );
        counter++;
      }}

    res /= counter;
    return res;
  }

  Double Tyy_1pt( ) const {
    Double res = 0.0l;
    int counter = 0;

    for(Idx x=0; x<Lx; x++){
      for(Idx y=0; y<Ly; y++){
        if( !is_site(x,y) ) continue;
        res += Tyy( x, y );
        counter++;
      }}

    res /= counter;
    return res;
  }

  Double TxxTxx_corr( const Idx dx, const Idx dy ) const {
    assert(0<=dx && dx<Lx);
    assert(0<=dy && dy<Ly);

    Double res = 0.0l;
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

    Double res = 0.0l;
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
    std::vector<Double> corr(N, 0.0l);

#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(Idx dx=0; dx<Lx; dx++){
      for(Idx dy=0; dy<Ly; dy++){
        corr[idx(dx,dy)] = TxxTxx_corr( dx, dy );
      }}
    return corr;
  }


  std::vector<Double> TxxTxy_corr() const {
    std::vector<Double> corr(N, 0.0l);

#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(Idx dx=0; dx<Lx; dx++){
      for(Idx dy=0; dy<Ly; dy++){
        corr[idx(dx,dy)] = TxxTxy_corr( dx, dy );
      }}
    return corr;
  }


  Double Txx_ss( const Idx x0, const Idx y0,
                 const Idx x1, const Idx y1,
                 const Idx x2, const Idx y2 ) const {
    assert(0<=x0 && x0<Lx);
    assert(0<=y0 && y0<Ly);
    assert(0<=x1 && x1<Lx);
    assert(0<=y1 && y1<Ly);
    assert(0<=x2 && x2<Lx);
    assert(0<=y2 && y2<Ly);

    Double res = 0.0l;
    int counter = 0;

    for(Idx dx=0; dx<Lx; dx++){
      for(Idx dy=0; dy<Ly; dy++){
        const Idx x0p = (dx+x0)%Lx;
        const Idx y0p = (dy+y0)%Ly;
        const Idx x1p = (dx+x1)%Lx;
        const Idx y1p = (dy+y1)%Ly;
        const Idx x2p = (dx+x2)%Lx;
        const Idx y2p = (dy+y2)%Ly;
        if( !is_site(x0p,y0p) || !is_site(x1p,y1p) || !is_site(x2p,y2p) ) continue;
        res += Txx(x0p,y0p) * (*this)(x1p,y1p) * (*this)(x2p,y2p);
        counter++;
      }}

    res /= counter;
    return res;
  }


  std::vector<Double> Txx_ss_corr( const Idx x1, const Idx y1,
                                   const Idx x2, const Idx y2 ) const {
    std::vector<Double> corr(N, 0.0l);

#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
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

    Double res = 0.0l;
    int counter = 0;

    for(Idx dx=0; dx<Lx; dx++){
      for(Idx dy=0; dy<Ly; dy++){
        const Idx x0p = (dx+x0)%Lx;
        const Idx y0p = (dy+y0)%Ly;
        const Idx x1p = (dx+x1)%Lx;
        const Idx y1p = (dy+y1)%Ly;
        const Idx x2p = (dx+x2)%Lx;
        const Idx y2p = (dy+y2)%Ly;
        if( !is_site(x0p,y0p) || !is_site(x1p,y1p) || !is_site(x2p,y2p) ) continue;
        res += Txy(x0p,y0p) * (*this)(x1p,y1p) * (*this)(x2p,y2p);
        counter++;
      }}

    res /= counter;
    return res;
  }


  std::vector<Double> Txy_ss_corr( const Idx x1, const Idx y1,
                                   const Idx x2, const Idx y2 ) const {
    std::vector<Double> corr(N, 0.0l);

#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
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

    Double res = 0.0l;
    int counter = 0;

    for(Idx dx=0; dx<Lx; dx++){
      for(Idx dy=0; dy<Ly; dy++){
        const Idx x0p = (dx+x0)%Lx;
        const Idx y0p = (dy+y0)%Ly;
        const Idx x1p = (dx+x1)%Lx;
        const Idx y1p = (dy+y1)%Ly;
        const Idx x2p = (dx+x2)%Lx;
        const Idx y2p = (dy+y2)%Ly;
        if( !is_site(x0p,y0p) || !is_site(x1p,y1p) || !is_site(x2p,y2p) ) continue;
        res += Tyy(x0p,y0p) * (*this)(x1p,y1p) * (*this)(x2p,y2p);
        counter++;
      }}

    res /= counter;
    return res;
  }


  std::vector<Double> Tyy_ss_corr( const Idx x1, const Idx y1,
                                   const Idx x2, const Idx y2 ) const {
    std::vector<Double> corr(N, 0.0l);

#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
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

    Double res = 0.0l;
    int counter = 0;

    for(Idx dx=0; dx<Lx; dx++){
      for(Idx dy=0; dy<Ly; dy++){
        const Idx x0p = (dx+x0)%Lx;
        const Idx y0p = (dy+y0)%Ly;
        const Idx x1p = (dx+x1)%Lx;
        const Idx y1p = (dy+y1)%Ly;
        const Idx x2p = (dx+x2)%Lx;
        const Idx y2p = (dy+y2)%Ly;
        if( !is_site(x0p,y0p) || !is_site(x1p,y1p) || !is_site(x2p,y2p) ) continue;
        res += Txx(x0p,y0p) * eps(x1p,y1p) * eps(x2p,y2p);
        counter++;
      }}

    res /= counter;
    return res;
  }


  std::vector<Double> Txx_epseps_corr( const Idx x1, const Idx y1,
                                       const Idx x2, const Idx y2 ) const {
    std::vector<Double> corr(N, 0.0l);

#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
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

    Double res = 0.0l;
    int counter = 0;

    for(Idx dx=0; dx<Lx; dx++){
      for(Idx dy=0; dy<Ly; dy++){
        const Idx x0p = (dx+x0)%Lx;
        const Idx y0p = (dy+y0)%Ly;
        const Idx x1p = (dx+x1)%Lx;
        const Idx y1p = (dy+y1)%Ly;
        const Idx x2p = (dx+x2)%Lx;
        const Idx y2p = (dy+y2)%Ly;
        if( !is_site(x0p,y0p) || !is_site(x1p,y1p) || !is_site(x2p,y2p) ) continue;
        res += Txy(x0p,y0p) * eps(x1p,y1p) * eps(x2p,y2p);
        counter++;
      }}

    res /= counter;
    return res;
  }


  std::vector<Double> Txy_epseps_corr( const Idx x1, const Idx y1,
                                       const Idx x2, const Idx y2 ) const {
    std::vector<Double> corr(N, 0.0l);

#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(Idx x0=0; x0<Lx; x0++){
      for(Idx y0=0; y0<Ly; y0++){
        corr[idx(x0,y0)] = Txy_epseps( x0, y0, x1, y1, x2, y2 );
      }}
    return corr;
  }


  Double Tyy_epseps( const Idx x0, const Idx y0,
                       const Idx x1, const Idx y1,
                       const Idx x2, const Idx y2 ) const {
    assert(0<=x0 && x0<Lx);
    assert(0<=y0 && y0<Ly);
    assert(0<=x1 && x1<Lx);
    assert(0<=y1 && y1<Ly);
    assert(0<=x2 && x2<Lx);
    assert(0<=y2 && y2<Ly);

    Double res = 0.0l;
    int counter = 0;

    for(Idx dx=0; dx<Lx; dx++){
      for(Idx dy=0; dy<Ly; dy++){
        const Idx x0p = (dx+x0)%Lx;
        const Idx y0p = (dy+y0)%Ly;
        const Idx x1p = (dx+x1)%Lx;
        const Idx y1p = (dy+y1)%Ly;
        const Idx x2p = (dx+x2)%Lx;
        const Idx y2p = (dy+y2)%Ly;
        if( !is_site(x0p,y0p) || !is_site(x1p,y1p) || !is_site(x2p,y2p) ) continue;
        res += Tyy(x0p,y0p) * eps(x1p,y1p) * eps(x2p,y2p);
        counter++;
      }}

    res /= counter;
    return res;
  }


  std::vector<Double> Tyy_epseps_corr( const Idx x1, const Idx y1,
                                       const Idx x2, const Idx y2 ) const {
    std::vector<Double> corr(N, 0.0l);

#ifdef _OPENMP
#pragma omp parallel for num_threads(nparallel)
#endif
    for(Idx x0=0; x0<Lx; x0++){
      for(Idx y0=0; y0<Ly; y0++){
        corr[idx(x0,y0)] = Tyy_epseps( x0, y0, x1, y1, x2, y2 );
      }}
    return corr;
  }



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

    int tmp = 0.0l;
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

    Double henv = 0.0l;
    for(int mu=0; mu<SIX; mu++){
      if( !is_link(i,mu) ) continue;
      Idx j;
      cshift( j, i, mu );
      henv += Betac[mu%3] * s[j];
    }

    const Double p = std::exp(2.0l * henv);
    const Double r = dist01();
    if( r<p/(1.0l+p) ) s[i] = 1;
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
      if( r < std::exp(-2.0l * Betac[mu%3]) ) continue; // reject

      is_cluster[q] = 1;
      stack_idx.push(q);
    }
  }
}



struct Scalar {
  Double v;

  Scalar()
    : v(0.0l)
  {}

  Scalar( const Double v_ )
    : v(v_)
  {}

  Scalar( const Scalar& other )
    : v(other.v)
  {}

  void clear(){ v = 0.0l; }

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
    : v(Lx*Ly, 0.0l)
  {}

  Corr( const std::vector<Double> v_ )
    : v(v_)
  {}

  Corr( const Corr& other )
    : v(other.v)
  {}

  Double& operator()(const Idx x, const Idx y) { return v[idx(x,y)]; }
  Double operator()(const Idx x, const Idx y) const { return v[idx(x,y)]; }

  void clear(){ for(Idx i=0; i<Lx*Ly; i++) v[i] = 0.0l; }

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


