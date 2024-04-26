#include <iostream>
#include <iomanip>
#include <filesystem>

#include <chrono>
#include <limits>

#include <omp.h>

#include "header.hpp"


int main( int argc, char *argv[] ){

  if (argc>1){
    mult = atoi(argv[1]);
  }
  Lx = 3*2*mult; // 12
  Ly = 3*1*mult;


  std::cout << std::scientific << std::setprecision(15);
  std::cout << "int = " << std::numeric_limits<int>::digits10 << std::endl;
  std::cout << "Idx = " << std::numeric_limits<Idx>::digits10 << std::endl;
  std::cout << "double = " << std::numeric_limits<double>::digits10 << std::endl;
  std::cout << "Double = " << std::numeric_limits<Double>::digits10 << std::endl;

#ifdef _OPENMP
  // omp_set_dynamic(0);
  omp_set_num_threads( nparallel );
#endif

  const std::string description = "Lx"+std::to_string(Lx)+"Ly"+std::to_string(Ly)+"nu"+std::to_string(nu);
  const std::string datadir = "./data_"+description+"/";
  std::filesystem::create_directories( datadir );
  const std::string configdir = "./config_"+description+"/";
  std::filesystem::create_directories( configdir );

  // routine
  const int Nbin = 1e4;
  const int binsize = 1e2; // 1e5

  const bool if_read = false;
  const bool if_write = true;

  const int Nheatbath = 4;
  const int Nwolff = 10;
  const int Nrepeat = 20;

  const unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  // std::cout << seed << std::endl;

  // init
  const int Nconf = Nbin*binsize;
  set_gen( seed );
  Spin s( Lx*Ly );

  int init_label = 0;
  if(if_read){
    std::filesystem::path dir{configdir};
    std::string file;
    for(auto const& dir_entry : std::filesystem::directory_iterator{dir}){
      file = dir_entry.path();
      const std::size_t c1 = file.find("ckpoint");
      const std::size_t c2 = file.find(".dat");
      const int tmp = std::stoi( file.substr(c1+7, c2-c1-7) );
      if(init_label < tmp) init_label = tmp;
    }
    const std::filesystem::path filepath = static_cast<std::string>(configdir)+"ckpoint"+std::to_string(init_label)+".dat";
    s.read( filepath );
  }
  else s.random();

  int ninit = init_label * binsize;

  {
    std::ofstream of( description+".log", std::ios::out | std::ios::app );
    if(!of) assert(false);
    of << "Lx = " << Lx << std::endl
       << "Ly = " << Ly << std::endl
       << "nu = " << nu << std::endl
       << "beta = " << Betac[0] << ", " << Betac[1] << ", " << Betac[2] << std::endl
       << "Nbin = " << Nbin << std::endl
       << "binsize = " << binsize << std::endl
       << "if_read= " << if_read << std::endl
       << "init_label = " << init_label << std::endl
       << "Nheatbath = " << Nheatbath << std::endl
       << "Nwolff = " << Nwolff << std::endl
       << "Nrepeat = " << Nrepeat << std::endl
       << "seed = " << seed << std::endl;
    of.close();
  }

  // observables
  //-----------
  Idx x1 = 0, y1 = 0, x2 = Lx/2, y2 = Ly/2;
  Obs<Corr> ss_corr( "ss_corr", binsize, [](const Spin& s0){ return Corr( s0.ss_corr() ); } );
  Obs<Scalar> eps_1pt( "eps_1pt", binsize, [](const Spin& s0){ return Scalar( s0.eps_1pt() ); } );
  Obs<Corr> epseps_corr( "epseps_corr", binsize, [](const Spin& s0){ return Corr( s0.epseps_corr() ); } );
  Obs<Scalar> Txx( "Txx", binsize, [](const Spin& s0){ return Scalar( s0.Txx_1pt() ); } );
  Obs<Scalar> Txy( "Txy", binsize, [](const Spin& s0){ return Scalar( s0.Txy_1pt() ); } );
  // Obs<Scalar> Tyy( "Tyy", binsize, [](const Spin& s0){ return Scalar( s0.Tyy_1pt() ); } );
  // Obs<Scalar> TxxN( "TxxN", binsize, [](const Spin& s0){ return Scalar( s0.TxxN_1pt() ); } );
  // Obs<Scalar> TyyN( "TyyN", binsize, [](const Spin& s0){ return Scalar( s0.TyyN_1pt() ); } );
  Obs<Corr> TxxTxx( "TxxTxx", binsize, [](const Spin& s0){ return Corr( s0.TxxTxx_corr() ); } );
  Obs<Corr> TxxTxy( "TxxTxy", binsize, [](const Spin& s0){ return Corr( s0.TxxTxy_corr() ); } );
  // Obs<Corr> Txx_eps( "Txx_eps", binsize, [](const Spin& s0){ return Corr( s0.Txx_eps_corr() ); } );
  // Obs<Corr> Txy_eps( "Txy_eps", binsize, [](const Spin& s0){ return Corr( s0.Txy_eps_corr() ); } );
  Obs<Corr> Txx_ss( "Txx_ss", binsize, [x1,y1,x2,y2](const Spin& s0){ return Corr( s0.Txx_ss_corr(x1, y1, x2, y2) ); } );
  Obs<Corr> Txy_ss( "Txy_ss", binsize, [x1,y1,x2,y2](const Spin& s0){ return Corr( s0.Txy_ss_corr(x1, y1, x2, y2) ); } );
  // Obs<Corr> Tyy_ss( "Tyy_ss", binsize, [x1,y1,x2,y2](const Spin& s0){ return Corr( s0.Tyy_ss_corr(x1, y1, x2, y2) ); } );
  Obs<Corr> Txx_epseps( "Txx_epseps", binsize, [x1,y1,x2,y2](const Spin& s0){ return Corr( s0.Txx_epseps_corr(x1, y1, x2, y2) ); } );
  Obs<Corr> Txy_epseps( "Txy_epseps", binsize, [x1,y1,x2,y2](const Spin& s0){ return Corr( s0.Txy_epseps_corr(x1, y1, x2, y2) ); } );
  // Obs<Corr> Tyy_epseps( "Txy_epseps", binsize, [x1,y1,x2,y2](const Spin& s0){ return Corr( s0.Tyy_epseps_corr(x1, y1, x2, y2) ); } );
  //----------------------------


  // run
  auto start = std::chrono::steady_clock::now();
  for(int n=ninit; n<Nconf; n++){

    // update routine
    for(int jj=0; jj<Nrepeat; jj++){
      for(int ii=0; ii<Nheatbath; ii++) heatbath( s );
      for(int ii=0; ii<Nwolff; ii++) wolff( s );
    }

    // measurement
    ss_corr.meas( s );
    eps_1pt.meas( s );
    epseps_corr.meas( s );
    Txx.meas( s );
    Txy.meas( s );
    // Tyy.meas( s );
    // TxxN.meas( s );
    // TyyN.meas( s );
    // //
    TxxTxx.meas( s );
    TxxTxy.meas( s );
    // Txx_eps.meas( s );
    // Txy_eps.meas( s );
    Txx_ss.meas( s );
    Txy_ss.meas( s );
    // Tyy_ss.meas( s );
    Txx_epseps.meas( s );
    Txy_epseps.meas( s );
    // Tyy_epseps.meas( s );
    //------

    // write out and clear
    if( (n+1)%binsize==0 && if_write ) {
      // -----------
      ss_corr.write_and_clear( datadir, (n+1)/binsize );
      eps_1pt.write_and_clear( datadir, (n+1)/binsize );
      epseps_corr.write_and_clear( datadir, (n+1)/binsize );
      //
      Txx.write_and_clear( datadir, (n+1)/binsize );
      Txy.write_and_clear( datadir, (n+1)/binsize );
      // Tyy.write_and_clear( datadir, (n+1)/binsize );
      // TxxN.write_and_clear( datadir, (n+1)/binsize );
      // TyyN.write_and_clear( datadir, (n+1)/binsize );
      //
      TxxTxx.write_and_clear( datadir, (n+1)/binsize );
      TxxTxy.write_and_clear( datadir, (n+1)/binsize );
      // Txx_eps.write_and_clear( datadir, (n+1)/binsize );
      // Txy_eps.write_and_clear( datadir, (n+1)/binsize );
      Txx_ss.write_and_clear( datadir, (n+1)/binsize );
      Txy_ss.write_and_clear( datadir, (n+1)/binsize );
      // Tyy_ss.write_and_clear( datadir, (n+1)/binsize );
      Txx_epseps.write_and_clear( datadir, (n+1)/binsize );
      Txy_epseps.write_and_clear( datadir, (n+1)/binsize );
      // Tyy_epseps.write_and_clear( datadir, (n+1)/binsize );
      // -------
      std::clog << "iter: " << n+1 << std::endl;

      const int label = (n+1)/binsize;
      const std::string filepath = configdir+"ckpoint"+std::to_string(label)+".dat";
      s.ckpoint( filepath );
      const int label2 = label-1;
      const std::string filepath2 = configdir+"ckpoint"+std::to_string(label2)+".dat";
      std::filesystem::remove( filepath2 );

      const auto end = std::chrono::steady_clock::now();
      const std::chrono::duration<double> elapsed_seconds{end - start};
      std::cout << elapsed_seconds.count() << " sec" << std::endl;
      start = std::chrono::steady_clock::now();
    }

  } // end for n


  return 0;
}

