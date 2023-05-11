// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2022 Jorge Ferreira and Paulo Rodrigues.

// ::gyronimo:: is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// ::gyronimo:: is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with ::gyronimo::.  If not, see <https://www.gnu.org/licenses/>.

// @vmecloss.cc, this file is part of ::gyronimo::

// Command-line tool to print guiding-centre orbits in `VMEC` equilibria.
// External dependencies:
// - [argh](https://github.com/adishavit/argh), a minimalist argument handler.
// - [GSL](https://www.gnu.org/software/gsl), the GNU Scientific Library.
// - [boost](https://www.boost.org), the boost library.
// - [netcdf-c++4] (https://github.com/Unidata/netcdf-cxx4.git).


#include <cmath>
#include <numbers>
#include <iostream>
#include <random>
#include <chrono>
#include <type_traits>
#include <exception>
#include <assert.h>
#include <argh.h>
#include <mpi.h>
#include <highfive/H5Easy.hpp>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <highfive/H5PropertyList.hpp>
#include <boost/numeric/odeint.hpp>
#include <gyronimo/version.hh>
#include <gyronimo/core/codata.hh>
#include <gyronimo/parsers/parser_vmec.hh>
#include <gyronimo/fields/equilibrium_vmec.hh>
#include <gyronimo/interpolators/cubic_gsl.hh>
#include <gyronimo/dynamics/guiding_centre.hh>
#include <gyronimo/dynamics/odeint_adapter.hh>

void print_help() {
  int mpi_rank; MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  if (mpi_rank == 0 ) {
    std::cout << "vmecloss, powered by ::gyronimo::v"
        << gyronimo::version_major << "." << gyronimo::version_minor << ".\n";
    std::cout << "usage: vmecloss [options] vmec_netcdf_file\n";
    std::cout <<
        "reads a vmec output file, prints the required orbit to stdout.\n";
    std::cout << "options:\n";
    std::cout << "  -lref=val        Reference length (in si, default 1).\n";
    std::cout << "  -vref=val        Reference velocity (in si, default 1).\n";
    std::cout << "  -mass=val        Particle mass (in m_proton, default 4).\n";
    std::cout << "  -flux=val        Initial normalised toroidal flux (vmec, default 0.6).\n";
    std::cout << "  -energy=val      Energy value (in eV, default 3.5 MeV).\n";
    std::cout << "  -tfinal=val      Time limit (in lref/vref, default 0.1).\n";
    std::cout << "  -charge=val      Particle charge (in q_proton, default 2).\n";
    std::cout << "  -nsamples=val    Number of orbit samples (default 1000).\n";
    std::cout << "  -lambda=val      Lambda value, signed as v_par (default 0).\n";
    std::cout << "  -pitch=val       Cosine pitch angle, signed as v_par (default, use lambda).\n";
    std::cout << "  -nparticles=val  Number of test particles (default 1000).\n";
  }
  MPI_Finalize();
  std::exit(0);
}

// HihgFive/PHDF5 supporting routines
void check_collective_io(const HighFive::DataTransferProps& xfer_props) {
    auto mnccp = HighFive::MpioNoCollectiveCause(xfer_props);
    if (mnccp.getLocalCause() || mnccp.getGlobalCause()) {
        std::cout
            << "The operation was successful, but couldn't use collective MPI-IO. local cause: "
            << mnccp.getLocalCause() << " global cause:" << mnccp.getGlobalCause() << std::endl;
    }
}

// ODEInt observer object to store diagnostics at each particle event
using namespace gyronimo;
using namespace boost::numeric::odeint;
class orbit_observer {
 public:
  enum states { init = 1, confined = 2, axis = 3, lost = 4, end = 5 }; 
  orbit_observer(
      const equilibrium_vmec* e, guiding_centre* g, 
      size_t* store_int, double* store_dbl, 
      enum states state = init, const size_t id = 0 )
    : eq_pointer_(e), gc_pointer_(g),
      store_int_(store_int), store_dbl_(store_dbl),
      id_(id), state_(state), timestamp_(0.0),
      half_cell_size_(e->metric()->get_half_cell_size()) {
        MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank_);        
      };
  void operator()( guiding_centre::state& s, double t) {
    IR3 q = gc_pointer_->get_position(s);
    double vpp = gc_pointer_->get_vpp(s);
    IR3 c = eq_pointer_->metric()->transform2cylindrical(q);
    double R = c[IR3::u], phi = c[IR3::v], z = c[IR3::w];
    double x = R*std::cos(phi), y = R*std::sin(phi);
    if (q[IR3::u] >= 1.0 && state_ == states::confined) {
      state_ = states::lost;
      timestamp_ = t;
    }
    else if (q[IR3::u] <= 2.0*half_cell_size_ && state_ == states::confined ) {
      state_ = states::axis;
      timestamp_ = t;
    };
    if ( (state_ == states::init || state_ == states::lost || state_ == states::axis ) 
          && (timestamp_ == t) || state_ == states::end ) {
      size_t shift = 1;
      if ( state_ == states::init ) shift = 0;
      else if ( state_ == states::end ) shift = 2;
      (store_int_+3*shift)[0] = mpi_rank_;
      (store_int_+3*shift)[1] = id_;
      (store_int_+3*shift)[2] = state_;
      (store_dbl_+14*shift)[0] = timestamp_;
      (store_dbl_+14*shift)[1] = t;
      (store_dbl_+14*shift)[2] = q[IR3::u];
      (store_dbl_+14*shift)[3] = q[IR3::v];
      (store_dbl_+14*shift)[4] = q[IR3::w];
      (store_dbl_+14*shift)[5] = gc_pointer_->energy_perpendicular(s, t);
      (store_dbl_+14*shift)[6] = gc_pointer_->energy_parallel(s);
      (store_dbl_+14*shift)[7] = x;
      (store_dbl_+14*shift)[8] = y;
      (store_dbl_+14*shift)[9] = z;
      (store_dbl_+14*shift)[10] = R;
      (store_dbl_+14*shift)[11] = eq_pointer_->metric()->jacobian(q);
      (store_dbl_+14*shift)[12] = eq_pointer_->magnitude(q, t);
      (store_dbl_+14*shift)[13] = vpp;
    };
    if (state_ == states::lost || state_ == states::axis ) {
      s[3] = 0.0;
      gc_pointer_->nullify_mu();
    };
  };
  void set_state(const enum states state) { state_ = state; };
  void set_timestamp(double t) { timestamp_ = t; };
  const enum states get_state() const { return state_; };
 private:
  const IR3field_c1* eq_pointer_;
  const size_t id_;
  const double half_cell_size_;
  const int mpi_rank_;
  guiding_centre* gc_pointer_;
  double timestamp_;
  enum states state_;
  size_t* store_int_;
  double* store_dbl_;
};

int main(int argc, char* argv[]) {
  // Initialize MPI
  MPI_Init(&argc, &argv);
  // Get the number of processes in MPI_COMM_WORLD
  int mpi_size; MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  // Get the rank of this process in MPI_COMM_WORLD
  int mpi_rank; MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  // Parse command line arguments
  auto command_line = argh::parser(argv);
  if (command_line[{"h", "help"}]) print_help();
  if (!command_line(1)) {  // the 1st non-option argument is the mapping file.
    if ( mpi_rank == 0 )std::cout << "vmecloss: no vmec equilibrium file provided; -h for help.\n";
    MPI_Finalize();
    std::exit(1);
  }
  cubic_gsl_factory ifactory;
  parser_vmec parser(command_line[1]);
  metric_vmec g(&parser, &ifactory, true);
  equilibrium_vmec veq(&g, &ifactory);

// Reads parameters from the command line:
  double flux; command_line("flux", 0.6) >> flux;
  double mass; command_line("mass", 4.0) >> mass;
  double lref; command_line("lref", 1.0) >> lref;
  double vref; command_line("vref", 1.0) >> vref;
  double tfinal; command_line("tfinal", 0.00001) >> tfinal;
  double charge; command_line("charge", 2.0) >> charge;
  double energy; command_line("energy", 3.5e6) >> energy;
  double lambda; command_line("lambda", -1000.0) >> lambda;
  double pitch; command_line("pitch", -1000.0) >> pitch;
  size_t nparticles; command_line("nparticles", 1000) >> nparticles;
  size_t nsamples; command_line("nsamples", 1000) >> nsamples;

  double energy_ref = 0.5*codata::m_proton*mass*vref*vref;
  double energy_si = energy*codata::e;

  // The vmecloss workflow
  std::string fileoutname = "./vmecloss.h5";
  size_t nloops = (nparticles % mpi_size) ? nparticles / mpi_size + 1 : nparticles / mpi_size;  
  size_t id_shift = static_cast<size_t>(std::pow(10, std::ceil(std::log10(nloops))+1));
  // Prints output header:
  if (mpi_rank == 0) {
    std::cout << "# vmecloss, powered by ::gyronimo::v"
        << version_major << "." << version_minor << ".\n";
    std::cout << "# args: ";
    for(int i = 1; i < argc; i++) std::cout << argv[i] << " ";
    std::cout << std::endl
        << "# E_ref: " << energy_ref << " [J]"
            << " B_axis: " << veq.m_factor() << " [T]" << "\n";
                // << " mu_tilde: " << gc.mu_tilde() << "\n";
    std::cout <<
        "# nparticles = " << nparticles << ", id_shift = " << id_shift << "\n";
  };
  // Random distributions
  std::random_device rd;  
  std::mt19937_64 rand_generator(rd());
  std::uniform_real_distribution<> theta_distro(0.0, 2.0*std::numbers::pi);
  std::uniform_real_distribution<> zeta_distro(0.0, 2.0*std::numbers::pi/veq.get_nfp());
  std::uniform_real_distribution<> pitch_distro(-1.0, 1.0);

  if (mpi_rank == 0) {
    try {
      HighFive::File fileout(fileoutname, HighFive::File::Create | HighFive::File::Truncate);
      H5Easy::dump(fileout, "/__version__", 
              "::gyronimo::v"+std::to_string(gyronimo::version_major)+"."+std::to_string(gyronimo::version_minor));
      H5Easy::dump(fileout, "/parameters/B_axis", veq.m_factor());
      H5Easy::dump(fileout, "/parameters/nfp", veq.get_nfp());
      H5Easy::dump(fileout, "/parameters/flux",flux);
      H5Easy::dump(fileout, "/parameters/t_final", tfinal);
      H5Easy::dump(fileout, "/parameters/particle/charge", charge);
      H5Easy::dump(fileout, "/parameters/particle/mass",mass);
      H5Easy::dump(fileout, "/parameters/particle/energy", energy);
      H5Easy::dump(fileout, "/parameters/particle/pitch", pitch);
      H5Easy::dump(fileout, "/parameters/particle/energy_ref", energy_ref);
      H5Easy::dump(fileout, "/parameters/particle/L_ref", lref);
      H5Easy::dump(fileout, "/parameters/particle/velocity_ref", vref);
      H5Easy::dump(fileout, "/parameters/#particles/asked", nparticles);
      H5Easy::dump(fileout, "/parameters/#particles/modelled", nloops*mpi_size);
      H5Easy::dump(fileout, "/parameters/#cores", mpi_size);
      H5Easy::dump(fileout, "/parameters/id_shift", id_shift);
    } catch (std::exception& err) {
      std::cerr << err.what() << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    };
  };

  size_t store_int[3*nloops][3] = {0};
  double store_dbl[3*nloops][14] = {0};
  std::cerr.precision(16);
  std::cerr.setf(std::ios::scientific);
  MPI_Barrier(MPI_COMM_WORLD);  
  for (int icyl=0; icyl < nloops; icyl++ ) {
    size_t id = (mpi_rank+1)*id_shift+icyl+1;
    double theta = theta_distro(rand_generator);
    double zeta = zeta_distro(rand_generator);
    double vpp_sign_i;
    double lambda_i;
    double pitch_i;
    if (lambda > -10.0) { 
      vpp_sign_i = std::copysign(1.0, lambda);  // lambda carries vpp sign  
      pitch_i =  vpp_sign_i * std::sqrt(1.0 - lambda * (&veq)->magnitude({flux, theta, zeta}, 0.0));
      lambda_i = std::abs(lambda);  // once vpp sign is stored, lambda turns unsigned.  
    }
    else if (pitch >= -1.0 && pitch <= 1.0) {
      vpp_sign_i = std::copysign(1.0, pitch);  // pitch carries vpp sign.
      pitch_i = pitch;
      lambda_i = (1.0 - pitch*pitch) * std::abs(1.0/(&veq)->magnitude({flux, theta, zeta}, 0.0));
    }
    else {
      pitch_i = pitch_distro(rand_generator);
      vpp_sign_i = std::copysign(1.0, pitch_i);  // pitch carries vpp sign.
      lambda_i = (1.0 - pitch_i*pitch_i) * std::abs(1.0/(&veq)->magnitude({flux, theta, zeta}, 0.0));
    };
    guiding_centre gc(lref, vref, charge/mass, lambda_i*energy_si/energy_ref, &veq);
    // set initial state
    guiding_centre::state state = gc.generate_state(
        {flux, theta, zeta}, energy_si/energy_ref,
            (vpp_sign_i > 0 ?  guiding_centre::plus : guiding_centre::minus));
    // set observer
    orbit_observer observer(&veq, &gc,  
             store_int[3*icyl], store_dbl[3*icyl],
             orbit_observer::states::init, id);
    observer(state, 0.0); 
    observer.set_state(orbit_observer::states::confined);
    typedef guiding_centre::state state_type;
    typedef runge_kutta_cash_karp54<state_type> error_stepper_type;
    // typedef runge_kutta_fehlberg78<guiding_centre::state> error_stepper_type;
    double abs_err = 1.0e-10; double rel_err = 1.0e-6;
    integrate_adaptive(
        make_controlled( abs_err, rel_err, error_stepper_type() ), odeint_adapter(&gc),
        state, 0.0, tfinal, 1e-12, observer);
    if( observer.get_state() == orbit_observer::states::confined ) {
      observer.set_state(orbit_observer::states::end);
      observer.set_timestamp(tfinal);
      observer(state, tfinal); 
    };
  };
  MPI_Barrier(MPI_COMM_WORLD);  
  
  // Parallel HDF5 output
  try {
    HighFive::FileAccessProps fapl;
    fapl.add(HighFive::MPIOFileAccess{MPI_COMM_WORLD, MPI_INFO_NULL});
    fapl.add(HighFive::MPIOCollectiveMetadata{});
    HighFive::File fileout(fileoutname, HighFive::File::ReadWrite, fapl);
    auto orbits = fileout.createGroup("orbits");
    std::vector<size_t> id_dims{3 * mpi_size * nloops, 3ul}; 
    std::vector<size_t> state_dims{3 * mpi_size * nloops, 14ul};
    HighFive::DataSet id_dset = fileout.createDataSet<size_t>("/orbits/id", HighFive::DataSpace(id_dims));
    HighFive::DataSet state_dset = fileout.createDataSet<double>("/orbits/state", HighFive::DataSpace(state_dims));  
    auto xfer_props = HighFive::DataTransferProps{};
    xfer_props.add(HighFive::UseCollectiveIO{});
    std::vector<size_t> offset{std::size_t(3 * mpi_rank * nloops), 0ul};
    std::vector<size_t> id_count{3 * nloops, 3ul}; 
    std::vector<size_t> state_count{3 * nloops, 14ul};
    id_dset.select(offset, id_count).write(&store_int[0], xfer_props);
    state_dset.select(offset, state_count).write(&store_dbl[0], xfer_props);
    check_collective_io(xfer_props);
    fileout.flush();
  } catch (std::exception& err) {
      std::cerr << err.what() << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
  };
  MPI_Finalize();
  return 0;
}
