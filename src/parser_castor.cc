// ::gyronimo:: - gyromotion for the people, by the people -
// An object-oriented library for gyromotion applications in plasma physics.
// Copyright (C) 2021 Paulo Rodrigues.

// @parser_castor.cc

#include <fstream>
#include <gyronimo/core/error.hh>
#include <gyronimo/core/io_streaming.hh>
#include <gyronimo/parsers/parser_castor.hh>

namespace gyronimo {

//! Reads and parses a CASTOR ceig file by name.
parser_castor::parser_castor(const std::string &filename) {
  std::ifstream input_stream;
  input_stream.open(filename);
  if (input_stream.rdstate() != std::ios_base::goodbit)
    error(__func__, __FILE__, __LINE__, "cannot open input file.", 1);
  input_stream >> n_psi_ >> n_harm_ >> n_tor_ >> w_imag_ >> w_real_;
  m_.resize(n_harm_); input_stream >> m_;
  s_.resize(n_psi_);
  this->initialise_variable_chunk(input_stream, v1_real_, v1_imag_);
  this->initialise_variable_chunk(input_stream, v2_real_, v2_imag_);
  this->initialise_variable_chunk(input_stream, v3_real_, v3_imag_);

// Shamefully, ceig stores -i*A1 instead of A1:
  this->initialise_variable_chunk(input_stream, a1_imag_, a1_real_);
  a1_imag_ *= -1.0;

// Other components are ok:
  this->initialise_variable_chunk(input_stream, a2_real_, a2_imag_);
  this->initialise_variable_chunk(input_stream, a3_real_, a3_imag_);
  this->initialise_variable_chunk(input_stream, rho_real_, rho_imag_);
  this->initialise_variable_chunk(input_stream, t_real_, t_imag_);
}
 
void parser_castor::initialise_variable_chunk(
    std::ifstream& input_stream, narray_type& real, narray_type& imag) {
  real.resize(n_psi_*n_harm_);
  imag.resize(n_psi_*n_harm_);
  for (size_t m = 0; m < n_harm_; m++)
    for (size_t i = 0; i < n_psi_; i++)
      input_stream >> s_[i] >> real[i + m*n_psi_] >> imag[i + m*n_psi_];
}

} // end namespace gyronimo.
