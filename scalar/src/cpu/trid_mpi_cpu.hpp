/*
 * Open source copyright declaration based on BSD open source template:
 * http://www.opensource.org/licenses/bsd-license.php
 *
 * This file is part of the scalar-tridiagonal solver distribution.
 *
 * Copyright (c) 2015, Endre László and others. Please see the AUTHORS file in
 * the main source directory for a full list of copyright holders.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * The name of Endre László may not be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY Endre László ''AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL Endre László BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

// Written by Endre Laszlo, University of Oxford, endre.laszlo@oerc.ox.ac.uk, 2013-2014 

#ifndef __TRID_MPI_CPU_HPP
#define __TRID_MPI_CPU_HPP

#include "trid_simd.h"
#include "transpose.hpp"
#include "math.h"

#define N_MPI_MAX 128

#ifdef __MIC__
#  if FPPREC == 0
#    define LOAD(reg, array, n, N)                                             \
      load(reg, array, n, N);                                                  \
      transpose16x16_intrinsic(reg);
#    define STORE(array, reg, n, N)                                            \
      transpose16x16_intrinsic(reg);                                           \
      store(array, reg, n, N);
#  elif FPPREC == 1
#    define LOAD(reg, array, n, N)                                             \
      load(reg, array, n, N);                                                  \
      transpose8x8_intrinsic(reg);
#    define STORE(array, reg, n, N)                                            \
      transpose8x8_intrinsic(reg);                                             \
      store(array, reg, n, N);
#  endif
#elif __AVX__
#  if FPPREC == 0
#    define LOAD(reg, array, n, N)                                             \
      load(reg, array, n, N);                                                  \
      transpose8x8_intrinsic(reg);
#    define STORE(array, reg, n, N)                                            \
      transpose8x8_intrinsic(reg);                                             \
      store(array, reg, n, N);
#  elif FPPREC == 1
#    define LOAD(reg, array, n, N)                                             \
      load(reg, array, n, N);                                                  \
      transpose4x4_intrinsic(reg);
#    define STORE(array, reg, n, N)                                            \
      transpose4x4_intrinsic(reg);                                             \
      store(array, reg, n, N);
#  endif
#endif
      
inline void load(SIMD_REG *__restrict__ dst, const FP *__restrict__ src, int n,
                 int pad) {
  __assume_aligned(src, SIMD_WIDTH);
  __assume_aligned(dst, SIMD_WIDTH);
  for (int i = 0; i < SIMD_VEC; i++) {
    dst[i] = *(SIMD_REG *)&(src[i * pad + n]);
  }
}

inline void store(FP *__restrict__ dst, SIMD_REG *__restrict__ src, int n,
                  int pad) {
  __assume_aligned(src, SIMD_WIDTH);
  __assume_aligned(dst, SIMD_WIDTH);
  for (int i = 0; i < SIMD_VEC; i++) {
    *(SIMD_REG *)&(dst[i * pad + n]) = src[i];
  }
}

////
//// Thomas solver for reduced system
////
//template<typename REAL>
//inline void pcr_on_reduced(void** a, void** c, void** d, int N, int stride) {
//  
//  REAL b;
//  int s=1;
//  REAL a2_array[N_MPI_MAX], c2_array[N_MPI_MAX], d2_array[N_MPI_MAX];
//  REAL *a2 = a2_array;
//  REAL *c2 = c2_array;
//  REAL *d2 = d2_array;
//
//  REAL *a_tmp, *c_tmp, *d_tmp;
//
//  int P = ceil(log2((double)N));
//
//  for(int i=0; i<N; i+=2) {
//    b       = static_cast<REAL>(1.0) - a[i+1] * c[i] - c[i+1] * a[i+2];
//    b       = static_cast<REAL>(1.0) / b;
//    d2[i+1] =   b * (d[i+1] - a[i+1] * d[i] - c[i+1] * d[i+2]);
//    a2[i+1] = - b * a[i+1] * a[i];
//    c2[i+1] = - b * c[i+1] * c[i+2];
//  }
//
//  for(int p=1; p<P; p++) {
//    int s = 1 << p;
//    for(int i=0; i<N; i+=2) {
//      b       = static_cast<REAL>(1.0) - a[i+1] * c[i+s+1] - c[i+1] * a[i+s+2];
//      b       = static_cast<REAL>(1.0) / b;
//      d2[i+1] = 1;//  b * (d[i+1] - a[i+1] * d[i-s+1] - c[i+1] * d[i+s+1]);
//      a2[i+1] = - b * a[i+1] * a[i+1-s];
//      c2[i+1] = - b * c[i+1] * c[i+1+s];
//    }    
//    a_tmp = a2;
//    a2    = a;
//    a     = a_tmp;
//    c_tmp = c2;
//    c2    = c;
//    c     = c_tmp;
//    d_tmp = d2;
//    d2    = d;
//    d     = d_tmp;
//  }
//}


//
// Thomas solver for reduced system
//
template<typename REAL>
inline void thomas_on_reduced(
    const REAL* __restrict__ aa_r, 
    const REAL* __restrict__ cc_r, 
          REAL* __restrict__ dd_r, 
    int N, 
    int stride) {
  int   i, ind = 0;
  FP aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];
  //
  // forward pass
  //
  bb    = static_cast<REAL>(1.0);
  cc    = cc_r[0];
  dd    = dd_r[0];
  c2[0] = cc;
  d2[0] = dd;

  for(i=1; i<N; i++) {
    ind   = ind + stride;
    aa    = aa_r[ind];
    bb    = static_cast<REAL>(1.0) - aa*cc;
    dd    = dd_r[ind] - aa*dd;
    bb    = static_cast<REAL>(1.0)/bb;
    cc    = bb*cc_r[ind];
    dd    = bb*dd;
    c2[i] = cc;
    d2[i] = dd;
  }
  //
  // reverse pass
  //
  dd_r[ind] = dd;
  for(i=N-2; i>=0; i--) {
    ind    = ind - stride;
    dd     = d2[i] - c2[i]*dd;
    dd_r[ind] = dd;
  }
}

//
// Modified Thomas forwards pass
//
template<typename REAL>
inline void thomas_forward(
    const REAL *__restrict__ a, 
    const REAL *__restrict__ b, 
    const REAL *__restrict__ c, 
    const REAL *__restrict__ d, 
    const REAL *__restrict__ u, 
          REAL *__restrict__ aa, 
          REAL *__restrict__ cc, 
          REAL *__restrict__ dd, 
    int N, 
    int stride) {

  REAL bbi;
  int ind = 0;

  if(N >=2) {
    // Start lower off-diagonal elimination
    for(int i=0; i<2; i++) {
      ind = i * stride;
      bbi   = static_cast<REAL>(1.0) / b[ind];
      //dd[i] = 66;//d[i] * bbi;
      dd[ind] = d[ind] * bbi;
      aa[ind] = a[ind] * bbi;
      cc[ind] = c[ind] * bbi;
    }
    if(N >=3 ) {
      // Eliminate lower off-diagonal
      for(int i=2; i<N; i++) {
        ind = i * stride;
        bbi   = static_cast<REAL>(1.0) / (b[ind] - a[ind] * cc[ind - stride]); 
        //dd[i] = 77;//(d[i] - a[i]*dd[i-1]) * bbi;
        dd[ind] = (d[ind] - a[ind]*dd[ind - stride]) * bbi;
        aa[ind] = (     - a[ind]*aa[ind - stride]) * bbi;
        cc[ind] =                 c[ind]  * bbi;
      }
      // Eliminate upper off-diagonal
      for(int i=N-3; i>0; i--) {
        ind = i * stride;
        //dd[i] = 88;//dd[i] - cc[i]*dd[i+1];
        dd[ind] = dd[ind] - cc[ind]*dd[ind + stride];
        aa[ind] = aa[ind] - cc[ind]*aa[ind + stride];
        cc[ind] =       - cc[ind]*cc[ind + stride];
      }
      bbi = static_cast<REAL>(1.0) / (static_cast<REAL>(1.0) - cc[0]*aa[stride]);
      dd[0] =  bbi * ( dd[0] - cc[0]*dd[stride] );
      aa[0] =  bbi *   aa[0];
      cc[0] =  bbi * (       - cc[0]*cc[stride] );
    }
    
  }
  else {
    exit(-1);
  }
}

template<typename REAL>
inline void thomas_forward_transpose_vec(
    const REAL *__restrict__ a, 
    const REAL *__restrict__ b, 
    const REAL *__restrict__ c, 
    const REAL *__restrict__ d, 
    const REAL *__restrict__ u, 
          REAL *__restrict__ aa, 
          REAL *__restrict__ cc, 
          REAL *__restrict__ dd, 
    int N,
    int sys_pad,
    int stride) {
  
  SIMD_REG bv;
  
  SIMD_REG tmp;
  
  SIMD_REG a_reg[SIMD_VEC];
  SIMD_REG b_reg[SIMD_VEC];
  SIMD_REG c_reg[SIMD_VEC];
  SIMD_REG d_reg[SIMD_VEC];
  
  SIMD_REG aa_reg[N * SIMD_VEC];
  SIMD_REG cc_reg[N * SIMD_VEC];
  SIMD_REG dd_reg[N * SIMD_VEC];
  
  SIMD_REG ones = SIMD_SET1_P(1.0);
  SIMD_REG minusOnes = SIMD_SET1_P(-1.0);
  
  int n = 0;
  
  LOAD(a_reg, a, n, sys_pad);
  LOAD(b_reg, b, n, sys_pad);
  LOAD(c_reg, c, n, sys_pad);
  LOAD(d_reg, d, n, sys_pad);
  
  for(int i = 0; i < 2; i++) {
    bv = b_reg[i];
#if FPPREC == 0
    bv = SIMD_RCP_P(bv);
#elif FPPREC == 1
    bv = SIMD_DIV_P(ones, bv);
#endif
    
    a_reg[i] = SIMD_MUL_P(bv, a_reg[i]);
    c_reg[i] = SIMD_MUL_P(bv, c_reg[i]);
    d_reg[i] = SIMD_MUL_P(bv, d_reg[i]);

    aa_reg[n + i] = a_reg[i];
    cc_reg[n + i] = c_reg[i];
    dd_reg[n + i] = d_reg[i];
  }
  
  for(int i = 2; i < SIMD_VEC; i++) {
    // bbi   = static_cast<REAL>(1.0) / (b[ind] - a[ind] * cc[ind - stride]); 
    tmp = SIMD_MUL_P(a_reg[i], cc_reg[n + i - 1]);
    bv = SIMD_SUB_P(b_reg[i], tmp);
#if FPPREC == 0
    bv = SIMD_RCP_P(bv);
#elif FPPREC == 1
    bv = SIMD_DIV_P(ones, bv);
#endif
    
    // dd[ind] = (d[ind] - a[ind]*dd[ind - stride]) * bbi;
    tmp = SIMD_MUL_P(a_reg[i], dd_reg[n + i - 1]);
    d_reg[i] = SIMD_SUB_P(d_reg[i], tmp);
    d_reg[i] = SIMD_MUL_P(d_reg[i], bv);
    dd_reg[n + i] = d_reg[i];
    
    // aa[ind] = (     - a[ind]*aa[ind - stride]) * bbi;
    a_reg[i] = SIMD_MUL_P(a_reg[i], aa_reg[n + i - 1]);
    a_reg[i] = SIMD_MUL_P(a_reg[i], minusOnes);
    a_reg[i] = SIMD_MUL_P(a_reg[i], bv);
    aa_reg[n + i] = a_reg[i];
    
    // cc[ind] =                 c[ind]  * bbi;
    c_reg[i] = SIMD_MUL_P(c_reg[i], bv);
    cc_reg[n + i] = c_reg[i];
  }
  
  for(n = SIMD_VEC; n < (N / SIMD_VEC) * SIMD_VEC; n += SIMD_VEC) {
    LOAD(a_reg, a, n, sys_pad);
    LOAD(b_reg, b, n, sys_pad);
    LOAD(c_reg, c, n, sys_pad);
    LOAD(d_reg, d, n, sys_pad);
    
    for(int i = 0; i < SIMD_VEC; i++) {
      // bbi   = static_cast<REAL>(1.0) / (b[ind] - a[ind] * cc[ind - stride]); 
      tmp = SIMD_MUL_P(a_reg[i], cc_reg[n + i - 1]);
      bv = SIMD_SUB_P(b_reg[i], tmp);
#if FPPREC == 0
      bv = SIMD_RCP_P(bv);
#elif FPPREC == 1
      bv = SIMD_DIV_P(ones, bv);
#endif
      
      // dd[ind] = (d[ind] - a[ind]*dd[ind - stride]) * bbi;
      tmp = SIMD_MUL_P(a_reg[i], dd_reg[n + i - 1]);
      d_reg[i] = SIMD_SUB_P(d_reg[i], tmp);
      d_reg[i] = SIMD_MUL_P(d_reg[i], bv);
      dd_reg[n + i] = d_reg[i];
      
      // aa[ind] = (     - a[ind]*aa[ind - stride]) * bbi;
      a_reg[i] = SIMD_MUL_P(a_reg[i], aa_reg[n + i - 1]);
      a_reg[i] = SIMD_MUL_P(a_reg[i], minusOnes);
      a_reg[i] = SIMD_MUL_P(a_reg[i], bv);
      aa_reg[n + i] = a_reg[i];
      
      // cc[ind] =                 c[ind]  * bbi;
      c_reg[i] = SIMD_MUL_P(c_reg[i], bv);
      cc_reg[n + i] = c_reg[i];
    }
  }
  
  if(N != sys_pad) {
    n = (N / SIMD_VEC) * SIMD_VEC;
    LOAD(a_reg, a, n, sys_pad);
    LOAD(b_reg, b, n, sys_pad);
    LOAD(c_reg, c, n, sys_pad);
    LOAD(d_reg, d, n, sys_pad);
    for(int i = 0; (n + i) < N; i++) {
      // bbi   = static_cast<REAL>(1.0) / (b[ind] - a[ind] * cc[ind - stride]); 
      tmp = SIMD_MUL_P(a_reg[i], cc_reg[n + i - 1]);
      bv = SIMD_SUB_P(b_reg[i], tmp);
#if FPPREC == 0
      bv = SIMD_RCP_P(bv);
#elif FPPREC == 1
      bv = SIMD_DIV_P(ones, bv);
#endif
      
      // dd[ind] = (d[ind] - a[ind]*dd[ind - stride]) * bbi;
      tmp = SIMD_MUL_P(a_reg[i], dd_reg[n + i - 1]);
      d_reg[i] = SIMD_SUB_P(d_reg[i], tmp);
      d_reg[i] = SIMD_MUL_P(d_reg[i], bv);
      dd_reg[n + i] = d_reg[i];
      
      // aa[ind] = (     - a[ind]*aa[ind - stride]) * bbi;
      a_reg[i] = SIMD_MUL_P(a_reg[i], aa_reg[n + i - 1]);
      a_reg[i] = SIMD_MUL_P(a_reg[i], minusOnes);
      a_reg[i] = SIMD_MUL_P(a_reg[i], bv);
      aa_reg[n + i] = a_reg[i];
      
      // cc[ind] =                 c[ind]  * bbi;
      c_reg[i] = SIMD_MUL_P(c_reg[i], bv);
      cc_reg[n + i] = c_reg[i];
    }
    
    STORE(aa, a_reg, n, sys_pad);
    STORE(cc, c_reg, n, sys_pad);
    STORE(dd, d_reg, n, sys_pad);
  } else {
    n = ((N / SIMD_VEC) * SIMD_VEC) - SIMD_VEC;
    STORE(aa, a_reg, n, sys_pad);
    STORE(cc, c_reg, n, sys_pad);
    STORE(dd, d_reg, n, sys_pad);
  }
  
  int N_3 = (N - 3) % SIMD_VEC;
  n = ((N - 3) / SIMD_VEC) * SIMD_VEC;
  LOAD(a_reg, aa, n, sys_pad);
  LOAD(c_reg, cc, n, sys_pad);
  LOAD(d_reg, dd, n, sys_pad);
  
  for(int i = N_3; i >= 0; i--) {
    // dd[ind] = dd[ind] - cc[ind]*dd[ind + stride];
    tmp = SIMD_MUL_P(cc_reg[n + i], dd_reg[n + i + 1]);
    d_reg[i] = SIMD_SUB_P(dd_reg[n + i], tmp);
    dd_reg[n + i] = d_reg[i];
    
    // aa[ind] = aa[ind] - cc[ind]*aa[ind + stride];
    tmp = SIMD_MUL_P(cc_reg[n + i], aa_reg[n + i + 1]);
    a_reg[i] = SIMD_SUB_P(aa_reg[n + i], tmp);
    aa_reg[n + i] = a_reg[i];
    
    // cc[ind] =       - cc[ind]*cc[ind + stride];
    tmp = SIMD_MUL_P(cc_reg[n + i], cc_reg[n + i + 1]);
    c_reg[i] = SIMD_MUL_P(minusOnes, tmp);
    cc_reg[n + i] = c_reg[i];
  }
  
  STORE(aa, a_reg, n, sys_pad);
  STORE(cc, c_reg, n, sys_pad);
  STORE(dd, d_reg, n, sys_pad);
  
  for(n -= SIMD_VEC; n > 0; n -= SIMD_VEC) {
    for(int i = SIMD_VEC - 1; i >= 0; i--) {
      // dd[ind] = dd[ind] - cc[ind]*dd[ind + stride];
      tmp = SIMD_MUL_P(cc_reg[n + i], dd_reg[n + i + 1]);
      d_reg[i] = SIMD_SUB_P(dd_reg[n + i], tmp);
      dd_reg[n + i] = d_reg[i];
      
      // aa[ind] = aa[ind] - cc[ind]*aa[ind + stride];
      tmp = SIMD_MUL_P(cc_reg[n + i], aa_reg[n + i + 1]);
      a_reg[i] = SIMD_SUB_P(aa_reg[n + i], tmp);
      aa_reg[n + i] = a_reg[i];
      
      // cc[ind] =       - cc[ind]*cc[ind + stride];
      tmp = SIMD_MUL_P(cc_reg[n + i], cc_reg[n + i + 1]);
      c_reg[i] = SIMD_MUL_P(minusOnes, tmp);
      cc_reg[n + i] = c_reg[i];
    }
    STORE(aa, a_reg, n, sys_pad);
    STORE(cc, c_reg, n, sys_pad);
    STORE(dd, d_reg, n, sys_pad);
  }
  
  for(int i = SIMD_VEC - 1; i > 0; i--) {
    // dd[ind] = dd[ind] - cc[ind]*dd[ind + stride];
    tmp = SIMD_MUL_P(cc_reg[i], dd_reg[i + 1]);
    d_reg[i] = SIMD_SUB_P(dd_reg[i], tmp);
    dd_reg[i] = d_reg[i];
    
    // aa[ind] = aa[ind] - cc[ind]*aa[ind + stride];
    tmp = SIMD_MUL_P(cc_reg[i], aa_reg[i + 1]);
    a_reg[i] = SIMD_SUB_P(aa_reg[i], tmp);
    aa_reg[i] = a_reg[i];
    
    // cc[ind] =       - cc[ind]*cc[ind + stride];
    tmp = SIMD_MUL_P(cc_reg[i], cc_reg[i + 1]);
    c_reg[i] = SIMD_MUL_P(minusOnes, tmp);
    cc_reg[i] = c_reg[i];
  }
  
  // bbi = static_cast<REAL>(1.0) / (static_cast<REAL>(1.0) - cc[0]*aa[stride]);
  bv = SIMD_MUL_P(cc_reg[0], aa_reg[1]);
  bv = SIMD_SUB_P(ones, bv);
#if FPPREC == 0
  bv = SIMD_RCP_P(bv);
#elif FPPREC == 1
  bv = SIMD_DIV_P(ones, bv);
#endif
  // dd[0] =  bbi * ( dd[0] - cc[0]*dd[stride] );
  tmp = SIMD_MUL_P(cc_reg[0], dd_reg[1]);
  tmp = SIMD_SUB_P(dd_reg[0], tmp);
  d_reg[0] = SIMD_MUL_P(tmp, bv);
  
  // aa[0] =  bbi *   aa[0];
  a_reg[0] = SIMD_MUL_P(aa_reg[0], bv);
  
  // cc[0] =  bbi * (       - cc[0]*cc[stride] );
  tmp = SIMD_MUL_P(cc_reg[0], cc_reg[1]);
  tmp = SIMD_MUL_P(minusOnes, tmp);
  c_reg[0] = SIMD_MUL_P(tmp, bv);
  
  STORE(aa, a_reg, n, sys_pad);
  STORE(cc, c_reg, n, sys_pad);
  STORE(dd, d_reg, n, sys_pad);
}

//
// Modified Thomas backward pass
//
template<typename REAL>
inline void thomas_backward(
    const REAL *__restrict__ aa, 
    const REAL *__restrict__ cc, 
    const REAL *__restrict__ dd, 
          REAL *__restrict__ d, 
    int N, 
    int stride) {

  int ind = 0;
  
  d[0] = dd[0];
  #pragma ivdep
  for (int i=1; i<N-1; i++) {
    ind = i * stride;
    //d[i] = dd[i];//dd[i] - aa[i]*dd[0] - cc[i]*dd[N-1];
    d[ind] = dd[ind] - aa[ind]*dd[0] - cc[ind]*dd[(N-1) * stride];
  }
  d[(N-1) * stride] = dd[(N-1) * stride];
}

template<typename REAL>
inline void thomas_backwardInc(
    const REAL *__restrict__ aa, 
    const REAL *__restrict__ cc, 
    const REAL *__restrict__ dd, 
          REAL *__restrict__ u, 
    int N, 
    int stride) {

  int ind = 0;
  
  u[0] += dd[0];
  #pragma ivdep
  for (int i=1; i<N-1; i++) {
    ind = i * stride;
    //d[i] = dd[i];//dd[i] - aa[i]*dd[0] - cc[i]*dd[N-1];
    u[ind] += dd[ind] - aa[ind]*dd[0] - cc[ind]*dd[(N-1) * stride];
  }
  u[(N-1) * stride] += dd[(N-1) * stride];
}

template<typename REAL>
inline void thomas_backwardInc_transpose_vec(
    const REAL *__restrict__ aa, 
    const REAL *__restrict__ cc, 
    const REAL *__restrict__ dd, 
          REAL *__restrict__ u, 
    int N, 
    int sys_pad,
    int stride) {
  
  SIMD_REG a_reg[SIMD_VEC];
  SIMD_REG c_reg[SIMD_VEC];
  SIMD_REG d_reg[SIMD_VEC];
  SIMD_REG u_reg[SIMD_VEC];
  
  SIMD_REG dd_s, dd_e;
  
  REAL start[SIMD_VEC];
  REAL end[SIMD_VEC];
  
  for(int i = 0; i < SIMD_VEC; i++) {
    start[i] = dd[i * sys_pad];
    end[i] = dd[i * sys_pad + N - 1];
  }
  
  dd_s = *((SIMD_REG *)start);
  dd_e = *((SIMD_REG *)end);
  
  SIMD_REG tmp1, tmp2;
  
  int n = 0;
  
  LOAD(a_reg, aa, 0, sys_pad);
  LOAD(c_reg, cc, 0, sys_pad);
  LOAD(d_reg, dd, 0, sys_pad);
  LOAD(u_reg, u, 0, sys_pad);
  
  u_reg[0] = SIMD_ADD_P(u_reg[0], dd_s);
  
  for(int i = 1; i < SIMD_VEC; i++) {
    tmp1 = SIMD_MUL_P(c_reg[i], dd_e);
    tmp2 = SIMD_MUL_P(a_reg[i], dd_s);
    tmp2 = SIMD_SUB_P(d_reg[i], tmp2);
    tmp2 = SIMD_SUB_P(tmp2, tmp1);
    u_reg[i] = SIMD_ADD_P(u_reg[i], tmp2);
  }
  
  STORE(u, u_reg, 0, sys_pad);
  
  for(n = SIMD_VEC; n < ((N / SIMD_VEC) - 1) * SIMD_VEC; n += SIMD_VEC) {
    LOAD(a_reg, aa, n, sys_pad);
    LOAD(c_reg, cc, n, sys_pad);
    LOAD(d_reg, dd, n, sys_pad);
    LOAD(u_reg, u, n, sys_pad);
    for(int i = 0; i < SIMD_VEC; i++) {
      tmp1 = SIMD_MUL_P(c_reg[i], dd_e);
      tmp2 = SIMD_MUL_P(a_reg[i], dd_s);
      tmp2 = SIMD_SUB_P(d_reg[i], tmp2);
      tmp2 = SIMD_SUB_P(tmp2, tmp1);
      u_reg[i] = SIMD_ADD_P(u_reg[i], tmp2);
    }
    STORE(u, u_reg, n, sys_pad);
  }
  
  if(N != sys_pad) {
    n = ((N / SIMD_VEC) - 1) * SIMD_VEC;
    LOAD(a_reg, aa, n, sys_pad);
    LOAD(c_reg, cc, n, sys_pad);
    LOAD(d_reg, dd, n, sys_pad);
    LOAD(u_reg, u, n, sys_pad);
    for(int i = 0; i < SIMD_VEC; i++) {
      tmp1 = SIMD_MUL_P(c_reg[i], dd_e);
      tmp2 = SIMD_MUL_P(a_reg[i], dd_s);
      tmp2 = SIMD_SUB_P(d_reg[i], tmp2);
      tmp2 = SIMD_SUB_P(tmp2, tmp1);
      u_reg[i] = SIMD_ADD_P(u_reg[i], tmp2);
    }
    STORE(u, u_reg, n, sys_pad);
    
    n = (N / SIMD_VEC) * SIMD_VEC;
    LOAD(a_reg, aa, n, sys_pad);
    LOAD(c_reg, cc, n, sys_pad);
    LOAD(d_reg, dd, n, sys_pad);
    LOAD(u_reg, u, n, sys_pad);
    int i;
    for(i = 0; i < (n + i) < N - 1; i++) {
      tmp1 = SIMD_MUL_P(c_reg[i], dd_e);
      tmp2 = SIMD_MUL_P(a_reg[i], dd_s);
      tmp2 = SIMD_SUB_P(d_reg[i], tmp2);
      tmp2 = SIMD_SUB_P(tmp2, tmp1);
      u_reg[i] = SIMD_ADD_P(u_reg[i], tmp2);
    }
    u_reg[i] = SIMD_ADD_P(u_reg[i], dd_e);
    STORE(u, u_reg, n, sys_pad);
  } else {
    n = ((N / SIMD_VEC) - 1) * SIMD_VEC;
    LOAD(a_reg, aa, n, sys_pad);
    LOAD(c_reg, cc, n, sys_pad);
    LOAD(d_reg, dd, n, sys_pad);
    LOAD(u_reg, u, n, sys_pad);
    for(int i = 0; i < SIMD_VEC - 1; i++) {
      tmp1 = SIMD_MUL_P(c_reg[i], dd_e);
      tmp2 = SIMD_MUL_P(a_reg[i], dd_s);
      tmp2 = SIMD_SUB_P(d_reg[i], tmp2);
      tmp2 = SIMD_SUB_P(tmp2, tmp1);
      u_reg[i] = SIMD_ADD_P(u_reg[i], tmp2);
    }
    u_reg[SIMD_VEC - 1] = SIMD_ADD_P(u_reg[SIMD_VEC - 1], dd_e);
    
    STORE(u, u_reg, n, sys_pad);
  }
}

template<typename REAL>
inline void thomas_backward_transpose_vec(
    const REAL *__restrict__ aa, 
    const REAL *__restrict__ cc, 
    const REAL *__restrict__ dd, 
          REAL *__restrict__ d, 
    int N, 
    int sys_pad,
    int stride) {
  
  SIMD_REG a_reg[SIMD_VEC];
  SIMD_REG c_reg[SIMD_VEC];
  SIMD_REG d_reg[SIMD_VEC];
  
  SIMD_REG dd_s, dd_e;
  
  REAL start[SIMD_VEC];
  REAL end[SIMD_VEC];
  
  for(int i = 0; i < SIMD_VEC; i++) {
    start[i] = dd[i * sys_pad];
    end[i] = dd[i * sys_pad + N - 1];
  }
  
  dd_s = *((SIMD_REG *)start);
  dd_e = *((SIMD_REG *)end);
  
  SIMD_REG tmp1, tmp2;
  
  int n = 0;
  
  LOAD(a_reg, aa, 0, sys_pad);
  LOAD(c_reg, cc, 0, sys_pad);
  LOAD(d_reg, dd, 0, sys_pad);
  
  d_reg[0] = dd_s;
  
  for(int i = 1; i < SIMD_VEC; i++) {
    tmp1 = SIMD_MUL_P(c_reg[i], dd_e);
    tmp2 = SIMD_MUL_P(a_reg[i], dd_s);
    tmp2 = SIMD_SUB_P(d_reg[i], tmp2);
    d_reg[i] = SIMD_SUB_P(tmp2, tmp1);
  }
  
  STORE(d, d_reg, 0, sys_pad);
  
  for(n = SIMD_VEC; n < (N / SIMD_VEC) * SIMD_VEC; n += SIMD_VEC) {
    LOAD(a_reg, aa, n, sys_pad);
    LOAD(c_reg, cc, n, sys_pad);
    LOAD(d_reg, dd, n, sys_pad);
    for(int i = 0; i < SIMD_VEC; i++) {
      tmp1 = SIMD_MUL_P(c_reg[i], dd_e);
      tmp2 = SIMD_MUL_P(a_reg[i], dd_s);
      tmp2 = SIMD_SUB_P(d_reg[i], tmp2);
      d_reg[i] = SIMD_SUB_P(tmp2, tmp1);
      
    }
    STORE(d, d_reg, n, sys_pad);
  }
  
  if(N != sys_pad) {
    n = (N / SIMD_VEC) * SIMD_VEC;
    LOAD(a_reg, aa, n, sys_pad);
    LOAD(c_reg, cc, n, sys_pad);
    LOAD(d_reg, dd, n, sys_pad);
    int i;
    for(i = 0; i < (n + i) < N - 1; i++) {
      tmp1 = SIMD_MUL_P(c_reg[i], dd_e);
      tmp2 = SIMD_MUL_P(a_reg[i], dd_s);
      tmp2 = SIMD_SUB_P(d_reg[i], tmp2);
      d_reg[i] = SIMD_SUB_P(tmp2, tmp1);
      
    }
    d_reg[i] = dd_e;
    STORE(d, d_reg, n, sys_pad);
  } else {
    n = ((N / SIMD_VEC) - 1) * SIMD_VEC;
    LOAD(d_reg, dd, n, sys_pad);
    d_reg[SIMD_VEC - 1] = dd_e;
    STORE(d, d_reg, n, sys_pad);
  }
}
#endif
