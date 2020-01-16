#include "trid_mpi_cpu.h"

#include "trid_simd.h"
#include "math.h"
#include "omp.h"

//#define N_MPI_MAX 128

#define ROUND_DOWN(N,step) (((N)/(step))*step)
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

//
// Thomas solver for reduced system
//
template<typename REAL>
inline void thomas_on_reduced(
    const REAL* __restrict__ aa_r, 
    const REAL* __restrict__ cc_r, 
          REAL* __restrict__ dd_r, 
    const int N, 
    const int stride) {
  int   i, ind = 0;
  //FP aa, bb, cc, dd, c2[N_MAX], d2[N_MAX];
  REAL aa, bb, cc, dd, c2[N], d2[N];
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
  #pragma omp simd
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
    const int N, 
    const int stride) {

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

//
// Modified Thomas forwards pass
//
template<typename REAL>
inline void thomas_forward_vec_strip(
    const REAL *__restrict__ a, 
    const REAL *__restrict__ b, 
    const REAL *__restrict__ c, 
    const REAL *__restrict__ d, 
    const REAL *__restrict__ u, 
          REAL *__restrict__ aa, 
          REAL *__restrict__ cc, 
          REAL *__restrict__ dd, 
    const int N, 
    const int stride,
    const int strip_len) {

  int ind = 0;
  int base = 0;
  
  int n = 0;
  
  REAL bbi;
  
  for(int i = 0; i < 2; i++) {
    base = i * stride;
    #pragma omp simd
    for(int j = 0; j < strip_len; j++) {
      ind = base + j;
      bbi   = static_cast<REAL>(1.0) / b[ind];
      //dd[i] = 66;//d[i] * bbi;
      dd[ind] = d[ind] * bbi;
      aa[ind] = a[ind] * bbi;
      cc[ind] = c[ind] * bbi;
    }
  }
  
  for(int i = 2; i < N; i++) {
    base = i * stride;
    #pragma omp simd
    for(int j = 0; j < strip_len; j++) {
      ind = base + j;
      bbi   = static_cast<REAL>(1.0) / (b[ind] - a[ind] * cc[ind - stride]); 
      //dd[i] = 77;//(d[i] - a[i]*dd[i-1]) * bbi;
      dd[ind] = (d[ind] - a[ind]*dd[ind - stride]) * bbi;
      aa[ind] = (     - a[ind]*aa[ind - stride]) * bbi;
      cc[ind] =                 c[ind]  * bbi;
    }
  }
  
  for(int i = N - 3; i > 0; i--) {
    base = i * stride;
    #pragma omp simd
    for(int j = 0; j < strip_len; j++) {
      ind = base + j;
      dd[ind] = dd[ind] - cc[ind]*dd[ind + stride];
      aa[ind] = aa[ind] - cc[ind]*aa[ind + stride];
      cc[ind] =       - cc[ind]*cc[ind + stride];
    }
  }
  
  #pragma omp simd
  for(int j = 0; j < strip_len; j++) {
    bbi = static_cast<REAL>(1.0) / (static_cast<REAL>(1.0) - cc[j]*aa[stride + j]);
    dd[j] =  bbi * ( dd[j] - cc[j]*dd[stride + j] );
    aa[j] =  bbi *   aa[j];
    cc[j] =  bbi * (       - cc[j]*cc[stride + j] );
  }
}

template<typename REAL>
inline void thomas_backward_vec_strip(
    const REAL *__restrict__ aa, 
    const REAL *__restrict__ cc, 
    const REAL *__restrict__ dd, 
          REAL *__restrict__ d, 
    const int N, 
    const int stride,
    const int strip_len) {

  int ind = 0;
  int base = 0;
  
  #pragma omp simd
  for(int j = 0; j < strip_len; j++) {
    d[j] = dd[j];
  }
  
  for(int i = 1; i < N - 1; i++) {
    base = i * stride;
    #pragma omp simd
    for(int j = 0; j < strip_len; j++) {
      d[base + j] = dd[base + j] - aa[base + j]*dd[j] - cc[base + j]*dd[(N-1) * stride + j];
    }
  }
  
  #pragma omp simd
  for(int j = 0; j < strip_len; j++) {
    d[(N-1) * stride + j] = dd[(N-1) * stride + j];
  }
}

template<typename REAL>
inline void thomas_backwardInc_vec_strip(
    const REAL *__restrict__ aa, 
    const REAL *__restrict__ cc, 
    const REAL *__restrict__ dd, 
          REAL *__restrict__ u, 
    const int N, 
    const int stride,
    const int strip_len) {

  int ind = 0;
  int base = 0;
  
  #pragma omp simd
  for(int j = 0; j < strip_len; j++) {
    u[j] += dd[j];
  }
  
  for(int i = 1; i < N - 1; i++) {
    base = i * stride;
    #pragma omp simd
    for(int j = 0; j < strip_len; j++) {
      u[base + j] += dd[base + j] - aa[base + j]*dd[j] - cc[base + j]*dd[(N-1) * stride + j];
    }
  }
  
  #pragma omp simd
  for(int j = 0; j < strip_len; j++) {
    u[(N-1) * stride + j] += dd[(N-1) * stride + j];
  }
}

template<typename REAL>
inline void thomas_backward(
    const REAL *__restrict__ aa, 
    const REAL *__restrict__ cc, 
    const REAL *__restrict__ dd, 
          REAL *__restrict__ d, 
    const int N, 
    const int stride) {

  int ind = 0;
  
  d[0] = dd[0];
  #pragma omp simd
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
    const int N, 
    const int stride) {

  int ind = 0;
  
  u[0] += dd[0];
  #pragma omp simd
  for (int i=1; i<N-1; i++) {
    ind = i * stride;
    //d[i] = dd[i];//dd[i] - aa[i]*dd[0] - cc[i]*dd[N-1];
    u[ind] += dd[ind] - aa[ind]*dd[0] - cc[ind]*dd[(N-1) * stride];
  }
  u[(N-1) * stride] += dd[(N-1) * stride];
}

void tridSInit(tridS_handle &handle, trid_mpi_handle &mpi_handle, int ndim, int *size) {
  // Get number of mpi procs and the rank of this mpi proc
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_handle.procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_handle.rank);
  
  // Split into multi dim arrangement of mpi procs
  handle.ndim = ndim;
  mpi_handle.pdims = (int *) calloc(handle.ndim, sizeof(int));
  mpi_handle.periodic = (int *) calloc(handle.ndim, sizeof(int));; //false
  mpi_handle.coords = (int *) calloc(handle.ndim, sizeof(int));
  MPI_Dims_create(mpi_handle.procs, handle.ndim, mpi_handle.pdims);
  
  // Create cartecian mpi comm
  MPI_Cart_create(MPI_COMM_WORLD, handle.ndim, mpi_handle.pdims, mpi_handle.periodic, 0,  &(mpi_handle.comm));
  
  // Get rand and coord of current mpi proc
  MPI_Comm_rank(mpi_handle.comm, &(mpi_handle.my_cart_rank));
  MPI_Cart_coords(mpi_handle.comm, mpi_handle.my_cart_rank, handle.ndim, mpi_handle.coords);
  
  // TODO extend to other dimensions
  // Create separate comms for x, y and z dimensions
  int free_coords[3];
  free_coords[0] = 1;
  free_coords[1] = 0;
  free_coords[2] = 0;
  MPI_Cart_sub(mpi_handle.comm, free_coords, &(mpi_handle.x_comm));
  MPI_Comm y_comm;
  free_coords[0] = 0;
  free_coords[1] = 1;
  free_coords[2] = 0;
  MPI_Cart_sub(mpi_handle.comm, free_coords, &(mpi_handle.y_comm));
  MPI_Comm z_comm;
  free_coords[0] = 0;
  free_coords[1] = 0;
  free_coords[2] = 1;
  MPI_Cart_sub(mpi_handle.comm, free_coords, &(mpi_handle.z_comm));
  
  // Store the global problem sizes
  handle.size_g = (int *) calloc(handle.ndim, sizeof(int));
  for(int i = 0; i < handle.ndim; i++) {
    handle.size_g[i] = size[i];
  }
  
  // Calculate size, padding, start and end for each dimension
  handle.size = (int *) calloc(handle.ndim, sizeof(int));
  handle.pads = (int *) calloc(handle.ndim, sizeof(int));
  handle.start_g = (int *) calloc(handle.ndim, sizeof(int));
  handle.end_g = (int *) calloc(handle.ndim, sizeof(int));
  
  for(int i = 0; i < handle.ndim; i++) {
    int tmp = 1 + (handle.size_g[i] - 1) / mpi_handle.pdims[i];
    handle.start_g[i] = mpi_handle.coords[i] * tmp;
    handle.end_g[i] = MIN(((mpi_handle.coords[i] + 1) * tmp) - 1, handle.size_g[i] - 1);
    handle.size[i] = handle.end_g[i] - handle.start_g[i] + 1;
    // Only pad the x dimension
    if(i == 0) {
      handle.pads[i] = (1 + ((tmp - 1) / SIMD_VEC)) * SIMD_VEC;
    } else {
      handle.pads[i] = handle.size[i];
    }
  }
  
  // Allocate memory for arrays
  int mem_size = sizeof(float);
  for(int i = 0; i < handle.ndim; i++) {
    mem_size *= handle.pads[i];
  }
  
  handle.a = (float *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.b = (float *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.c = (float *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.du = (float *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.h_u = (float *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.aa = (float *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.cc = (float *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.dd = (float *) _mm_malloc(mem_size, SIMD_WIDTH);
  
  // Calculate reduced system sizes for each dimension
  handle.sys_len_l = (int *) calloc(handle.ndim, sizeof(int));
  handle.n_sys_g = (int *) calloc(handle.ndim, sizeof(int));
  handle.n_sys_l = (int *) calloc(handle.ndim, sizeof(int));
  
  for(int i = 0; i < handle.ndim; i++) {
    handle.sys_len_l[i] = mpi_handle.pdims[i] * 2;
    handle.n_sys_g[i] = 1;
    handle.n_sys_l[i] = 1;
    for(int j = 0; j < handle.ndim; j++) {
      if(j != i) {
        handle.n_sys_g[i] *= handle.size[j];
        handle.n_sys_l[i] *= handle.size[j];
      }
    }
  }
  
  // Allocate memory for container used to communicate reduced system
  int max = 0;
  for(int i = 0; i < handle.ndim; i++) {
    if(handle.n_sys_l[i] * handle.sys_len_l[i] > max) {
      max = handle.n_sys_l[i] * handle.sys_len_l[i];
    }
  }
  
  handle.halo_sndbuf = (float *) _mm_malloc(max * 3 * sizeof(float), SIMD_WIDTH);
  handle.halo_rcvbuf = (float *) _mm_malloc(max * 3 * sizeof(float), SIMD_WIDTH);
  
  // Allocate memory for reduced system arrays
  max = 0;
  for(int i = 0; i < handle.ndim; i++) {
    if(handle.sys_len_l[i] * handle.n_sys_l[i] > max) {
      max = handle.sys_len_l[i] * handle.n_sys_l[i];
    }
  }
  
  handle.aa_r = (float *) _mm_malloc(sizeof(float) * max, SIMD_WIDTH);
  handle.cc_r = (float *) _mm_malloc(sizeof(float) * max, SIMD_WIDTH);
  handle.dd_r = (float *) _mm_malloc(sizeof(float) * max, SIMD_WIDTH);
}

void tridDInit(tridD_handle &handle, trid_mpi_handle &mpi_handle, int ndim, int *size) {
  // Get number of mpi procs and the rank of this mpi proc
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_handle.procs);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_handle.rank);
  
  // Split into multi dim arrangement of mpi procs
  handle.ndim = ndim;
  mpi_handle.pdims = (int *) calloc(handle.ndim, sizeof(int));
  mpi_handle.periodic = (int *) calloc(handle.ndim, sizeof(int));; //false
  mpi_handle.coords = (int *) calloc(handle.ndim, sizeof(int));
  MPI_Dims_create(mpi_handle.procs, handle.ndim, mpi_handle.pdims);
  
  // Create cartecian mpi comm
  MPI_Cart_create(MPI_COMM_WORLD, handle.ndim, mpi_handle.pdims, mpi_handle.periodic, 0,  &mpi_handle.comm);
  
  // Get rand and coord of current mpi proc
  MPI_Comm_rank(mpi_handle.comm, &mpi_handle.my_cart_rank);
  MPI_Cart_coords(mpi_handle.comm, mpi_handle.my_cart_rank, handle.ndim, mpi_handle.coords);
  
  // TODO extend to other dimensions
  // Create separate comms for x, y and z dimensions
  int free_coords[3];
  free_coords[0] = 1;
  free_coords[1] = 0;
  free_coords[2] = 0;
  MPI_Cart_sub(mpi_handle.comm, free_coords, &mpi_handle.x_comm);
  MPI_Comm y_comm;
  free_coords[0] = 0;
  free_coords[1] = 1;
  free_coords[2] = 0;
  MPI_Cart_sub(mpi_handle.comm, free_coords, &mpi_handle.y_comm);
  MPI_Comm z_comm;
  free_coords[0] = 0;
  free_coords[1] = 0;
  free_coords[2] = 1;
  MPI_Cart_sub(mpi_handle.comm, free_coords, &mpi_handle.z_comm);
  
  // Store the global problem sizes
  handle.size_g = (int *) calloc(handle.ndim, sizeof(int));
  for(int i = 0; i < handle.ndim; i++) {
    handle.size_g[i] = size[i];
  }
  
  // Calculate size, padding, start and end for each dimension
  handle.size = (int *) calloc(handle.ndim, sizeof(int));
  handle.pads = (int *) calloc(handle.ndim, sizeof(int));
  handle.start_g = (int *) calloc(handle.ndim, sizeof(int));
  handle.end_g = (int *) calloc(handle.ndim, sizeof(int));
  
  for(int i = 0; i < handle.ndim; i++) {
    int tmp = 1 + (handle.size_g[i] - 1) / mpi_handle.pdims[i];
    handle.start_g[i] = mpi_handle.coords[i] * tmp;
    handle.end_g[i] = MIN(((mpi_handle.coords[i] + 1) * tmp) - 1, handle.size_g[i] - 1);
    handle.size[i] = handle.end_g[i] - handle.start_g[i] + 1;
    // Only pad the x dimension
    if(i == 0) {
      handle.pads[i] = (1 + ((tmp - 1) / SIMD_VEC)) * SIMD_VEC;
    } else {
      handle.pads[i] = handle.size[i];
    }
  }
  
  // Allocate memory for arrays
  int mem_size = sizeof(double);
  for(int i = 0; i < handle.ndim; i++) {
    mem_size *= handle.pads[i];
  }
  
  handle.a = (double *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.b = (double *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.c = (double *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.du = (double *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.h_u = (double *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.aa = (double *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.cc = (double *) _mm_malloc(mem_size, SIMD_WIDTH);
  handle.dd = (double *) _mm_malloc(mem_size, SIMD_WIDTH);
  
  // Calculate reduced system sizes for each dimension
  handle.sys_len_l = (int *) calloc(handle.ndim, sizeof(int));
  handle.n_sys_g = (int *) calloc(handle.ndim, sizeof(int));
  handle.n_sys_l = (int *) calloc(handle.ndim, sizeof(int));
  
  for(int i = 0; i < handle.ndim; i++) {
    handle.sys_len_l[i] = mpi_handle.pdims[i] * 2;
    handle.n_sys_g[i] = 1;
    handle.n_sys_l[i] = 1;
    for(int j = 0; j < handle.ndim; j++) {
      if(j != i) {
        handle.n_sys_g[i] *= handle.size[j];
        handle.n_sys_l[i] *= handle.size[j];
      }
    }
  }
  
  // Allocate memory for container used to communicate reduced system
  int max = 0;
  for(int i = 0; i < handle.ndim; i++) {
    if(handle.n_sys_l[i] * handle.sys_len_l[i] > max) {
      max = handle.n_sys_l[i] * handle.sys_len_l[i];
    }
  }
  
  handle.halo_sndbuf = (double *) _mm_malloc(max * 3 * sizeof(double), SIMD_WIDTH);
  handle.halo_rcvbuf = (double *) _mm_malloc(max * 3 * sizeof(double), SIMD_WIDTH);
  
  // Allocate memory for reduced system arrays
  max = 0;
  for(int i = 0; i < handle.ndim; i++) {
    if(handle.sys_len_l[i] * handle.n_sys_l[i] > max) {
      max = handle.sys_len_l[i] * handle.n_sys_l[i];
    }
  }
  
  handle.aa_r = (double *) _mm_malloc(sizeof(double) * max, SIMD_WIDTH);
  handle.cc_r = (double *) _mm_malloc(sizeof(double) * max, SIMD_WIDTH);
  handle.dd_r = (double *) _mm_malloc(sizeof(double) * max, SIMD_WIDTH);
}

void tridSClean(tridS_handle &handle, trid_mpi_handle &mpi_handle) {
  free(mpi_handle.pdims);
  free(mpi_handle.periodic);
  free(mpi_handle.coords);
  free(handle.size_g);
  free(handle.size);
  free(handle.start_g);
  free(handle.end_g);
  _mm_free(handle.a);
  _mm_free(handle.b);
  _mm_free(handle.c);
  _mm_free(handle.du);
  _mm_free(handle.h_u);
  _mm_free(handle.aa);
  _mm_free(handle.cc);
  _mm_free(handle.dd);
  free(handle.sys_len_l);
  free(handle.n_sys_g);
  free(handle.n_sys_l);
  _mm_free(handle.halo_sndbuf);
  _mm_free(handle.halo_rcvbuf);
  _mm_free(handle.aa_r);
  _mm_free(handle.cc_r);
  _mm_free(handle.dd_r);
}

void tridDClean(tridD_handle &handle, trid_mpi_handle &mpi_handle) {
  free(mpi_handle.pdims);
  free(mpi_handle.periodic);
  free(mpi_handle.coords);
  free(handle.size_g);
  free(handle.size);
  free(handle.start_g);
  free(handle.end_g);
  _mm_free(handle.a);
  _mm_free(handle.b);
  _mm_free(handle.c);
  _mm_free(handle.du);
  _mm_free(handle.h_u);
  _mm_free(handle.aa);
  _mm_free(handle.cc);
  _mm_free(handle.dd);
  free(handle.sys_len_l);
  free(handle.n_sys_g);
  free(handle.n_sys_l);
  _mm_free(handle.halo_sndbuf);
  _mm_free(handle.halo_rcvbuf);
  _mm_free(handle.aa_r);
  _mm_free(handle.cc_r);
  _mm_free(handle.dd_r);
}

void tridBatchS(tridS_handle &handle, trid_mpi_handle &mpi_handle, int solveDim) {
  if(solveDim == 0) {
    /*********************
     * 
     * X Dimension Solve
     * 
     *********************/
    
    // Do modified thomas forward pass
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[0]; id++) {
      int base = id * handle.pads[0];
      thomas_forward<float>(&handle.a[base], &handle.b[base], &handle.c[base], &handle.du[base],
                     &handle.h_u[base], &handle.aa[base], &handle.cc[base],
                     &handle.dd[base], handle.size[0], 1);
    }
    
    // Pack boundary values
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[0]; id++) {
      // Gather coefficients of a,c,d
      int halo_base = id * 6;
      int data_base = id * handle.pads[0];
      handle.halo_sndbuf[halo_base]     = handle.aa[data_base];
      handle.halo_sndbuf[halo_base + 1] = handle.aa[data_base + handle.size[0]-1];
      handle.halo_sndbuf[halo_base + 2] = handle.cc[data_base];
      handle.halo_sndbuf[halo_base + 3] = handle.cc[data_base + handle.size[0]-1];
      handle.halo_sndbuf[halo_base + 4] = handle.dd[data_base];
      handle.halo_sndbuf[halo_base + 5] = handle.dd[data_base + handle.size[0]-1];
    }
    
    // Communicate boundary values
    MPI_Gather(handle.halo_sndbuf, handle.n_sys_l[0]*3*2, MPI_FLOAT, handle.halo_rcvbuf,
               handle.n_sys_l[0]*3*2, MPI_FLOAT, 0, mpi_handle.x_comm);
    
    // Unpack boundary values
    if(mpi_handle.coords[0] == 0) {
      #pragma omp parallel for collapse(2)
      for(int p = 0; p < mpi_handle.pdims[0]; p++) {
        for(int id = 0; id < handle.n_sys_l[0]; id++) {
          int halo_base = p * handle.n_sys_l[0] * 6 + id * 6;
          int data_base = id * handle.sys_len_l[0] + p * 2;
          handle.aa_r[data_base]     = handle.halo_rcvbuf[halo_base];
          handle.aa_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 1];
          handle.cc_r[data_base]     = handle.halo_rcvbuf[halo_base + 2];
          handle.cc_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 3];
          handle.dd_r[data_base]     = handle.halo_rcvbuf[halo_base + 4];
          handle.dd_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 5];
        }
      }
    }
    
    // Compute reduced system
    if(mpi_handle.coords[0] == 0) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys_l[0]; id++) {
        int base = id * handle.sys_len_l[0];
        thomas_on_reduced<float>(&handle.aa_r[base], &handle.cc_r[base], &handle.dd_r[base],
                          handle.sys_len_l[0], 1);
      }
    }
    
    // Pack boundary solution data
    if(mpi_handle.coords[0] == 0) {
      #pragma omp parallel for
      for(int p = 0; p < mpi_handle.pdims[0]; p++) {
        for(int id = 0; id < handle.n_sys_l[0]; id++) {
          int halo_base = p * handle.n_sys_l[0] * 2 + id * 2;
          int data_base = id * handle.sys_len_l[0] + p * 2;
          handle.halo_rcvbuf[halo_base]     = handle.dd_r[data_base];
          handle.halo_rcvbuf[halo_base + 1] = handle.dd_r[data_base + 1];
        }
      }
    }
    
    // Send back new values
    MPI_Scatter(handle.halo_rcvbuf, handle.n_sys_l[0] * 2, MPI_FLOAT, handle.halo_sndbuf,
                handle.n_sys_l[0] * 2, MPI_FLOAT, 0, mpi_handle.x_comm);
    
    // Unpack boundary solution
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[0]; id++) {
      // Gather coefficients of a,c,d
      int data_base = id * handle.pads[0];
      int halo_base = id * 2;
      handle.dd[data_base]                    = handle.halo_sndbuf[halo_base];
      handle.dd[data_base + handle.size[0]-1] = handle.halo_sndbuf[halo_base + 1];
    }
    
    // Do the backward pass of modified Thomas
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[0]; id++) {
      int ind = id * handle.pads[0];
      thomas_backward<float>(&handle.aa[ind], &handle.cc[ind], &handle.dd[ind], 
                      &handle.h_u[ind], handle.size[0], 1);
    }
  } else if(solveDim == 1) {
    /*********************
     * 
     * Y Dimension Solve
     * 
     *********************/
    
    // Do modified thomas forward pass
    #pragma omp parallel for
    for(int z = 0; z < handle.size[2]; z++) {
      int base = z * handle.pads[0] * handle.pads[1];
      thomas_forward_vec_strip<float>(&handle.a[base], &handle.b[base], &handle.c[base],
                               &handle.du[base], &handle.h_u[base], &handle.aa[base],
                               &handle.cc[base], &handle.dd[base], handle.size[1],
                               handle.pads[0], handle.size[0]);
    }
    
    // Pack boundary values
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[1]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] * handle.pads[1] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * (handle.size[1] - 1));
      int halo_base = id * 6;
      // Gather coefficients of a,c,d
      handle.halo_sndbuf[halo_base]     = handle.aa[start];
      handle.halo_sndbuf[halo_base + 1] = handle.aa[end];
      handle.halo_sndbuf[halo_base + 2] = handle.cc[start];
      handle.halo_sndbuf[halo_base + 3] = handle.cc[end];
      handle.halo_sndbuf[halo_base + 4] = handle.dd[start];
      handle.halo_sndbuf[halo_base + 5] = handle.dd[end];
    }
    
    // Communicate boundary values
    MPI_Gather(handle.halo_sndbuf, handle.n_sys_l[1]*3*2, MPI_FLOAT, handle.halo_rcvbuf,
               handle.n_sys_l[1]*3*2, MPI_FLOAT, 0, mpi_handle.y_comm);
    
    // Unpack boundary values
    if(mpi_handle.coords[1] == 0) {
      #pragma omp parallel for collapse(2)
      for(int p = 0; p < mpi_handle.pdims[1]; p++) {
        for(int id = 0; id < handle.n_sys_l[1]; id++) {
          int halo_base = p * handle.n_sys_l[1] * 6 + id * 6;
          int data_base = id * handle.sys_len_l[1] + p * 2;
          handle.aa_r[data_base]     = handle.halo_rcvbuf[halo_base];
          handle.aa_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 1];
          handle.cc_r[data_base]     = handle.halo_rcvbuf[halo_base + 2];
          handle.cc_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 3];
          handle.dd_r[data_base]     = handle.halo_rcvbuf[halo_base + 4];
          handle.dd_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 5];
        }
      }
    }
    
    // Compute reduced system
    if(mpi_handle.coords[1] == 0) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys_l[1]; id++) {
        int base = id * handle.sys_len_l[1];
        thomas_on_reduced<float>(&handle.aa_r[base], &handle.cc_r[base], &handle.dd_r[base],
                                 handle.sys_len_l[1], 1);
      }
    }
    
    // Pack boundary solution data
    if(mpi_handle.coords[1] == 0) {
      #pragma omp parallel for
      for(int p = 0; p < mpi_handle.pdims[1]; p++) {
        for(int id = 0; id < handle.n_sys_l[1]; id++) {
          int halo_base = p * handle.n_sys_l[1] * 2 + id * 2;
          int data_base = id * handle.sys_len_l[1] + p * 2;
          handle.halo_rcvbuf[halo_base]     = handle.dd_r[data_base];
          handle.halo_rcvbuf[halo_base + 1] = handle.dd_r[data_base + 1];
        }
      }
    }
    
    // Send back new values
    MPI_Scatter(handle.halo_rcvbuf, handle.n_sys_l[1]*2, MPI_FLOAT, handle.halo_sndbuf,
                handle.n_sys_l[1]*2, MPI_FLOAT, 0, mpi_handle.y_comm);
    
    // Unpack boundary solution
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[1]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] * handle.pads[1] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * (handle.size[1] - 1));
      int halo_base = id * 2;
      handle.dd[start] = handle.halo_sndbuf[halo_base];
      handle.dd[end]   = handle.halo_sndbuf[halo_base + 1];
    }
    
    // Do the backward pass of modified Thomas
    #pragma omp parallel for
    for(int z = 0; z < handle.size[2]; z++) {
      int base = z * handle.pads[0] * handle.pads[1];
      thomas_backward_vec_strip<float>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                       &handle.h_u[base], handle.size[1], handle.pads[0],
                                       handle.size[0]);
    }
  } else {
    /*********************
     * 
     * Z Dimension Solve
     * 
     *********************/
    
    // Do modified thomas forward pass
    #pragma omp parallel for
    for(int y = 0; y < handle.size[1]; y++) {
      int base = y * handle.pads[0];
      thomas_forward_vec_strip<float>(&handle.a[base], &handle.b[base], &handle.c[base],
                               &handle.du[base], &handle.h_u[base], &handle.aa[base],
                               &handle.cc[base], &handle.dd[base], handle.size[2],
                               handle.pads[0] * handle.pads[1], handle.size[0]);
    }
    
    // Pack boundary values
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[2]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * handle.pads[1] * (handle.size[2] - 1));
      int halo_base = id * 6;
      // Gather coefficients of a,c,d
      handle.halo_sndbuf[halo_base]     = handle.aa[start];
      handle.halo_sndbuf[halo_base + 1] = handle.aa[end];
      handle.halo_sndbuf[halo_base + 2] = handle.cc[start];
      handle.halo_sndbuf[halo_base + 3] = handle.cc[end];
      handle.halo_sndbuf[halo_base + 4] = handle.dd[start];
      handle.halo_sndbuf[halo_base + 5] = handle.dd[end];
    }
    
    // Communicate boundary values
    MPI_Gather(handle.halo_sndbuf, handle.n_sys_l[2]*3*2, MPI_FLOAT, handle.halo_rcvbuf,
               handle.n_sys_l[2]*3*2, MPI_FLOAT, 0, mpi_handle.z_comm);
    
    // Unpack boundary data
    if(mpi_handle.coords[2] == 0) {
      #pragma omp parallel for collapse(2)
      for(int p = 0; p < mpi_handle.pdims[2]; p++) {
        for(int id = 0; id < handle.n_sys_l[2]; id++) {
          int data_base = id * handle.sys_len_l[2] + p * 2;
          int halo_base = p * handle.n_sys_l[2] * 6 + id * 6;
          handle.aa_r[data_base]     = handle.halo_rcvbuf[halo_base];
          handle.aa_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 1];
          handle.cc_r[data_base]     = handle.halo_rcvbuf[halo_base + 2];
          handle.cc_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 3];
          handle.dd_r[data_base]     = handle.halo_rcvbuf[halo_base + 4];
          handle.dd_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 5];
        }
      }
    }
    
    // Compute reduced system
    if(mpi_handle.coords[2] == 0) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys_l[2]; id++) {
        int base = id * handle.sys_len_l[2];
        thomas_on_reduced<float>(&handle.aa_r[base], &handle.cc_r[base], &handle.dd_r[base],
                                 handle.sys_len_l[2], 1);
      }
    }
    
    // Pack boundary solution data
    if(mpi_handle.coords[2] == 0) {
      #pragma omp parallel for
      for(int p = 0; p < mpi_handle.pdims[2]; p++) {
        for(int id = 0; id < handle.n_sys_l[2]; id++) {
          int halo_base = p * handle.n_sys_l[2] * 2 + id * 2;
          int data_base = id * handle.sys_len_l[2] + p * 2;
          handle.halo_rcvbuf[halo_base]     = handle.dd_r[data_base];
          handle.halo_rcvbuf[halo_base + 1] = handle.dd_r[data_base + 1];
        }
      }
    }
    
    // Send back new values
    MPI_Scatter(handle.halo_rcvbuf, handle.n_sys_l[2]*2, MPI_FLOAT, handle.halo_sndbuf,
                handle.n_sys_l[2]*2, MPI_FLOAT, 0, mpi_handle.z_comm);
    
    // Unpack boundary solution
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[2]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * handle.pads[1] * (handle.size[2] - 1));
      int halo_base = id * 2;
      handle.dd[start] = handle.halo_sndbuf[halo_base];
      handle.dd[end]   = handle.halo_sndbuf[halo_base + 1];
    }
    
    // Do the backward pass of modified Thomas
    #pragma omp parallel for
    for(int y = 0; y < handle.size[1]; y++) {
      int base = y * handle.pads[0];
      thomas_backward_vec_strip<float>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                       &handle.h_u[base], handle.size[2], 
                                       handle.pads[0] * handle.pads[1], handle.size[0]);
    }
  }
}

void tridBatchSInc(tridS_handle &handle, trid_mpi_handle &mpi_handle, int solveDim) {
  if(solveDim == 0) {
    /*********************
     * 
     * X Dimension Solve
     * 
     *********************/
    
    // Do modified thomas forward pass
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[0]; id++) {
      int base = id * handle.pads[0];
      thomas_forward<float>(&handle.a[base], &handle.b[base], &handle.c[base], &handle.du[base],
                     &handle.h_u[base], &handle.aa[base], &handle.cc[base],
                     &handle.dd[base], handle.size[0], 1);
    }
    
    // Pack boundary values
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[0]; id++) {
      // Gather coefficients of a,c,d
      int halo_base = id * 6;
      int data_base = id * handle.pads[0];
      handle.halo_sndbuf[halo_base]     = handle.aa[data_base];
      handle.halo_sndbuf[halo_base + 1] = handle.aa[data_base + handle.size[0]-1];
      handle.halo_sndbuf[halo_base + 2] = handle.cc[data_base];
      handle.halo_sndbuf[halo_base + 3] = handle.cc[data_base + handle.size[0]-1];
      handle.halo_sndbuf[halo_base + 4] = handle.dd[data_base];
      handle.halo_sndbuf[halo_base + 5] = handle.dd[data_base + handle.size[0]-1];
    }
    
    // Communicate boundary values
    MPI_Gather(handle.halo_sndbuf, handle.n_sys_l[0]*3*2, MPI_FLOAT, handle.halo_rcvbuf,
               handle.n_sys_l[0]*3*2, MPI_FLOAT, 0, mpi_handle.x_comm);
    
    // Unpack boundary values
    if(mpi_handle.coords[0] == 0) {
      #pragma omp parallel for collapse(2)
      for(int p = 0; p < mpi_handle.pdims[0]; p++) {
        for(int id = 0; id < handle.n_sys_l[0]; id++) {
          int halo_base = p * handle.n_sys_l[0] * 6 + id * 6;
          int data_base = id * handle.sys_len_l[0] + p * 2;
          handle.aa_r[data_base]     = handle.halo_rcvbuf[halo_base];
          handle.aa_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 1];
          handle.cc_r[data_base]     = handle.halo_rcvbuf[halo_base + 2];
          handle.cc_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 3];
          handle.dd_r[data_base]     = handle.halo_rcvbuf[halo_base + 4];
          handle.dd_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 5];
        }
      }
    }
    
    // Compute reduced system
    if(mpi_handle.coords[0] == 0) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys_l[0]; id++) {
        int base = id * handle.sys_len_l[0];
        thomas_on_reduced<float>(&handle.aa_r[base], &handle.cc_r[base], &handle.dd_r[base],
                          handle.sys_len_l[0], 1);
      }
    }
    
    // Pack boundary solution data
    if(mpi_handle.coords[0] == 0) {
      #pragma omp parallel for
      for(int p = 0; p < mpi_handle.pdims[0]; p++) {
        for(int id = 0; id < handle.n_sys_l[0]; id++) {
          int halo_base = p * handle.n_sys_l[0] * 2 + id * 2;
          int data_base = id * handle.sys_len_l[0] + p * 2;
          handle.halo_rcvbuf[halo_base]     = handle.dd_r[data_base];
          handle.halo_rcvbuf[halo_base + 1] = handle.dd_r[data_base + 1];
        }
      }
    }
    
    // Send back new values
    MPI_Scatter(handle.halo_rcvbuf, handle.n_sys_l[0] * 2, MPI_FLOAT, handle.halo_sndbuf,
                handle.n_sys_l[0] * 2, MPI_FLOAT, 0, mpi_handle.x_comm);
    
    // Unpack boundary solution
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[0]; id++) {
      // Gather coefficients of a,c,d
      int data_base = id * handle.pads[0];
      int halo_base = id * 2;
      handle.dd[data_base]            = handle.halo_sndbuf[halo_base];
      handle.dd[data_base + handle.size[0]-1] = handle.halo_sndbuf[halo_base + 1];
    }
    
    // Do the backward pass of modified Thomas
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[0]; id++) {
      int ind = id * handle.pads[0];
      thomas_backwardInc<float>(&handle.aa[ind], &handle.cc[ind], &handle.dd[ind], 
                      &handle.h_u[ind], handle.size[0], 1);
    }
  } else if(solveDim == 1) {
    /*********************
     * 
     * Y Dimension Solve
     * 
     *********************/
    
    // Do modified thomas forward pass
    #pragma omp parallel for
    for(int z = 0; z < handle.size[2]; z++) {
      int base = z * handle.pads[0] * handle.pads[1];
      thomas_forward_vec_strip<float>(&handle.a[base], &handle.b[base], &handle.c[base],
                               &handle.du[base], &handle.h_u[base], &handle.aa[base],
                               &handle.cc[base], &handle.dd[base], handle.size[1],
                               handle.pads[0], handle.size[0]);
    }
    
    // Pack boundary values
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[1]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] * handle.pads[1] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * (handle.size[1] - 1));
      int halo_base = id * 6;
      // Gather coefficients of a,c,d
      handle.halo_sndbuf[halo_base]     = handle.aa[start];
      handle.halo_sndbuf[halo_base + 1] = handle.aa[end];
      handle.halo_sndbuf[halo_base + 2] = handle.cc[start];
      handle.halo_sndbuf[halo_base + 3] = handle.cc[end];
      handle.halo_sndbuf[halo_base + 4] = handle.dd[start];
      handle.halo_sndbuf[halo_base + 5] = handle.dd[end];
    }
    
    // Communicate boundary values
    MPI_Gather(handle.halo_sndbuf, handle.n_sys_l[1]*3*2, MPI_FLOAT, handle.halo_rcvbuf,
               handle.n_sys_l[1]*3*2, MPI_FLOAT, 0, mpi_handle.y_comm);
    
    // Unpack boundary values
    if(mpi_handle.coords[1] == 0) {
      #pragma omp parallel for collapse(2)
      for(int p = 0; p < mpi_handle.pdims[1]; p++) {
        for(int id = 0; id < handle.n_sys_l[1]; id++) {
          int halo_base = p * handle.n_sys_l[1] * 6 + id * 6;
          int data_base = id * handle.sys_len_l[1] + p * 2;
          handle.aa_r[data_base]     = handle.halo_rcvbuf[halo_base];
          handle.aa_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 1];
          handle.cc_r[data_base]     = handle.halo_rcvbuf[halo_base + 2];
          handle.cc_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 3];
          handle.dd_r[data_base]     = handle.halo_rcvbuf[halo_base + 4];
          handle.dd_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 5];
        }
      }
    }
    
    // Compute reduced system
    if(mpi_handle.coords[1] == 0) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys_l[1]; id++) {
        int base = id * handle.sys_len_l[1];
        thomas_on_reduced<float>(&handle.aa_r[base], &handle.cc_r[base], &handle.dd_r[base],
                                 handle.sys_len_l[1], 1);
      }
    }
    
    // Pack boundary solution data
    if(mpi_handle.coords[1] == 0) {
      #pragma omp parallel for
      for(int p = 0; p < mpi_handle.pdims[1]; p++) {
        for(int id = 0; id < handle.n_sys_l[1]; id++) {
          int halo_base = p * handle.n_sys_l[1] * 2 + id * 2;
          int data_base = id * handle.sys_len_l[1] + p * 2;
          handle.halo_rcvbuf[halo_base]     = handle.dd_r[data_base];
          handle.halo_rcvbuf[halo_base + 1] = handle.dd_r[data_base + 1];
        }
      }
    }
    
    // Send back new values
    MPI_Scatter(handle.halo_rcvbuf, handle.n_sys_l[1]*2, MPI_FLOAT, handle.halo_sndbuf,
                handle.n_sys_l[1]*2, MPI_FLOAT, 0, mpi_handle.y_comm);
    
    // Unpack boundary solution
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[1]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] * handle.pads[1] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * (handle.size[1] - 1));
      int halo_base = id * 2;
      handle.dd[start] = handle.halo_sndbuf[halo_base];
      handle.dd[end]   = handle.halo_sndbuf[halo_base + 1];
    }
    
    // Do the backward pass of modified Thomas
    #pragma omp parallel for
    for(int z = 0; z < handle.size[2]; z++) {
      int base = z * handle.pads[0] * handle.pads[1];
      thomas_backwardInc_vec_strip<float>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                       &handle.h_u[base], handle.size[1], handle.pads[0],
                                       handle.size[0]);
    }
  } else {
    /*********************
     * 
     * Z Dimension Solve
     * 
     *********************/
    
    // Do modified thomas forward pass
    #pragma omp parallel for
    for(int y = 0; y < handle.size[1]; y++) {
      int base = y * handle.pads[0];
      thomas_forward_vec_strip<float>(&handle.a[base], &handle.b[base], &handle.c[base],
                               &handle.du[base], &handle.h_u[base], &handle.aa[base],
                               &handle.cc[base], &handle.dd[base], handle.size[2],
                               handle.pads[0] * handle.pads[1], handle.size[0]);
    }
    
    // Pack boundary values
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[2]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * handle.pads[1] * (handle.size[2] - 1));
      int halo_base = id * 6;
      // Gather coefficients of a,c,d
      handle.halo_sndbuf[halo_base]     = handle.aa[start];
      handle.halo_sndbuf[halo_base + 1] = handle.aa[end];
      handle.halo_sndbuf[halo_base + 2] = handle.cc[start];
      handle.halo_sndbuf[halo_base + 3] = handle.cc[end];
      handle.halo_sndbuf[halo_base + 4] = handle.dd[start];
      handle.halo_sndbuf[halo_base + 5] = handle.dd[end];
    }
    
    // Communicate boundary values
    MPI_Gather(handle.halo_sndbuf, handle.n_sys_l[2]*3*2, MPI_FLOAT, handle.halo_rcvbuf,
               handle.n_sys_l[2]*3*2, MPI_FLOAT, 0, mpi_handle.z_comm);
    
    // Unpack boundary data
    if(mpi_handle.coords[2] == 0) {
      #pragma omp parallel for collapse(2)
      for(int p = 0; p < mpi_handle.pdims[2]; p++) {
        for(int id = 0; id < handle.n_sys_l[2]; id++) {
          int data_base = id * handle.sys_len_l[2] + p * 2;
          int halo_base = p * handle.n_sys_l[2] * 6 + id * 6;
          handle.aa_r[data_base]     = handle.halo_rcvbuf[halo_base];
          handle.aa_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 1];
          handle.cc_r[data_base]     = handle.halo_rcvbuf[halo_base + 2];
          handle.cc_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 3];
          handle.dd_r[data_base]     = handle.halo_rcvbuf[halo_base + 4];
          handle.dd_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 5];
        }
      }
    }
    
    // Compute reduced system
    if(mpi_handle.coords[2] == 0) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys_l[2]; id++) {
        int base = id * handle.sys_len_l[2];
        thomas_on_reduced<float>(&handle.aa_r[base], &handle.cc_r[base], &handle.dd_r[base],
                                 handle.sys_len_l[2], 1);
      }
    }
    
    // Pack boundary solution data
    if(mpi_handle.coords[2] == 0) {
      #pragma omp parallel for
      for(int p = 0; p < mpi_handle.pdims[2]; p++) {
        for(int id = 0; id < handle.n_sys_l[2]; id++) {
          int halo_base = p * handle.n_sys_l[2] * 2 + id * 2;
          int data_base = id * handle.sys_len_l[2] + p * 2;
          handle.halo_rcvbuf[halo_base]     = handle.dd_r[data_base];
          handle.halo_rcvbuf[halo_base + 1] = handle.dd_r[data_base + 1];
        }
      }
    }
    
    // Send back new values
    MPI_Scatter(handle.halo_rcvbuf, handle.n_sys_l[2]*2, MPI_FLOAT, handle.halo_sndbuf,
                handle.n_sys_l[2]*2, MPI_FLOAT, 0, mpi_handle.z_comm);
    
    // Unpack boundary solution
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[2]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * handle.pads[1] * (handle.size[2] - 1));
      int halo_base = id * 2;
      handle.dd[start] = handle.halo_sndbuf[halo_base];
      handle.dd[end]   = handle.halo_sndbuf[halo_base + 1];
    }
    
    // Do the backward pass of modified Thomas
    #pragma omp parallel for
    for(int y = 0; y < handle.size[1]; y++) {
      int base = y * handle.pads[0];
      thomas_backwardInc_vec_strip<float>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                       &handle.h_u[base], handle.size[2], 
                                       handle.pads[0] * handle.pads[1], handle.size[0]);
    }
  }
}

void tridBatchD(tridD_handle &handle, trid_mpi_handle &mpi_handle, int solveDim) {
  if(solveDim == 0) {
    /*********************
     * 
     * X Dimension Solve
     * 
     *********************/
    
    // Do modified thomas forward pass
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[0]; id++) {
      int base = id * handle.pads[0];
      thomas_forward<double>(&handle.a[base], &handle.b[base], &handle.c[base], &handle.du[base],
                     &handle.h_u[base], &handle.aa[base], &handle.cc[base],
                     &handle.dd[base], handle.size[0], 1);
    }
    
    // Pack boundary values
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[0]; id++) {
      // Gather coefficients of a,c,d
      int halo_base = id * 6;
      int data_base = id * handle.pads[0];
      handle.halo_sndbuf[halo_base]     = handle.aa[data_base];
      handle.halo_sndbuf[halo_base + 1] = handle.aa[data_base + handle.size[0]-1];
      handle.halo_sndbuf[halo_base + 2] = handle.cc[data_base];
      handle.halo_sndbuf[halo_base + 3] = handle.cc[data_base + handle.size[0]-1];
      handle.halo_sndbuf[halo_base + 4] = handle.dd[data_base];
      handle.halo_sndbuf[halo_base + 5] = handle.dd[data_base + handle.size[0]-1];
    }
    
    // Communicate boundary values
    MPI_Gather(handle.halo_sndbuf, handle.n_sys_l[0]*3*2, MPI_DOUBLE, handle.halo_rcvbuf,
               handle.n_sys_l[0]*3*2, MPI_DOUBLE, 0, mpi_handle.x_comm);
    
    // Unpack boundary values
    if(mpi_handle.coords[0] == 0) {
      #pragma omp parallel for collapse(2)
      for(int p = 0; p < mpi_handle.pdims[0]; p++) {
        for(int id = 0; id < handle.n_sys_l[0]; id++) {
          int halo_base = p * handle.n_sys_l[0] * 6 + id * 6;
          int data_base = id * handle.sys_len_l[0] + p * 2;
          handle.aa_r[data_base]     = handle.halo_rcvbuf[halo_base];
          handle.aa_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 1];
          handle.cc_r[data_base]     = handle.halo_rcvbuf[halo_base + 2];
          handle.cc_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 3];
          handle.dd_r[data_base]     = handle.halo_rcvbuf[halo_base + 4];
          handle.dd_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 5];
        }
      }
    }
    
    // Compute reduced system
    if(mpi_handle.coords[0] == 0) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys_l[0]; id++) {
        int base = id * handle.sys_len_l[0];
        thomas_on_reduced<double>(&handle.aa_r[base], &handle.cc_r[base], &handle.dd_r[base],
                          handle.sys_len_l[0], 1);
      }
    }
    
    // Pack boundary solution data
    if(mpi_handle.coords[0] == 0) {
      #pragma omp parallel for
      for(int p = 0; p < mpi_handle.pdims[0]; p++) {
        for(int id = 0; id < handle.n_sys_l[0]; id++) {
          int halo_base = p * handle.n_sys_l[0] * 2 + id * 2;
          int data_base = id * handle.sys_len_l[0] + p * 2;
          handle.halo_rcvbuf[halo_base]     = handle.dd_r[data_base];
          handle.halo_rcvbuf[halo_base + 1] = handle.dd_r[data_base + 1];
        }
      }
    }
    
    // Send back new values
    MPI_Scatter(handle.halo_rcvbuf, handle.n_sys_l[0] * 2, MPI_DOUBLE, handle.halo_sndbuf,
                handle.n_sys_l[0] * 2, MPI_DOUBLE, 0, mpi_handle.x_comm);
    
    // Unpack boundary solution
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[0]; id++) {
      // Gather coefficients of a,c,d
      int data_base = id * handle.pads[0];
      int halo_base = id * 2;
      handle.dd[data_base]            = handle.halo_sndbuf[halo_base];
      handle.dd[data_base + handle.size[0]-1] = handle.halo_sndbuf[halo_base + 1];
    }
    
    // Do the backward pass of modified Thomas
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[0]; id++) {
      int ind = id * handle.pads[0];
      thomas_backward<double>(&handle.aa[ind], &handle.cc[ind], &handle.dd[ind], 
                      &handle.h_u[ind], handle.size[0], 1);
    }
  } else if(solveDim == 1) {
    /*********************
     * 
     * Y Dimension Solve
     * 
     *********************/
    
    // Do modified thomas forward pass
    #pragma omp parallel for
    for(int z = 0; z < handle.size[2]; z++) {
      int base = z * handle.pads[0] * handle.pads[1];
      thomas_forward_vec_strip<double>(&handle.a[base], &handle.b[base], &handle.c[base],
                               &handle.du[base], &handle.h_u[base], &handle.aa[base],
                               &handle.cc[base], &handle.dd[base], handle.size[1],
                               handle.pads[0], handle.size[0]);
    }
    
    // Pack boundary values
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[1]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] * handle.pads[1] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * (handle.size[1] - 1));
      int halo_base = id * 6;
      // Gather coefficients of a,c,d
      handle.halo_sndbuf[halo_base]     = handle.aa[start];
      handle.halo_sndbuf[halo_base + 1] = handle.aa[end];
      handle.halo_sndbuf[halo_base + 2] = handle.cc[start];
      handle.halo_sndbuf[halo_base + 3] = handle.cc[end];
      handle.halo_sndbuf[halo_base + 4] = handle.dd[start];
      handle.halo_sndbuf[halo_base + 5] = handle.dd[end];
    }
    
    // Communicate boundary values
    MPI_Gather(handle.halo_sndbuf, handle.n_sys_l[1]*3*2, MPI_DOUBLE, handle.halo_rcvbuf,
               handle.n_sys_l[1]*3*2, MPI_DOUBLE, 0, mpi_handle.y_comm);
    
    // Unpack boundary values
    if(mpi_handle.coords[1] == 0) {
      #pragma omp parallel for collapse(2)
      for(int p = 0; p < mpi_handle.pdims[1]; p++) {
        for(int id = 0; id < handle.n_sys_l[1]; id++) {
          int halo_base = p * handle.n_sys_l[1] * 6 + id * 6;
          int data_base = id * handle.sys_len_l[1] + p * 2;
          handle.aa_r[data_base]     = handle.halo_rcvbuf[halo_base];
          handle.aa_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 1];
          handle.cc_r[data_base]     = handle.halo_rcvbuf[halo_base + 2];
          handle.cc_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 3];
          handle.dd_r[data_base]     = handle.halo_rcvbuf[halo_base + 4];
          handle.dd_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 5];
        }
      }
    }
    
    // Compute reduced system
    if(mpi_handle.coords[1] == 0) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys_l[1]; id++) {
        int base = id * handle.sys_len_l[1];
        thomas_on_reduced<double>(&handle.aa_r[base], &handle.cc_r[base], &handle.dd_r[base],
                                 handle.sys_len_l[1], 1);
      }
    }
    
    // Pack boundary solution data
    if(mpi_handle.coords[1] == 0) {
      #pragma omp parallel for
      for(int p = 0; p < mpi_handle.pdims[1]; p++) {
        for(int id = 0; id < handle.n_sys_l[1]; id++) {
          int halo_base = p * handle.n_sys_l[1] * 2 + id * 2;
          int data_base = id * handle.sys_len_l[1] + p * 2;
          handle.halo_rcvbuf[halo_base]     = handle.dd_r[data_base];
          handle.halo_rcvbuf[halo_base + 1] = handle.dd_r[data_base + 1];
        }
      }
    }
    
    // Send back new values
    MPI_Scatter(handle.halo_rcvbuf, handle.n_sys_l[1]*2, MPI_DOUBLE, handle.halo_sndbuf,
                handle.n_sys_l[1]*2, MPI_DOUBLE, 0, mpi_handle.y_comm);
    
    // Unpack boundary solution
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[1]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] * handle.pads[1] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * (handle.size[1] - 1));
      int halo_base = id * 2;
      handle.dd[start] = handle.halo_sndbuf[halo_base];
      handle.dd[end]   = handle.halo_sndbuf[halo_base + 1];
    }
    
    // Do the backward pass of modified Thomas
    #pragma omp parallel for
    for(int z = 0; z < handle.size[2]; z++) {
      int base = z * handle.pads[0] * handle.pads[1];
      thomas_backward_vec_strip<double>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                       &handle.h_u[base], handle.size[1], handle.pads[0],
                                       handle.size[0]);
    }
  } else {
    /*********************
     * 
     * Z Dimension Solve
     * 
     *********************/
    
    // Do modified thomas forward pass
    #pragma omp parallel for
    for(int y = 0; y < handle.size[1]; y++) {
      int base = y * handle.pads[0];
      thomas_forward_vec_strip<double>(&handle.a[base], &handle.b[base], &handle.c[base],
                               &handle.du[base], &handle.h_u[base], &handle.aa[base],
                               &handle.cc[base], &handle.dd[base], handle.size[2],
                               handle.pads[0] * handle.pads[1], handle.size[0]);
    }
    
    // Pack boundary values
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[2]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * handle.pads[1] * (handle.size[2] - 1));
      int halo_base = id * 6;
      // Gather coefficients of a,c,d
      handle.halo_sndbuf[halo_base]     = handle.aa[start];
      handle.halo_sndbuf[halo_base + 1] = handle.aa[end];
      handle.halo_sndbuf[halo_base + 2] = handle.cc[start];
      handle.halo_sndbuf[halo_base + 3] = handle.cc[end];
      handle.halo_sndbuf[halo_base + 4] = handle.dd[start];
      handle.halo_sndbuf[halo_base + 5] = handle.dd[end];
    }
    
    // Communicate boundary values
    MPI_Gather(handle.halo_sndbuf, handle.n_sys_l[2]*3*2, MPI_DOUBLE, handle.halo_rcvbuf,
               handle.n_sys_l[2]*3*2, MPI_DOUBLE, 0, mpi_handle.z_comm);
    
    // Unpack boundary data
    if(mpi_handle.coords[2] == 0) {
      #pragma omp parallel for collapse(2)
      for(int p = 0; p < mpi_handle.pdims[2]; p++) {
        for(int id = 0; id < handle.n_sys_l[2]; id++) {
          int data_base = id * handle.sys_len_l[2] + p * 2;
          int halo_base = p * handle.n_sys_l[2] * 6 + id * 6;
          handle.aa_r[data_base]     = handle.halo_rcvbuf[halo_base];
          handle.aa_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 1];
          handle.cc_r[data_base]     = handle.halo_rcvbuf[halo_base + 2];
          handle.cc_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 3];
          handle.dd_r[data_base]     = handle.halo_rcvbuf[halo_base + 4];
          handle.dd_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 5];
        }
      }
    }
    
    // Compute reduced system
    if(mpi_handle.coords[2] == 0) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys_l[2]; id++) {
        int base = id * handle.sys_len_l[2];
        thomas_on_reduced<double>(&handle.aa_r[base], &handle.cc_r[base], &handle.dd_r[base],
                                 handle.sys_len_l[2], 1);
      }
    }
    
    // Pack boundary solution data
    if(mpi_handle.coords[2] == 0) {
      #pragma omp parallel for
      for(int p = 0; p < mpi_handle.pdims[2]; p++) {
        for(int id = 0; id < handle.n_sys_l[2]; id++) {
          int halo_base = p * handle.n_sys_l[2] * 2 + id * 2;
          int data_base = id * handle.sys_len_l[2] + p * 2;
          handle.halo_rcvbuf[halo_base]     = handle.dd_r[data_base];
          handle.halo_rcvbuf[halo_base + 1] = handle.dd_r[data_base + 1];
        }
      }
    }
    
    // Send back new values
    MPI_Scatter(handle.halo_rcvbuf, handle.n_sys_l[2]*2, MPI_DOUBLE, handle.halo_sndbuf,
                handle.n_sys_l[2]*2, MPI_DOUBLE, 0, mpi_handle.z_comm);
    
    // Unpack boundary solution
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[2]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * handle.pads[1] * (handle.size[2] - 1));
      int halo_base = id * 2;
      handle.dd[start] = handle.halo_sndbuf[halo_base];
      handle.dd[end]   = handle.halo_sndbuf[halo_base + 1];
    }
    
    // Do the backward pass of modified Thomas
    #pragma omp parallel for
    for(int y = 0; y < handle.size[1]; y++) {
      int base = y * handle.pads[0];
      thomas_backward_vec_strip<double>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                       &handle.h_u[base], handle.size[2], 
                                       handle.pads[0] * handle.pads[1], handle.size[0]);
    }
  }
}

void tridBatchDInc(tridD_handle &handle, trid_mpi_handle &mpi_handle, int solveDim) {
  if(solveDim == 0) {
    /*********************
     * 
     * X Dimension Solve
     * 
     *********************/
    
    // Do modified thomas forward pass
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[0]; id++) {
      int base = id * handle.pads[0];
      thomas_forward<double>(&handle.a[base], &handle.b[base], &handle.c[base], &handle.du[base],
                     &handle.h_u[base], &handle.aa[base], &handle.cc[base],
                     &handle.dd[base], handle.size[0], 1);
    }
    
    // Pack boundary values
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[0]; id++) {
      // Gather coefficients of a,c,d
      int halo_base = id * 6;
      int data_base = id * handle.pads[0];
      handle.halo_sndbuf[halo_base]     = handle.aa[data_base];
      handle.halo_sndbuf[halo_base + 1] = handle.aa[data_base + handle.size[0]-1];
      handle.halo_sndbuf[halo_base + 2] = handle.cc[data_base];
      handle.halo_sndbuf[halo_base + 3] = handle.cc[data_base + handle.size[0]-1];
      handle.halo_sndbuf[halo_base + 4] = handle.dd[data_base];
      handle.halo_sndbuf[halo_base + 5] = handle.dd[data_base + handle.size[0]-1];
    }
    
    // Communicate boundary values
    MPI_Gather(handle.halo_sndbuf, handle.n_sys_l[0]*3*2, MPI_DOUBLE, handle.halo_rcvbuf,
               handle.n_sys_l[0]*3*2, MPI_DOUBLE, 0, mpi_handle.x_comm);
    
    // Unpack boundary values
    if(mpi_handle.coords[0] == 0) {
      #pragma omp parallel for collapse(2)
      for(int p = 0; p < mpi_handle.pdims[0]; p++) {
        for(int id = 0; id < handle.n_sys_l[0]; id++) {
          int halo_base = p * handle.n_sys_l[0] * 6 + id * 6;
          int data_base = id * handle.sys_len_l[0] + p * 2;
          handle.aa_r[data_base]     = handle.halo_rcvbuf[halo_base];
          handle.aa_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 1];
          handle.cc_r[data_base]     = handle.halo_rcvbuf[halo_base + 2];
          handle.cc_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 3];
          handle.dd_r[data_base]     = handle.halo_rcvbuf[halo_base + 4];
          handle.dd_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 5];
        }
      }
    }
    
    // Compute reduced system
    if(mpi_handle.coords[0] == 0) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys_l[0]; id++) {
        int base = id * handle.sys_len_l[0];
        thomas_on_reduced<double>(&handle.aa_r[base], &handle.cc_r[base], &handle.dd_r[base],
                          handle.sys_len_l[0], 1);
      }
    }
    
    // Pack boundary solution data
    if(mpi_handle.coords[0] == 0) {
      #pragma omp parallel for
      for(int p = 0; p < mpi_handle.pdims[0]; p++) {
        for(int id = 0; id < handle.n_sys_l[0]; id++) {
          int halo_base = p * handle.n_sys_l[0] * 2 + id * 2;
          int data_base = id * handle.sys_len_l[0] + p * 2;
          handle.halo_rcvbuf[halo_base]     = handle.dd_r[data_base];
          handle.halo_rcvbuf[halo_base + 1] = handle.dd_r[data_base + 1];
        }
      }
    }
    
    // Send back new values
    MPI_Scatter(handle.halo_rcvbuf, handle.n_sys_l[0] * 2, MPI_DOUBLE, handle.halo_sndbuf,
                handle.n_sys_l[0] * 2, MPI_DOUBLE, 0, mpi_handle.x_comm);
    
    // Unpack boundary solution
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[0]; id++) {
      // Gather coefficients of a,c,d
      int data_base = id * handle.pads[0];
      int halo_base = id * 2;
      handle.dd[data_base]            = handle.halo_sndbuf[halo_base];
      handle.dd[data_base + handle.size[0]-1] = handle.halo_sndbuf[halo_base + 1];
    }
    
    // Do the backward pass of modified Thomas
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[0]; id++) {
      int ind = id * handle.pads[0];
      thomas_backwardInc<double>(&handle.aa[ind], &handle.cc[ind], &handle.dd[ind], 
                      &handle.h_u[ind], handle.size[0], 1);
    }
  } else if(solveDim == 1) {
    /*********************
     * 
     * Y Dimension Solve
     * 
     *********************/
    
    // Do modified thomas forward pass
    #pragma omp parallel for
    for(int z = 0; z < handle.size[2]; z++) {
      int base = z * handle.pads[0] * handle.pads[1];
      thomas_forward_vec_strip<double>(&handle.a[base], &handle.b[base], &handle.c[base],
                               &handle.du[base], &handle.h_u[base], &handle.aa[base],
                               &handle.cc[base], &handle.dd[base], handle.size[1],
                               handle.pads[0], handle.size[0]);
    }
    
    // Pack boundary values
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[1]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] * handle.pads[1] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * (handle.size[1] - 1));
      int halo_base = id * 6;
      // Gather coefficients of a,c,d
      handle.halo_sndbuf[halo_base]     = handle.aa[start];
      handle.halo_sndbuf[halo_base + 1] = handle.aa[end];
      handle.halo_sndbuf[halo_base + 2] = handle.cc[start];
      handle.halo_sndbuf[halo_base + 3] = handle.cc[end];
      handle.halo_sndbuf[halo_base + 4] = handle.dd[start];
      handle.halo_sndbuf[halo_base + 5] = handle.dd[end];
    }
    
    // Communicate boundary values
    MPI_Gather(handle.halo_sndbuf, handle.n_sys_l[1]*3*2, MPI_DOUBLE, handle.halo_rcvbuf,
               handle.n_sys_l[1]*3*2, MPI_DOUBLE, 0, mpi_handle.y_comm);
    
    // Unpack boundary values
    if(mpi_handle.coords[1] == 0) {
      #pragma omp parallel for collapse(2)
      for(int p = 0; p < mpi_handle.pdims[1]; p++) {
        for(int id = 0; id < handle.n_sys_l[1]; id++) {
          int halo_base = p * handle.n_sys_l[1] * 6 + id * 6;
          int data_base = id * handle.sys_len_l[1] + p * 2;
          handle.aa_r[data_base]     = handle.halo_rcvbuf[halo_base];
          handle.aa_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 1];
          handle.cc_r[data_base]     = handle.halo_rcvbuf[halo_base + 2];
          handle.cc_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 3];
          handle.dd_r[data_base]     = handle.halo_rcvbuf[halo_base + 4];
          handle.dd_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 5];
        }
      }
    }
    
    // Compute reduced system
    if(mpi_handle.coords[1] == 0) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys_l[1]; id++) {
        int base = id * handle.sys_len_l[1];
        thomas_on_reduced<double>(&handle.aa_r[base], &handle.cc_r[base], &handle.dd_r[base],
                                 handle.sys_len_l[1], 1);
      }
    }
    
    // Pack boundary solution data
    if(mpi_handle.coords[1] == 0) {
      #pragma omp parallel for
      for(int p = 0; p < mpi_handle.pdims[1]; p++) {
        for(int id = 0; id < handle.n_sys_l[1]; id++) {
          int halo_base = p * handle.n_sys_l[1] * 2 + id * 2;
          int data_base = id * handle.sys_len_l[1] + p * 2;
          handle.halo_rcvbuf[halo_base]     = handle.dd_r[data_base];
          handle.halo_rcvbuf[halo_base + 1] = handle.dd_r[data_base + 1];
        }
      }
    }
    
    // Send back new values
    MPI_Scatter(handle.halo_rcvbuf, handle.n_sys_l[1]*2, MPI_DOUBLE, handle.halo_sndbuf,
                handle.n_sys_l[1]*2, MPI_DOUBLE, 0, mpi_handle.y_comm);
    
    // Unpack boundary solution
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[1]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] * handle.pads[1] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * (handle.size[1] - 1));
      int halo_base = id * 2;
      handle.dd[start] = handle.halo_sndbuf[halo_base];
      handle.dd[end]   = handle.halo_sndbuf[halo_base + 1];
    }
    
    // Do the backward pass of modified Thomas
    #pragma omp parallel for
    for(int z = 0; z < handle.size[2]; z++) {
      int base = z * handle.pads[0] * handle.pads[1];
      thomas_backwardInc_vec_strip<double>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                       &handle.h_u[base], handle.size[1], handle.pads[0],
                                       handle.size[0]);
    }
  } else {
    /*********************
     * 
     * Z Dimension Solve
     * 
     *********************/
    
    // Do modified thomas forward pass
    #pragma omp parallel for
    for(int y = 0; y < handle.size[1]; y++) {
      int base = y * handle.pads[0];
      thomas_forward_vec_strip<double>(&handle.a[base], &handle.b[base], &handle.c[base],
                               &handle.du[base], &handle.h_u[base], &handle.aa[base],
                               &handle.cc[base], &handle.dd[base], handle.size[2],
                               handle.pads[0] * handle.pads[1], handle.size[0]);
    }
    
    // Pack boundary values
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[2]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * handle.pads[1] * (handle.size[2] - 1));
      int halo_base = id * 6;
      // Gather coefficients of a,c,d
      handle.halo_sndbuf[halo_base]     = handle.aa[start];
      handle.halo_sndbuf[halo_base + 1] = handle.aa[end];
      handle.halo_sndbuf[halo_base + 2] = handle.cc[start];
      handle.halo_sndbuf[halo_base + 3] = handle.cc[end];
      handle.halo_sndbuf[halo_base + 4] = handle.dd[start];
      handle.halo_sndbuf[halo_base + 5] = handle.dd[end];
    }
    
    // Communicate boundary values
    MPI_Gather(handle.halo_sndbuf, handle.n_sys_l[2]*3*2, MPI_DOUBLE, handle.halo_rcvbuf,
               handle.n_sys_l[2]*3*2, MPI_DOUBLE, 0, mpi_handle.z_comm);
    
    // Unpack boundary data
    if(mpi_handle.coords[2] == 0) {
      #pragma omp parallel for collapse(2)
      for(int p = 0; p < mpi_handle.pdims[2]; p++) {
        for(int id = 0; id < handle.n_sys_l[2]; id++) {
          int data_base = id * handle.sys_len_l[2] + p * 2;
          int halo_base = p * handle.n_sys_l[2] * 6 + id * 6;
          handle.aa_r[data_base]     = handle.halo_rcvbuf[halo_base];
          handle.aa_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 1];
          handle.cc_r[data_base]     = handle.halo_rcvbuf[halo_base + 2];
          handle.cc_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 3];
          handle.dd_r[data_base]     = handle.halo_rcvbuf[halo_base + 4];
          handle.dd_r[data_base + 1] = handle.halo_rcvbuf[halo_base + 5];
        }
      }
    }
    
    // Compute reduced system
    if(mpi_handle.coords[2] == 0) {
      #pragma omp parallel for
      for(int id = 0; id < handle.n_sys_l[2]; id++) {
        int base = id * handle.sys_len_l[2];
        thomas_on_reduced<double>(&handle.aa_r[base], &handle.cc_r[base], &handle.dd_r[base],
                                 handle.sys_len_l[2], 1);
      }
    }
    
    // Pack boundary solution data
    if(mpi_handle.coords[2] == 0) {
      #pragma omp parallel for
      for(int p = 0; p < mpi_handle.pdims[2]; p++) {
        for(int id = 0; id < handle.n_sys_l[2]; id++) {
          int halo_base = p * handle.n_sys_l[2] * 2 + id * 2;
          int data_base = id * handle.sys_len_l[2] + p * 2;
          handle.halo_rcvbuf[halo_base]     = handle.dd_r[data_base];
          handle.halo_rcvbuf[halo_base + 1] = handle.dd_r[data_base + 1];
        }
      }
    }
    
    // Send back new values
    MPI_Scatter(handle.halo_rcvbuf, handle.n_sys_l[2]*2, MPI_DOUBLE, handle.halo_sndbuf,
                handle.n_sys_l[2]*2, MPI_DOUBLE, 0, mpi_handle.z_comm);
    
    // Unpack boundary solution
    #pragma omp parallel for
    for(int id = 0; id < handle.n_sys_g[2]; id++) {
      int start = (id/handle.size[0]) * handle.pads[0] + (id % handle.size[0]);
      int end = start + (handle.pads[0] * handle.pads[1] * (handle.size[2] - 1));
      int halo_base = id * 2;
      handle.dd[start] = handle.halo_sndbuf[halo_base];
      handle.dd[end]   = handle.halo_sndbuf[halo_base + 1];
    }
    
    // Do the backward pass of modified Thomas
    #pragma omp parallel for
    for(int y = 0; y < handle.size[1]; y++) {
      int base = y * handle.pads[0];
      thomas_backwardInc_vec_strip<double>(&handle.aa[base], &handle.cc[base], &handle.dd[base],
                                       &handle.h_u[base], handle.size[2], 
                                       handle.pads[0] * handle.pads[1], handle.size[0]);
    }
  }
}
