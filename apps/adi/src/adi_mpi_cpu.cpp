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

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <getopt.h>
#include <float.h>
#define FP double

#include "adi_cpu.h"
#include "adi_mpi.h"
#include "preproc_mpi.hpp"
#include "trid_mpi_cpu.h"

#include "trid_common.h"

#include "omp.h"
//#include "offload.h"
#include "mpi.h"

#ifdef __MKL__
  //#include "lapacke.h"
  #include "mkl_lapacke.h"
  //#include "mkl.h"
#endif

#define ROUND_DOWN(N,step) (((N)/(step))*step)
#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))
#define MAX(X,Y) ((X) > (Y) ? (X) : (Y))

extern char *optarg;
extern int  optind, opterr, optopt;
static struct option options[] = {
  {"nx",   required_argument, 0,  0   },
  {"ny",   required_argument, 0,  0   },
  {"nz",   required_argument, 0,  0   },
  {"iter", required_argument, 0,  0   },
  {"opt",  required_argument, 0,  0   },
  {"prof", required_argument, 0,  0   },
  {"help", no_argument,       0,  'h' },
  {0,      0,                 0,  0   }
};

/*
 * Print essential infromation on the use of the program
 */
void print_help() {
  printf("Please specify the ADI configuration, e.g.: \n$ ./adi_* -nx NX -ny NY -nz NZ -iter ITER [-opt CUDAOPT] -prof PROF\n");
  exit(0);
}

void rms(char* name, FP* array, app_handle &app, mpi_handle &mpi) {
  //Sum the square of values in app.h_u
  double sum = 0.0;
  for(int k=0; k<app.nz; k++) {
    for(int j=0; j<app.ny; j++) {
      for(int i=0; i<app.nx; i++) {
        int ind = k*app.nx_pad*app.ny + j*app.nx_pad + i;
        //sum += array[ind]*array[ind];
        sum += array[ind];
      }
    }
  }

  double global_sum = 0.0;
  MPI_Allreduce(&sum, &global_sum,1, MPI_DOUBLE,MPI_SUM, mpi.comm);

  if(mpi.rank ==0) {
    printf("%s sum = %lg\n", name, global_sum);
    //printf("%s rms = %2.15lg\n",name, sqrt(global_sum)/((double)(app.nx_g*app.ny_g*app.nz_g)));
  }

}

void print_array_onrank(int rank, FP* array, app_handle &app, mpi_handle &mpi) {
  if(mpi.rank == rank) {
    printf("On mpi rank %d\n",rank);
    for(int k=0; k<2; k++) {
        printf("k = %d\n",k);
        for(int j=0; j<MIN(app.ny,17); j++) {
          printf(" %d   ", j);
          for(int i=0; i<MIN(app.nx,17); i++) {
            int ind = k*app.nx_pad*app.ny + j*app.nx_pad + i;
            printf(" %5.5g ", array[ind]);
          }
          printf("\n");
        }
        printf("\n");
      }
  }
}

int init(trid_handle<FP> &trid_handle, trid_mpi_handle &mpi_handle, preproc_handle<FP> &pre_handle, int argc, char* argv[]) {
  if( MPI_Init(&argc,&argv) != MPI_SUCCESS) { printf("MPI Couldn't initialize. Exiting"); exit(-1);}

  //int nx, ny, nz, iter, opt, prof;
  int nx_g = 256;
  int ny_g = 256;
  int nz_g = 256;
  int iter = 10;
  int opt  = 0;
  int prof = 1;

  pre_handle.lambda = 1.0f;

  // Process arguments
  int opt_index = 0;
  while( getopt_long_only(argc, argv, "", options, &opt_index) != -1) {
    if(strcmp((char*)options[opt_index].name,"nx"  ) == 0) nx_g = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"ny"  ) == 0) ny_g = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"nz"  ) == 0) nz_g = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"iter") == 0) iter = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"opt" ) == 0) opt  = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"prof") == 0) prof = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"help") == 0) print_help();
  }
  
  int size[3] = {nx_g, ny_g, nz_g};
  
  tridInit<FP>(trid_handle, mpi_handle, 3, size);

  if(mpi_handle.rank==0) {
    printf("\nGlobal grid dimensions: %d x %d x %d\n", 
           trid_handle.size_g[0], trid_handle.size_g[1], trid_handle.size_g[2]);

    printf("\nNumber of MPI procs in each dimenstion %d, %d, %d\n",
           mpi_handle.pdims[0], mpi_handle.pdims[1], mpi_handle.pdims[2]);
  }

  printf("Check parameters: SIMD_WIDTH = %d, sizeof(FP) = %d\n", SIMD_WIDTH, sizeof(FP));
  printf("Check parameters: nx_pad (padded) = %d\n", trid_handle.pads[0]);
  printf("Check parameters: nx = %d, x_start_g = %d, x_end_g = %d \n", 
         trid_handle.size[0], trid_handle.start_g[0], trid_handle.end_g[0]);
  printf("Check parameters: ny = %d, y_start_g = %d, y_end_g = %d \n", 
         trid_handle.size[1], trid_handle.start_g[1], trid_handle.end_g[1]);
  printf("Check parameters: nz = %d, z_start_g = %d, z_end_g = %d \n",
         trid_handle.size[2], trid_handle.start_g[2], trid_handle.end_g[2]);
  
  // Initialize
  for(int k = 0; k < trid_handle.size[2]; k++) {
    for(int j = 0; j < trid_handle.size[1]; j++) {
      for(int i = 0; i < trid_handle.size[0]; i++) {
        int ind = k * trid_handle.pads[0] * trid_handle.pads[1] + j*trid_handle.pads[0] + i;
        if( (trid_handle.start_g[0]==0 && i==0) || 
            (trid_handle.end_g[0]==trid_handle.size_g[0]-1 && i==trid_handle.size[0]-1) ||
            (trid_handle.start_g[1]==0 && j==0) || 
            (trid_handle.end_g[1]==trid_handle.size_g[1]-1 && j==trid_handle.size[1]-1) ||
            (trid_handle.start_g[2]==0 && k==0) || 
            (trid_handle.end_g[2]==trid_handle.size_g[2]-1 && k==trid_handle.size[2]-1)) {
          trid_handle.h_u[ind] = 1.0f;
        } else {
          trid_handle.h_u[ind] = 0.0f;
        }
      }
    }
  }
  
  pre_handle.halo_snd_x = (FP*) _mm_malloc(2 * trid_handle.size[1] * trid_handle.size[2] * sizeof(FP), SIMD_WIDTH);
  pre_handle.halo_rcv_x = (FP*) _mm_malloc(2 * trid_handle.size[1] * trid_handle.size[2] * sizeof(FP), SIMD_WIDTH);
  pre_handle.halo_snd_y = (FP*) _mm_malloc(2 * trid_handle.size[0] * trid_handle.size[2] * sizeof(FP), SIMD_WIDTH);
  pre_handle.halo_rcv_y = (FP*) _mm_malloc(2 * trid_handle.size[0] * trid_handle.size[2] * sizeof(FP), SIMD_WIDTH);
  pre_handle.halo_snd_z = (FP*) _mm_malloc(2 * trid_handle.size[1] * trid_handle.size[0] * sizeof(FP), SIMD_WIDTH);
  pre_handle.halo_rcv_z = (FP*) _mm_malloc(2 * trid_handle.size[1] * trid_handle.size[0] * sizeof(FP), SIMD_WIDTH);

  return 0;

}

void finalize(trid_handle<FP> &trid_handle, trid_mpi_handle &mpi_handle, preproc_handle<FP> &pre_handle) {
  tridClean<FP>(trid_handle, mpi_handle);
  _mm_free(pre_handle.halo_snd_x);
  _mm_free(pre_handle.halo_rcv_x);
  _mm_free(pre_handle.halo_snd_y);
  _mm_free(pre_handle.halo_rcv_y);
  _mm_free(pre_handle.halo_snd_z);
  _mm_free(pre_handle.halo_rcv_z);
}

int main(int argc, char* argv[]) {
  trid_mpi_handle mpi_handle;
  trid_handle<FP> trid_handle;
  preproc_handle<FP> pre_handle;
  int ret;
  init(trid_handle, mpi_handle, pre_handle, argc, argv);

  // Declare and reset elapsed time counters
  double timer           = 0.0;
  double timer1          = 0.0;
  double timer2          = 0.0;
  double elapsed         = 0.0;
  double elapsed_total   = 0.0;
  double elapsed_preproc = 0.0;
  double elapsed_trid_x  = 0.0;
  double elapsed_trid_y  = 0.0;
  double elapsed_trid_z  = 0.0;

//#define TIMERS 11
//  double elapsed_time[TIMERS];
//  double   timers_min[TIMERS];
//  double   timers_max[TIMERS];
//  double   timers_avg[TIMERS];
//  char   elapsed_name[TIMERS][256] = {"forward","halo1","alltoall1","halo2","reduced","halo3","alltoall2","halo4","backward","pre_mpi","pre_comp"};
  /*strcpy(app.elapsed_name[ 0], "forward");
  strcpy(app.elapsed_name[ 1], "halo1");
  strcpy(app.elapsed_name[ 2], "gather");
  strcpy(app.elapsed_name[ 3], "halo2");
  strcpy(app.elapsed_name[ 4], "reduced");
  strcpy(app.elapsed_name[ 5], "halo3");
  strcpy(app.elapsed_name[ 6], "scatter");
  strcpy(app.elapsed_name[ 7], "halo4");
  strcpy(app.elapsed_name[ 8], "backward");
  strcpy(app.elapsed_name[ 9], "pre_mpi");
  strcpy(app.elapsed_name[10], "pre_comp");*/
  double elapsed_forward   = 0.0;
  double elapsed_reduced   = 0.0;
  double elapsed_backward  = 0.0;
  double elapsed_alltoall1 = 0.0;
  double elapsed_alltoall2 = 0.0;
  double elapsed_halo1     = 0.0;
  double elapsed_halo2     = 0.0;
  double elapsed_halo3     = 0.0;
  double elapsed_halo4     = 0.0;

  /*for(int i=0; i<TIMERS; i++) {
    app.elapsed_time_x[i] = 0.0;
    app.elapsed_time_y[i] = 0.0;
    app.elapsed_time_z[i] = 0.0;
    app.timers_min[i]   = DBL_MAX;
    app.timers_avg[i]   = 0.0;
    app.timers_max[i]   = 0.0;
  }*/

  // Warm up computation: result stored in h_tmp which is not used later
  //preproc<FP>(lambda, h_tmp, h_du, h_ax, h_bx, h_cx, h_ay, h_by, h_cy, h_az, h_bz, h_cz, nx, nx_pad, ny, nz);

  //int i, j, k, ind, it;
  //
  // calculate r.h.s. and set tri-diagonal coefficients
  //
  
  //timing_start(app.prof, &timer1);

  for(int it=0; it</*app.iter*/10; it++) {

    MPI_Barrier(MPI_COMM_WORLD);
    //timing_start(app.prof, &timer);
        preproc_mpi<FP>(pre_handle, trid_handle.h_u, trid_handle.du, trid_handle.a,
                    trid_handle.b, trid_handle.c, trid_handle, mpi_handle);
        
    printf("Preproc\n");
    
    //timing_end(app.prof, &timer, &elapsed_preproc, "preproc");

    //
    // perform tri-diagonal solves in x-direction
    //
    MPI_Barrier(MPI_COMM_WORLD);
    //timing_start(app.prof, &timer);
    
    tridBatch<FP, 1>(trid_handle, mpi_handle, 0);

    printf("X\n");
    
    //timing_end(app.prof, &timer, &elapsed_trid_x, "trid-x");
  
    //rms("post x h_u", app.h_u, app, mpi);
    //rms("post x du", app.du, app, mpi);

    //
    // perform tri-diagonal solves in y-direction
    //
    
    //MPI_Barrier(MPI_COMM_WORLD);
    //timing_start(app.prof, &timer);

    tridBatch<FP, 1>(trid_handle, mpi_handle, 1);
    
    printf("Y\n");
    
    //timing_end(app.prof, &timer, &elapsed_trid_y, "trid-y");
    
    //rms("post y h_u", app.h_u, app, mpi);
    //rms("post y du", app.du, app, mpi);
    
    //
    // perform tri-diagonal solves in z-direction
    //
    //MPI_Barrier(MPI_COMM_WORLD);
    //timing_start(app.prof, &timer);
    
    tridBatch<FP, 1>(trid_handle, mpi_handle, 2);
    
    printf("Z\n");
    
    //timing_end(app.prof, &timer, &elapsed_trid_z, "trid-z");
    
    //rms("post z h_u", app.h_u, app, mpi);
    //rms("post z du", app.du, app, mpi);
  }
  
  //timing_end(app.prof, &timer1, &elapsed_total, "total");
  
  //rms("end h_u", app.h_u, app, mpi);
  //rms("end du", app.du, app, mpi);

/*{
  int nx = app.nx;
  int ny = app.ny;
  int nz = app.nz;
  int ldim = app.nx_pad;
  FP *h_u = app.h_u;
  //h_u = du;
  for(int r=0; r<mpi.procs; r++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if(r==mpi.rank) {
      printf("Data on rank = %d +++++++++++++++++++++++\n", mpi.rank);
      #include "print_array.c"
    }
  }
}*/
  // Normalize timers to one iteration
  /*for(int i=0; i<TIMERS; i++) {
    app.elapsed_time_x[i] /= app.iter;
    app.elapsed_time_y[i] /= app.iter;
    app.elapsed_time_z[i] /= app.iter;
  }*/

  MPI_Barrier(MPI_COMM_WORLD);
  /*MPI_Reduce(app.elapsed_time,app.timers_min,TIMERS,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD);
  MPI_Reduce(app.elapsed_time,app.timers_max,TIMERS,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  MPI_Reduce(app.elapsed_time,app.timers_avg,TIMERS,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  for(int i=0; i<TIMERS; i++)
    app.timers_avg[i] /= mpi.procs;*/
  
  /*double avg_total = 0.0;

  MPI_Reduce(app.elapsed_time_x,app.timers_avg,TIMERS,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Reduce(&elapsed_trid_x,&avg_total,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  
  if(mpi.rank == 0) {
    for(int i=0; i<TIMERS; i++)
        app.timers_avg[i] /= mpi.procs;
    
    avg_total /= mpi.procs;
  }
  
  for(int i=0; i<mpi.procs; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    //sleep(0.2);
    if(i==mpi.rank) {
      if(mpi.rank==0) {
        printf("Time in trid-x segments[ms]: \n[total] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[checksum]\n",
            app.elapsed_name[0], app.elapsed_name[1], app.elapsed_name[2], app.elapsed_name[3], app.elapsed_name[4], app.elapsed_name[5], app.elapsed_name[6], app.elapsed_name[7], app.elapsed_name[8]);
      }
      printf("%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf\n",
      1000.0*elapsed_trid_x ,
      1000.0*app.elapsed_time_x[0],
      1000.0*app.elapsed_time_x[1],
      1000.0*app.elapsed_time_x[2],
      1000.0*app.elapsed_time_x[3],
      1000.0*app.elapsed_time_x[4],
      1000.0*app.elapsed_time_x[5],
      1000.0*app.elapsed_time_x[6],
      1000.0*app.elapsed_time_x[7],
      1000.0*app.elapsed_time_x[8],
      1000.0*(app.elapsed_time_x[0] + app.elapsed_time_x[1] + app.elapsed_time_x[2] + app.elapsed_time_x[3] + app.elapsed_time_x[4] + app.elapsed_time_x[5] + app.elapsed_time_x[6] + app.elapsed_time_x[7] + app.elapsed_time_x[8]));
    }
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpi.rank == 0) {
    printf("Avg:\n");
    printf("%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf\n",
        1000.0*avg_total ,
        1000.0*app.timers_avg[0],
        1000.0*app.timers_avg[1],
        1000.0*app.timers_avg[2],
        1000.0*app.timers_avg[3],
        1000.0*app.timers_avg[4],
        1000.0*app.timers_avg[5],
        1000.0*app.timers_avg[6],
        1000.0*app.timers_avg[7],
        1000.0*app.timers_avg[8]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Reduce(app.elapsed_time_y,app.timers_avg,TIMERS,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Reduce(&elapsed_trid_y,&avg_total,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  
  if(mpi.rank == 0) {
    for(int i=0; i<TIMERS; i++)
        app.timers_avg[i] /= mpi.procs;
    
    avg_total /= mpi.procs;
  }
  
  for(int i=0; i<mpi.procs; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    //sleep(0.2);
    if(i==mpi.rank) {
      if(mpi.rank==0) {
        printf("Time in trid-y segments[ms]: \n[total] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[checksum]\n",
            app.elapsed_name[0], app.elapsed_name[1], app.elapsed_name[2], app.elapsed_name[3], app.elapsed_name[4], app.elapsed_name[5], app.elapsed_name[6], app.elapsed_name[7], app.elapsed_name[8]);
      }
      printf("%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf\n",
      1000.0*elapsed_trid_y ,
      1000.0*app.elapsed_time_y[0],
      1000.0*app.elapsed_time_y[1],
      1000.0*app.elapsed_time_y[2],
      1000.0*app.elapsed_time_y[3],
      1000.0*app.elapsed_time_y[4],
      1000.0*app.elapsed_time_y[5],
      1000.0*app.elapsed_time_y[6],
      1000.0*app.elapsed_time_y[7],
      1000.0*app.elapsed_time_y[8],
      1000.0*(app.elapsed_time_y[0] + app.elapsed_time_y[1] + app.elapsed_time_y[2] + app.elapsed_time_y[3] + app.elapsed_time_y[4] + app.elapsed_time_y[5] + app.elapsed_time_y[6] + app.elapsed_time_y[7] + app.elapsed_time_y[8]));
    }
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpi.rank == 0) {
    printf("Avg:\n");
    printf("%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf\n",
        1000.0*avg_total ,
        1000.0*app.timers_avg[0],
        1000.0*app.timers_avg[1],
        1000.0*app.timers_avg[2],
        1000.0*app.timers_avg[3],
        1000.0*app.timers_avg[4],
        1000.0*app.timers_avg[5],
        1000.0*app.timers_avg[6],
        1000.0*app.timers_avg[7],
        1000.0*app.timers_avg[8]);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Reduce(app.elapsed_time_z,app.timers_avg,TIMERS,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Reduce(&elapsed_trid_z,&avg_total,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  
  if(mpi.rank == 0) {
    for(int i=0; i<TIMERS; i++)
        app.timers_avg[i] /= mpi.procs;
    
    avg_total /= mpi.procs;
  }
  
  for(int i=0; i<mpi.procs; i++) {
    MPI_Barrier(MPI_COMM_WORLD);
    //sleep(0.2);
    if(i==mpi.rank) {
      if(mpi.rank==0) {
        printf("Time in trid-z segments[ms]: \n[total] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[%s] \t[checksum]\n",
            app.elapsed_name[0], app.elapsed_name[1], app.elapsed_name[2], app.elapsed_name[3], app.elapsed_name[4], app.elapsed_name[5], app.elapsed_name[6], app.elapsed_name[7], app.elapsed_name[8]);
      }
      printf("%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf\n",
      1000.0*elapsed_trid_z ,
      1000.0*app.elapsed_time_z[0],
      1000.0*app.elapsed_time_z[1],
      1000.0*app.elapsed_time_z[2],
      1000.0*app.elapsed_time_z[3],
      1000.0*app.elapsed_time_z[4],
      1000.0*app.elapsed_time_z[5],
      1000.0*app.elapsed_time_z[6],
      1000.0*app.elapsed_time_z[7],
      1000.0*app.elapsed_time_z[8],
      1000.0*(app.elapsed_time_z[0] + app.elapsed_time_z[1] + app.elapsed_time_z[2] + app.elapsed_time_z[3] + app.elapsed_time_z[4] + app.elapsed_time_z[5] + app.elapsed_time_z[6] + app.elapsed_time_z[7] + app.elapsed_time_z[8]));
    }
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpi.rank == 0) {
    printf("Avg:\n");
    printf("%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf \t%lf\n",
        1000.0*avg_total ,
        1000.0*app.timers_avg[0],
        1000.0*app.timers_avg[1],
        1000.0*app.timers_avg[2],
        1000.0*app.timers_avg[3],
        1000.0*app.timers_avg[4],
        1000.0*app.timers_avg[5],
        1000.0*app.timers_avg[6],
        1000.0*app.timers_avg[7],
        1000.0*app.timers_avg[8]);
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  if(mpi.rank == 0) {
    // Print execution times
    if(app.prof == 0) {
      printf("Avg(per iter) \n[total]\n");
      printf("%f\n", elapsed_total/app.iter);
    }
    else if(app.prof == 1) {
    printf("Time per section: \n[total] \t[prepro] \t[trid_x] \t[trid_y] \t[trid_z]\n");
    printf("%e \t%e \t%e \t%e \t%e\n",
        elapsed_total,
        elapsed_preproc,
        elapsed_trid_x,
        elapsed_trid_y,
        elapsed_trid_z);
    printf("Time per element averaged on %d iterations: \n[total] \t[prepro] \t[trid_x] \t[trid_y] \t[trid_z]\n", app.iter);
    printf("%e \t%e \t%e \t%e \t%e\n",
        (elapsed_total/app.iter  ) / (app.nx_g * app.ny_g * app.nz_g),
        (elapsed_preproc/app.iter) / (app.nx_g * app.ny_g * app.nz_g),
        (elapsed_trid_x/app.iter ) / (app.nx_g * app.ny_g * app.nz_g),
        (elapsed_trid_y/app.iter ) / (app.nx_g * app.ny_g * app.nz_g),
        (elapsed_trid_z/app.iter ) / (app.nx_g * app.ny_g * app.nz_g));
    }
  }
  finalize(app, mpi);*/
  
  finalize(trid_handle, mpi_handle, pre_handle);
  
  MPI_Finalize();
  return 0;

}
