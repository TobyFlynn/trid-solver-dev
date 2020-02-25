#include <stdio.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <limits>

#include "mpi.h"

typedef std::numeric_limits< double > dbl;

extern char *optarg;
extern int  optind, opterr, optopt;
static struct option options[] = {
  {"nx",   required_argument, 0,  0   },
  {"ny",   required_argument, 0,  0   },
  {"nz",   required_argument, 0,  0   },
  {"iter", required_argument, 0,  0   }
};

int main(int argc, char* argv[]) {
  if( MPI_Init(&argc,&argv) != MPI_SUCCESS) { printf("MPI Couldn't initialize. Exiting"); exit(-1);}
  
  int nx = 0;
  int ny = 0;
  int nz = 0;
  int iter = 0;
  
  // Get size and number of files to process
  int opt_index = 0;
  while( getopt_long_only(argc, argv, "", options, &opt_index) != -1) {
    if(strcmp((char*)options[opt_index].name,"nx"  ) == 0) nx = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"ny"  ) == 0) ny = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"nz"  ) == 0) nz = atoi(optarg);
    if(strcmp((char*)options[opt_index].name,"iter") == 0) iter = atoi(optarg);
  }
  
  // Allocate array to store one iterations array
  int numElements = nx * ny * nz;
  double *array = (double *) malloc(numElements * sizeof(double));
  
  MPI_Datatype subArrayType;
  int size_g[3] = {nx, ny, nz};
  int size[3] = {nx, ny, nz};
  int start_g[3] = {0, 0, 0};
  MPI_Type_create_subarray(3, size_g, size, start_g, MPI_ORDER_FORTRAN, MPI_DOUBLE, &subArrayType);
  MPI_Type_commit(&subArrayType);
  
  // Cycle through each file and process it into a csv file
  for(int i = 0; i < iter; i++) {
    // Create input and output filenames
    std::string inFilename = "mpi-i-" + std::to_string(i) + ".out";
    std::string outFilename = "mpi-i-" + std::to_string(i) + ".csv";
    
    // Read in the MPI file
    MPI_File fh;
    MPI_File_open(MPI_COMM_WORLD, inFilename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    MPI_Offset disp = 0;
    MPI_File_set_view(fh, disp, MPI_DOUBLE, subArrayType, "native", MPI_INFO_NULL);
    MPI_Status status;
    MPI_File_read_all(fh, array, numElements, MPI_DOUBLE, &status);
    MPI_File_close(&fh);
    
    // Write out the file as a csv file
    std::ofstream outFile;
    outFile.open(outFilename.c_str());
    outFile << std::fixed << std::setprecision(dbl::max_digits10);
    for(int z = 0; z < nz; z++) {
      for(int y = 0; y < ny; y++) {
        for(int x = 0; x < nx; x++) {
          int ind = z * ny * nx + y * nx + x;
          outFile << array[ind] << " ";
        }
        outFile << "\n";
      }
    }
    outFile.close();
  }
  
  return 0;
}
