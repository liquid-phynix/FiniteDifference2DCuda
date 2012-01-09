#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<unistd.h>
#include<sys/poll.h>
#include<sys/time.h>
#include<cuda.h>

#include<pthread.h>
#include<readline/readline.h>
#include<readline/history.h>


#include"tex.h"
#include"eqns.cu"
#include"vtk_writer_lib.h"

const char usageInfo[] =
"q: quit\n"
"h: help";
bool runSim = true;
pthread_t workerTh, controlTh;

tex_t tex1st;
tex_t tex2nd;

enum {UPDATE_PHI, UPDATE_C};

template<int update_field>
__global__ void kernelStencil(dataType* output, bool sem){
  unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;

  int north = row - 1;
  int south = row + 1;
  int east  = col + 1;
  int west  = col - 1;

#if BCOND == PER
  if(north < 0)			north = GD_Y * BD_Y - 1;
  if(south >= (GD_Y * BD_Y))	south = 0;
  if(west < 0)			west = GD_X * BD_X - 1;
  if(east >= (GD_X * BD_X))	east = 0;
#else
  if(north < 0)			north = 0;
  if(south >= (GD_Y * BD_Y))	south = (GD_Y * BD_Y) - 1;
  if(west < 0)			west = 0;
  if(east >= (GD_X * BD_X))	east = (GD_X * BD_X) - 1;
#endif

  if(update_field == UPDATE_PHI){
    dataType tmpTexel;
    // nearest neigbors
    tmpTexel = FETCH_TEX(sem, col, row);
    cellType C_phi = PHI_FROM_TEXEL(tmpTexel);
    cellType C_c   = C_FROM_TEXEL(tmpTexel);
    tmpTexel = FETCH_TEX(sem, west, row);
    cellType W_phi = PHI_FROM_TEXEL(tmpTexel);
    tmpTexel = FETCH_TEX(sem, east, row);
    cellType E_phi = PHI_FROM_TEXEL(tmpTexel);
    tmpTexel = FETCH_TEX(sem, col, north);
    cellType N_phi = PHI_FROM_TEXEL(tmpTexel);
    tmpTexel = FETCH_TEX(sem, col, south);
    cellType S_phi = PHI_FROM_TEXEL(tmpTexel);

#if ANISOTROPY == YES
    tmpTexel = FETCH_TEX(sem, west, north);
    cellType NW_phi = PHI_FROM_TEXEL(tmpTexel);
    tmpTexel = FETCH_TEX(sem, east, north);
    cellType NE_phi = PHI_FROM_TEXEL(tmpTexel);
    tmpTexel = FETCH_TEX(sem, west, south);
    cellType SW_phi = PHI_FROM_TEXEL(tmpTexel);
    tmpTexel = FETCH_TEX(sem, east, south);
    cellType SE_phi = PHI_FROM_TEXEL(tmpTexel);
#endif
    cellType newPhi = C_phi + DT * phiDot(PHIDOT_UNTYPED_ARG_LIST);

#if FTYPE == FLOAT
    output[row * GD_X * BD_X + col].x = newPhi;
#else
    output[row * GD_X * BD_X + col].x = __double2loint(newPhi);
    output[row * GD_X * BD_X + col].y = __double2hiint(newPhi);
#endif
  }
  //UPDATE_C
  else{
    dataType tmpTexel;
    // nearest neigbors
    tmpTexel = FETCH_TEX(sem, col, row);
    cellType C_phi = PHI_FROM_TEXEL(tmpTexel);
    cellType C_c   = C_FROM_TEXEL(tmpTexel);
    tmpTexel = FETCH_TEX(sem, west, row);
    cellType W_phi = PHI_FROM_TEXEL(tmpTexel);
    cellType W_c   = C_FROM_TEXEL(tmpTexel);
    tmpTexel = FETCH_TEX(sem, east, row);
    cellType E_phi = PHI_FROM_TEXEL(tmpTexel);
    cellType E_c   = C_FROM_TEXEL(tmpTexel);
    tmpTexel = FETCH_TEX(sem, col, north);
    cellType N_phi = PHI_FROM_TEXEL(tmpTexel);
    cellType N_c   = C_FROM_TEXEL(tmpTexel);
    tmpTexel = FETCH_TEX(sem, col, south);
    cellType S_phi = PHI_FROM_TEXEL(tmpTexel);
    cellType S_c   = C_FROM_TEXEL(tmpTexel);
    // next-nearest neigbors
    tmpTexel = FETCH_TEX(sem, west, north);
    cellType NW_phi = PHI_FROM_TEXEL(tmpTexel);
    tmpTexel = FETCH_TEX(sem, east, north);
    cellType NE_phi = PHI_FROM_TEXEL(tmpTexel);
    tmpTexel = FETCH_TEX(sem, west, south);
    cellType SW_phi = PHI_FROM_TEXEL(tmpTexel);
    tmpTexel = FETCH_TEX(sem, east, south);
    cellType SE_phi = PHI_FROM_TEXEL(tmpTexel);

    cellType C_phi_new = PHI_FROM_TEXEL(output[row * GD_X * BD_X + col]);
    cellType W_phi_new = PHI_FROM_TEXEL(output[row * GD_X * BD_X + west]);
    cellType E_phi_new = PHI_FROM_TEXEL(output[row * GD_X * BD_X + east]);
    cellType N_phi_new = PHI_FROM_TEXEL(output[north * GD_X * BD_X + col]);
    cellType S_phi_new = PHI_FROM_TEXEL(output[south * GD_X * BD_X + col]);

    cellType newC   = C_c   + DT * cDot(CDOT_UNTYPED_ARG_LIST);
#if FTYPE == FLOAT
    output[row * GD_X * BD_X + col].y = newC;
#else
    output[row * GD_X * BD_X + col].z = __double2loint(newC);
    output[row * GD_X * BD_X + col].w = __double2hiint(newC);
#endif
  }
}

//****************************************************************************************************
// several initial conditions
//****************************************************************************************************

void writeInitialConditionSingleNucleus(float2* fields, int xDim, int yDim){
  int cR = xDim / 2;
  int cC = yDim / 2;
  float r = 0.0;
  float r0 = 2 * 10.0;
  float tmp;
  for(int row = 0; row < xDim; row++){
    for(int col = 0; col < yDim; col++){
      if((r=sqrt((row - cR) * (row - cR) + (col - cC) * (col - cC))) < r0){
	tmp = -tanh((r-r0/2.0)/(M_SQRT2/0.4));
	fields[col * xDim + row].x = tmp;
	fields[col * xDim + row].y = (1.0-(1.0-KPART)*OMEGA)*((1.0+KPART)/2.0-(1.0-KPART)/2.0*tmp);
      }
    }
  }
}

void writeInitialConditionHomogene(float2* output,  int xDim, int yDim, float valField1, float valField2){
  int elements = xDim * yDim;
  for(int index = 0; index < elements; index++){
    output[index].x = valField1;
    output[index].y = valField2;
  }
}

void writeInitialConditionSingleNucleus(int4* fields, int xDim, int yDim){
  int cR = xDim / 2;
  int cC = yDim / 2;
  double r = 0.0;
  double r0 = 2 * 10.0;
  double2 tmp;
  for(int row = 0; row < xDim; row++){
    for(int col = 0; col < yDim; col++){
      if((r=sqrt((row - cR) * (row - cR) + (col - cC) * (col - cC))) < r0){
	tmp.x = -tanh((r-r0/2.0)/(M_SQRT2/0.4));
	tmp.y = (1.0-(1.0-KPART)*OMEGA)*((1.0+KPART)/2.0-(1.0-KPART)/2.0*tmp.x);
	*((double2*)fields + col * xDim + row) = tmp;
      }
    }
  }
}

void writeInitialConditionHomogene(int4* output,  int xDim, int yDim, double valField1, double valField2){
  int elements = xDim * yDim;
  double2 tmp;
  tmp.x = valField1;
  tmp.y = valField2;

  for(int index = 0; index < elements; index++){
    *((double2*)output + index) = tmp;
  }
}


void cumulativeTimerStop(cudaEvent_t startEvent, cudaEvent_t stopEvent, float* cumTm){
  float tm;
  cudaEventRecord(stopEvent);
  cudaEventSynchronize(stopEvent);
  cudaEventElapsedTime(&tm, startEvent, stopEvent);
  *cumTm += tm;
}

void cumulativeTimerStart(cudaEvent_t startEvent){
  cudaEventRecord(startEvent);
}

void saveFields(float2* input, int xDim, int yDim, int fileCounter){
  char fileName[STRINGLENGTH];
  float* field = (float*)malloc(xDim * yDim * sizeof(float));
  for(int index = 0; index < xDim * yDim; index++){
    field[index] = input[index].x;
  }
  writeImageData(genFileName(fileName, "tex_phi_", fileCounter), xDim, yDim, field, false);
  for(int index = 0; index < xDim * yDim; index++){
    field[index] = input[index].y;
  }
  writeImageData(genFileName(fileName, "tex_c_", fileCounter), xDim, yDim, field, 0);
  free(field);
}

void saveFields(int4* input, int xDim, int yDim, int fileCounter){
  char fileName[STRINGLENGTH];
  double2 tmp;

#ifdef SAVEFLOAT
  float* field = (float*)malloc(xDim * yDim * sizeof(float));
#else
  double* field = (double*)malloc(xDim * yDim * sizeof(double));
#endif

  for(int index = 0; index < xDim * yDim; index++){
    tmp = *((double2*)input + index);
    field[index] = tmp.x;
  }
  writeImageData(genFileName(fileName, "tex_phi_", fileCounter), xDim, yDim, field, false);

  for(int index = 0; index < xDim * yDim; index++){
    tmp = *((double2*)input + index);
    field[index] = tmp.y;
  }
  writeImageData(genFileName(fileName, "tex_c_", fileCounter), xDim, yDim, field, 0);

  free(field);
}

char* genFileName(char* fileName, char* prefix, int fileCounter){
  char fileCounterString[STRINGLENGTH];
  memset(fileName, 0, STRINGLENGTH);
  strcat(fileName, prefix);
  sprintf(fileCounterString, "%.4d", fileCounter);
  strcat(fileName, fileCounterString);
  strcat(fileName, ".vti");
  return fileName;
}

void commandHandler(char* command){
  if(command){
    if (strcmp(command,"h") == 0){
      printf("%s\n",usageInfo);
    }
    if (strcmp(command,"q") == 0){
      runSim = false;
    }
    printf("\n~ %s",command);
    if (command[0]!=0)
      add_history(command);
  }
  else{
    runSim = false;
  }
}

void* controlThread(void* ptr){
  pollfd cinfd[1];
  rl_callback_handler_install ("\n$>", commandHandler);

  cinfd[0].fd = fileno(stdin);
  cinfd[0].events = POLLIN;

  rl_bind_key('\t',rl_abort);

  while(runSim){
    if(poll(cinfd, 1, 1)){
      rl_callback_read_char();
    }
  }
  rl_callback_handler_remove();
  return 0;
}


void* workerThread(void* ptr){

  int iterations = ITERS;
  int fileCounter = 0;
  struct timeval startTime, endTime;
  simulationVariables_t<dataType> sVars;

  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);
  float gpuElapsedTime;

  //execution configuration
  dim3 gridDims(GD_X, GD_Y);
  dim3 blockDims(BD_X, BD_Y);

  writeInitialConditionHomogene(sVars.hostFields, GD_X * BD_X, GD_Y * BD_Y,
  				F(-1.0),
  				F(1.0) - (F(1.0) - KPART) * OMEGA);
  writeInitialConditionSingleNucleus(sVars.hostFields, GD_X * BD_X, GD_Y * BD_Y);

  sVars.upload();

  //setting up texture bindings
  tex1st.filterMode = cudaFilterModePoint;
  tex1st.normalized = 0;
  tex2nd.filterMode = cudaFilterModePoint;
  tex2nd.normalized = 0;
  cudaBindTexture2D(NULL, tex1st, sVars.deviceFields1st,
		    channelDescriptor,
		    GD_X * BD_X,
		    GD_Y * BD_Y,
		    GD_X * BD_X * sizeof(dataType));
  cudaBindTexture2D(NULL, tex2nd, sVars.deviceFields2nd,
		    channelDescriptor,
		    GD_X * BD_X,
		    GD_Y * BD_Y,
		    GD_X * BD_X * sizeof(dataType));


  bool semaphore = 0;
  // main loop
  cudaThreadSynchronize();
  printf("initialization done\n");
  gettimeofday(&startTime, NULL);
  while((iterations--) && runSim){
    semaphore = !semaphore;
    if(!TIMEIT){
      printf("%d. iteration\n", fileCounter);
      if((fileCounter % WRITEEVERY == 0) || iterations == 0){
        cudaMemcpy(sVars.hostFields, semaphore ? sVars.deviceFields1st : sVars.deviceFields2nd, sVars.fieldsNBytes, cudaMemcpyDeviceToHost);
	saveFields(sVars.hostFields, GD_X * BD_X, GD_Y * BD_Y, fileCounter);
      }
    }
    cumulativeTimerStart(startEvent);
    kernelStencil<UPDATE_PHI><<<gridDims, blockDims>>>(semaphore ? sVars.deviceFields2nd : sVars.deviceFields1st, semaphore);
    kernelStencil<UPDATE_C><<<gridDims, blockDims>>>(semaphore ? sVars.deviceFields2nd : sVars.deviceFields1st, semaphore);
    cumulativeTimerStop(startEvent, stopEvent, &gpuElapsedTime);
    fileCounter++;
  }
  // timing
  cudaThreadSynchronize();
  gettimeofday(&endTime, NULL);
  printf("GPU timer: %d ms\n", (int)gpuElapsedTime);
  printf("CPU timer: %d ms\n",
	 (int)(((endTime.tv_sec  - startTime.tv_sec) * 1000 
		+ (endTime.tv_usec - startTime.tv_usec)/1000.0) 
	       + 0.5));
  // cleaning up
  cudaUnbindTexture(tex1st);
  cudaUnbindTexture(tex2nd);

  runSim = false;

  return 0;
}

int main(void){
  pthread_create(&workerTh, NULL, workerThread, NULL);
  pthread_create(&controlTh, NULL, controlThread, NULL);

  pthread_join(workerTh, NULL);
  pthread_join(controlTh, NULL);

  return 0;
}
