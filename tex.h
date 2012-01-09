#define PER 1
#define NFX 2
#define FLOAT 1
#define DOUBLE 2
#define YES 1
#define NO 2

#define TEST 1

#if TEST == 0
#define GD_X 64
#define GD_Y 512
#define BD_X 64
#define BD_Y 8
#define WRITEEVERY 100
#define ITERS 1000
#define TIMEIT 1
#define BCOND NFX
#define FTYPE FLOAT
#define ANISOTROPY NO
#endif

#if TEST == 1
#define GD_X 100
#define GD_Y 100
#define BD_X 16
#define BD_Y 16
#define WRITEEVERY 100
#define ITERS 1000000
#define TIMEIT 0
#define BCOND NFX
#define FTYPE FLOAT
#define ANISOTROPY YES
#endif

#if TEST == 2
#define GD_X 16
#define GD_Y 16
#define BD_X 16
#define BD_Y 16
#define WRITEEVERY 10
#define ITERS 1000
#define TIMEIT 1
#define BCOND NFX
#define FTYPE DOUBLE
#define SAVEFLOAT
#define ANISOTROPY NO
#endif

#if TEST == 3
#define GD_X 256
#define GD_Y 1024
#define BD_X 16
#define BD_Y 4
#define WRITEEVERY 10
#define ITERS 10
#define TIMEIT 1
#define BCOND NFX
#define FTYPE DOUBLE
#define ANISOTROPY NO
#endif

// simulation constants
#define LAMBDAMUL	F(2.0)
#define DD		F(0.4)
#define DT		F(0.008)
#define OMEGA		F(0.55)
#define EPSILON		F(0.02)
#define KPART		F(0.18)
#define A1		F(0.8839)
#define A2		F(0.6267)

#define LAMBDA		(F(3.1913) * LAMBDAMUL)
#define DTILDE		(A2 * LAMBDA)
#define D0		(A1 / LAMBDA)

/* #define RSEED	F(3.0466894369065898) */
/* #define GAMMA	F(0.336) */
/* #define TMELTING     F(1811) */
/* #define TSIM		F(1780) */
/* #define LATENTHEAT	F(1.9464033850493653e9) */
/* #define DIFFUSION	F(1.e-9) */
/* #define D00		F(1.2298420278452437e-8) */
/* #define W0		F(8.880630984189447e-8) */
/* #define TAU0		F(0.00003154604881927703) */

char*	genFileName(char*, char*, int);
void	writeInitialConditionHomogene(float2*, int, int, float, float);
void	writeInitialConditionHomogene(int4*, int, int, double, double);
void	writeInitialConditionSingleNucleus(float2*, int, int);
void	writeInitialConditionSingleNucleus(int4*, int, int);
void	saveFields(float2*, int, int, int);
void	saveFields(int4*, int, int, int);

void*	workerThread(void*);
void*	controlThread(void*);
void	commandHandler(char*);

void cumulativeTimerStart(cudaEvent_t);
void cumulativeTimerStop(cudaEvent_t, cudaEvent_t, float*);

#define STRINGLENGTH 100

#define FETCH_TEX(BOOL,X,Y) ((BOOL)?(tex2D(tex1st, (X), (Y))):(tex2D(tex2nd, (X), (Y))))

#if FTYPE == FLOAT
#define F(X) X ## f
typedef float cellType;
typedef float2 dataType;
typedef texture<float2, cudaTextureType2D, cudaReadModeElementType> tex_t;
cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
#define PHI_FROM_TEXEL(TEXEL) (TEXEL.x)
#define C_FROM_TEXEL(TEXEL) (TEXEL.y)

#elif FTYPE == DOUBLE
#define F(X) X
typedef double cellType;
typedef int4 dataType;
typedef texture<int4, cudaTextureType2D, cudaReadModeElementType> tex_t;
cudaChannelFormatDesc channelDescriptor = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindSigned);
#define PHI_FROM_TEXEL(TEXEL) (__hiloint2double(TEXEL.y, TEXEL.x))
#define C_FROM_TEXEL(TEXEL) (__hiloint2double(TEXEL.w, TEXEL.z))
#endif

template<typename T>
struct simulationVariables_t{
  T* hostFields;

  T* deviceFields1st;
  T* deviceFields2nd;

  int fieldElements;
  size_t fieldsNBytes;

  simulationVariables_t(void){
    // elements per field
    fieldElements = GD_X * GD_Y * BD_X * BD_Y;
    fieldsNBytes = fieldElements * sizeof(T);
    // allocating host memory for fields
    T* tmp = (T*)malloc(fieldElements * sizeof(T));
    if(!tmp){
      printf("error allocating host memory fields\n");
      exit(1);
    }
    memset(tmp, 0, fieldElements * sizeof(T));
    hostFields = tmp;
    // allocation device memory for fields
    cudaMalloc(&tmp, 2 * fieldElements * sizeof(T));
    if(!tmp){
      printf("error allocating device memory for fields\n");
      exit(1);
    }
    cudaMemset(tmp, 0, 2 * fieldElements * sizeof(T));
    deviceFields1st = tmp;
    deviceFields2nd = tmp + fieldElements;
  }
  ~simulationVariables_t(void){
  //  void freeResources(void){
    // the destructor frees the allocated memory
    free(hostFields);
    cudaFree(deviceFields1st);
  }
  void upload(void){
    cudaMemcpy(deviceFields1st, hostFields, fieldElements * sizeof(T), cudaMemcpyHostToDevice);
  }
};
