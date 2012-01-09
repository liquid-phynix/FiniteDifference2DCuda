#if FTYPE == FLOAT
#define LOG(X) __logf(X)
#else
#define LOG(X) log(X)
#endif

#define KPM1 (KPART - F(1.0))
#define C2M1 (C_phi * C_phi - F(1.0))
#define ULOG (LOG(C_c / (F(1.0) + C_phi * KPM1 + KPART)))

#define U(PHI,C) (LOG(expu(PHI,C)))
#define LAPLACE_PHI ((E_phi + N_phi + S_phi + W_phi - F(4.0) * C_phi) / (DD * DD))

#if ANISOTROPY == YES
#define PHIDOT_ARG_LIST cellType N_phi,  cellType W_phi,  cellType E_phi,  cellType S_phi, cellType C_phi, cellType C_c,\
                        cellType NW_phi, cellType NE_phi, cellType SW_phi, cellType SE_phi
#define PHIDOT_UNTYPED_ARG_LIST N_phi, W_phi, E_phi, S_phi, C_phi, C_c,\
                                NW_phi, NE_phi, SW_phi, SE_phi
#else
#define PHIDOT_ARG_LIST cellType N_phi,  cellType W_phi,  cellType E_phi,  cellType S_phi, cellType C_phi, cellType C_c
#define PHIDOT_UNTYPED_ARG_LIST N_phi, W_phi, E_phi, S_phi, C_phi, C_c
#endif


#define CDOT_ARG_LIST cellType N_phi,  cellType W_phi,  cellType E_phi,  cellType S_phi, cellType C_phi,\
                      cellType N_c,    cellType W_c,    cellType E_c,    cellType S_c,   cellType C_c,\
                      cellType NW_phi, cellType NE_phi, cellType SW_phi, cellType SE_phi,\
                      cellType N_phi_new, cellType W_phi_new, cellType E_phi_new, cellType S_phi_new, cellType C_phi_new

#define CDOT_UNTYPED_ARG_LIST N_phi,   W_phi,   E_phi,   S_phi,  C_phi,\
                              N_c,     W_c,     E_c,     S_c,    C_c,\
                              NW_phi,  NE_phi,  SW_phi,  SE_phi,\
                              N_phi_new, W_phi_new, E_phi_new, S_phi_new, C_phi_new

__device__ cellType dfun(cellType phi, cellType c){
  return c * (F(1.0) - phi) / (1 + KPART - (1 - KPART) * phi);
}

__device__ cellType expu(cellType phi, cellType c){
  return 2 * c / (1 + KPART - (1 - KPART) * phi);
}

__device__ cellType phiDotPart2(cellType phi, cellType c){
  return (1 - phi * phi) * (phi - LAMBDA * (1 - phi * phi) * (expu(phi, c) - 1) / (1 - KPART));
}


__device__ cellType antiTrapping(cellType phi, cellType newPhi, cellType c, cellType grad0Phi2, cellType grad1Phi2){
  if(F(0.0) < (grad0Phi2 + grad1Phi2)){
    return (newPhi - phi) / DT * expu(phi, c) * DD * rsqrt(grad0Phi2 + grad1Phi2);
  }
  return F(0.0);
}

__device__ cellType surfaceAnis(cellType grad0Phi2, cellType grad1Phi2){
  cellType result = F(0.0);
  cellType gradPhi2 = grad0Phi2 + grad1Phi2;
  if(gradPhi2 > F(0.0)){
    cellType grad0PhiNormed = grad0Phi2 / gradPhi2;
    cellType grad1PhiNormed = grad1Phi2 / gradPhi2;
    result = grad0PhiNormed * grad0PhiNormed + grad1PhiNormed * grad1PhiNormed;
    cellType eta = F(1.0) + EPSILON * (F(4.0) * result - F(3.0));
    result = eta * eta + F(16.0) * eta * EPSILON * (grad0PhiNormed - result);
  }
  return result;
}

__device__ cellType invKineticAnis2(cellType gradXPhi2, cellType gradYPhi2){
  cellType result = F(0.0);
  cellType gradPhi2 = gradXPhi2 + gradYPhi2;
  if(F(0.0) < gradPhi2){
    result = (gradXPhi2 * gradXPhi2 + gradYPhi2 * gradYPhi2) / (gradPhi2 * gradPhi2);
    result = EPSILON * (result * F(4.0) - F(3.0));
  }
  result += F(1.0);
  return F(1.0) / (result * result);
}
#if ANISOTROPY == YES
__device__ cellType divType2SurfaceAnis(PHIDOT_ARG_LIST){
  cellType result = 0;
  cellType gradXPhi = 0;
  cellType gradYPhi = 0;

  gradXPhi = E_phi - C_phi;
  gradYPhi = F(0.25) * ((NE_phi - SE_phi) + (N_phi - S_phi));

  result += (gradXPhi * surfaceAnis(gradXPhi * gradXPhi,
				    gradYPhi * gradYPhi));
  gradXPhi = C_phi - W_phi;
  gradYPhi = F(0.25) * ((NW_phi - SW_phi) + (N_phi - S_phi));
  result -= (gradXPhi * surfaceAnis(gradXPhi * gradXPhi,
				    gradYPhi * gradYPhi));
  gradXPhi = F(0.25) * ((NE_phi - NW_phi) + (E_phi - W_phi));
  gradYPhi = N_phi - C_phi;
  result += (gradYPhi * surfaceAnis(gradYPhi * gradYPhi,
				    gradXPhi * gradXPhi));
  gradXPhi = F(0.25) * ((SE_phi - SW_phi) + (E_phi - W_phi));
  gradYPhi = C_phi - S_phi;
  result -= (gradYPhi * surfaceAnis(gradYPhi * gradYPhi,
				    gradXPhi * gradXPhi));
  return result / (DD * DD);
}
#endif


__device__ cellType divType2AntiTrapping(CDOT_ARG_LIST){
  cellType result = 0;
  cellType gradXPhi = 0;
  cellType gradYPhi = 0;

  gradXPhi = E_phi - C_phi;
  gradYPhi = F(0.25) * ((NE_phi - SE_phi) + (N_phi - S_phi));

  result += (gradXPhi * antiTrapping(F(0.5) * (E_phi + C_phi),
				     F(0.5) * (E_phi_new + C_phi_new),
				     F(0.5) * (E_c + C_c),
				     gradXPhi * gradXPhi,
				     gradYPhi * gradYPhi));
  gradXPhi = C_phi - W_phi;
  gradYPhi = F(0.25) * ((NW_phi - SW_phi) + (N_phi - S_phi));
  result -= (gradXPhi * antiTrapping(F(0.5) * (W_phi + C_phi),
				     F(0.5) * (W_phi_new + C_phi_new),
				     F(0.5) * (W_c + C_c),
				     gradXPhi * gradXPhi,
				     gradYPhi * gradYPhi));
  gradXPhi = F(0.25) * ((NE_phi - NW_phi) + (E_phi - W_phi));
  gradYPhi = N_phi - C_phi;
  result += (gradYPhi * antiTrapping(F(0.5) * (N_phi + C_phi),
				     F(0.5) * (N_phi_new + C_phi_new),
				     F(0.5) * (N_c + C_c),
				     gradYPhi * gradYPhi,
				     gradXPhi * gradXPhi));
  gradXPhi = F(0.25) * ((SE_phi - SW_phi) + (E_phi - W_phi));
  gradYPhi = C_phi - S_phi;
  result -= (gradYPhi * antiTrapping(F(0.5) * (S_phi + C_phi),
				     F(0.5) * (S_phi_new + C_phi_new),
				     F(0.5) * (S_c + C_c),
				     gradYPhi * gradYPhi,
				     gradXPhi * gradXPhi));
  return result / (DD * DD);
}

//phiDot()
__device__ cellType phiDot(PHIDOT_ARG_LIST){
  return (
#if ANISOTROPY == YES
	  divType2SurfaceAnis(PHIDOT_UNTYPED_ARG_LIST)
#else
	  LAPLACE_PHI
#endif
	  +
	  phiDotPart2(C_phi, C_c))
#if ANISOTROPY == YES
    * invKineticAnis2((E_phi - W_phi) * (E_phi - W_phi), (N_phi - S_phi) * (N_phi - S_phi));
#else
  ;
#endif
}

// __device__ cellType phiDot(PHIDOT_ARG_LIST){
//   return (divType2SurfaceAnis(PHIDOT_UNTYPED_ARG_LIST)
// 	  + 
// 	  phiDotPart2(C_phi, C_c)) * invKineticAnis2((E_phi - W_phi) * (E_phi - W_phi), (N_phi - S_phi) * (N_phi - S_phi));
// }


// cDot()
__device__ cellType cDot(CDOT_ARG_LIST){
  // D~ * Div(DFun(r_)*Grad(U(r_)))
  return DTILDE / (DD * DD) * (// 1,0
			       dfun(F(0.5) * (E_phi + C_phi), F(0.5) * (E_c + C_c)) * (U(E_phi, E_c) - U(C_phi, C_c)) -
			       // -1,0
			       dfun(F(0.5) * (W_phi + C_phi), F(0.5) * (W_c + C_c)) * (U(C_phi, C_c) - U(W_phi,W_c)) +
			       // 0,1
			       dfun(F(0.5) * (N_phi + C_phi), F(0.5) * (N_c + C_c)) * (U(N_phi, N_c) - U(C_phi, C_c)) -
			       // 0,-1
			       dfun(F(0.5) * (S_phi + C_phi), F(0.5) * (S_c + C_c)) * (U(C_phi, C_c) - U(S_phi, S_c))
			       )
    + 0.5 * M_SQRT1_2 * (F(1.0) - KPART) * divType2AntiTrapping(CDOT_UNTYPED_ARG_LIST)
    ;
}

// diffusion only
// __device__ cellType phiDotSimpleDiffusion(FULL_ARG_LIST){
//   return (- 4.0 * C_phi + W_phi + E_phi + N_phi + S_phi);
// }

// __device__ cellType cDotSimpleDiffusion(FULL_ARG_LIST){
//   return (- 4.0 * C_c + W_c + E_c + N_c + S_c);
// }