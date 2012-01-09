tex: tex.cu tex.h vtk_writer_lib.h eqns.cu makefile
	nvcc -arch=sm_13 -lpthread -lreadline -o tex tex.cu
keep: tex.cu tex.h vtk_writer_lib.h eqns.cu makefile
	nvcc --keep --keep-dir keeps -O3 -arch=sm_13 -lpthread -lreadline -o tex tex.cu
v: tex.cu tex.h vtk_writer_lib.h
	nvcc --ptxas-options -v -arch=sm_13 -o tex tex.cu
#-use_fast_math 
