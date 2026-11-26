## The GPU statevector simulator (needs nvcc: build it on a GPU machine such as SOL; it compiles anywhere nvcc is)
##   make                 build ./qsim_bench and libnova_runtime_gpu.a for ARCH (sm_80: A100; sm_90: H100)
##   make run             build and run the benchmark, sizes 16 to 28 (options: the top of qsim_bench.cu)
##   make check           build and run a quick check: small sizes only, every amplitude compared with the CPU's
##   make dropin-check    a circuit through the runtime's C functions, with the CPU runtime and with the GPU one
## libnova_runtime_gpu.a is the drop-in runtime: link a compiled Nova program with it (nova-compile --gpu) and its
## circuits run on the GPU.

NVCC   ?= nvcc
ARCH   ?= sm_80
CC     ?= gcc
NVFLAGS := -O3 -std=c++17 -arch=$(ARCH) -lineinfo
RT     := ..

.PHONY: all run check dropin-check clean

all: qsim_bench libnova_runtime_gpu.a

# The CPU simulator, with OpenMP, for the benchmark to compare against
nova_quantum_cpu.o: $(RT)/nova_quantum.c
	$(CC) -O3 -fopenmp -c $< -o $@

qsim_bench: qsim_bench.cu statevector.cu statevector.cuh nova_quantum_cpu.o
	$(NVCC) $(NVFLAGS) -Xcompiler -fopenmp qsim_bench.cu statevector.cu nova_quantum_cpu.o -o $@ -lgomp

# The runtime a compiled program links with, its circuits on the GPU: the GPU's statevector functions, the shared
# part of nova_quantum.c (printing, random numbers, hardware instructions by site), and the tensor runtime
statevector.o: statevector.cu statevector.cuh
	$(NVCC) $(NVFLAGS) -Xcompiler -fPIC -c $< -o $@
nova_quantum_gpu.o: nova_quantum_gpu.cu statevector.cuh
	$(NVCC) $(NVFLAGS) -Xcompiler -fPIC -c $< -o $@
nova_quantum_shared.o: $(RT)/nova_quantum.c
	$(CC) -O2 -fPIC -DNOVA_QUANTUM_GPU -c $< -o $@
nova_runtime.o: $(RT)/nova_runtime.c
	$(CC) -O2 -fPIC -c $< -o $@
libnova_runtime_gpu.a: statevector.o nova_quantum_gpu.o nova_quantum_shared.o nova_runtime.o
	ar rcs $@ $^

# The drop-in runtime: the same calls a compiled circuit makes, with the CPU runtime and with the GPU runtime; the two
# must print the same
dropin_check_cpu: dropin_check.c $(RT)/nova_quantum.c
	$(CC) -O2 dropin_check.c $(RT)/nova_quantum.c -lm -o $@
dropin_check_gpu: dropin_check.c libnova_runtime_gpu.a
	$(NVCC) -arch=$(ARCH) dropin_check.c libnova_runtime_gpu.a -lm -o $@
dropin-check: dropin_check_cpu dropin_check_gpu
	./dropin_check_cpu > dropin_cpu.txt
	./dropin_check_gpu > dropin_gpu.txt
	diff dropin_cpu.txt dropin_gpu.txt && echo "drop-in runtime: the CPU and the GPU print the same"

run: qsim_bench
	./qsim_bench

check: qsim_bench
	./qsim_bench --min 10 --max 16 --cpu-max 16 --depth 4

clean:
	rm -f qsim_bench dropin_check_cpu dropin_check_gpu dropin_*.txt *.o *.a
