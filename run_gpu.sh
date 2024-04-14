nvcc -O3 run_gpu.cu -o run_gpu
./run_gpu
g++ -O3 -o dat_to_png dat_to_png.cpp `pkg-config --cflags --libs opencv4`
./dat_to_png
rm out.dat