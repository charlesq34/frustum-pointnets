# set -e
# if [ 'tf_sampling_g.cu.o' -ot 'tf_sampling_g.cu' ] ; then
echo 'nvcc'
/usr/local/cuda-8.0/bin/nvcc tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
#fi
#if [ 'tf_sampling_so.so' -ot 'tf_sampling.cpp' ] || [ 'tf_sampling_so.so' -ot 'tf_sampling_g.cu.o' ] ; then
echo 'g++'
g++ -std=c++11 tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-8.0/include -lcudart -L /usr/local/cuda-8.0/lib64/ -O2
#fi
