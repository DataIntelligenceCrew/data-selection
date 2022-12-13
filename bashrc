# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# export path
# export PATH=$HOME/opt/cmake3.18.2/bin:$PATH
# export PATH=$HOME/.local/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/opt/usr/local/lib
export LIBRARY_PATH=$LIBRARY_PATH:$HOME/opt/usr/local/lib
export CPLUS_INCLUDE_PATH=$CPLUS_INCLUDE_PATH:$HOME/opt/usr/local/include
export C_INCLUDE_PATH=$C_INCLUDE_PATH:$HOME/opt/usr/local/include
export CUDA_HOME=/usr/local/cuda
export CPATH=/usr/local/cuda/targets/x86_64-linux/include:$CPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/targets/x86_64-linux/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/localdisk2/pmundra/miniconda3/envs/data-selection/lib
export PATH=/usr/local/cuda/bin:$PATH
# uncomment the LD_PRELOAD for faiss if using python
export LD_PRELOAD=/opt/intel/mkl/lib/intel64/libmkl_def.so:/opt/intel/mkl/lib/intel64/libmkl_avx2.so:/opt/intel/mkl/lib/intel64/libmkl_core.so:/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so:/opt/intel/mkl/lib/intel64/libmkl_intel_thread.so:/opt/intel/lib/intel64_lin/libiomp5.so
# export CMAKE_PREFIX_PATH="/opt/intel/compilers_and_libraries_2018.0.128/linux/mkl:$PATH"
# User specific aliases and functions
