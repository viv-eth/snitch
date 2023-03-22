echo "Setting up environment for Occamy simulations."
export CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER=/usr/pack/gcc-9.2.0-af/linux-x64/bin/gcc
export LLVM_SYS_120_PREFIX=/usr/pack/llvm-12.0.1-af
export GCC_DIR=/usr/pack/gcc-9.2.0-af
export GCC_DIR2=$GCC_DIR/linux-x64
export CC=$GCC_DIR2/bin/gcc
export CXX=$GCC_DIR2/bin/g++
export C_INCLUDE_PATH=$GCC_DIR/include
export CPLUS_INCLUDE_PATH=$GCC_DIR/include
export LD_LIBRARY_PATH=$GCC_DIR2/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=$GCC_DIR2/lib64:$LIBRARY_PATH
export PATH=$GCC_DIR/linux-x64/bin:$PATH
export RISCV=/usr/pack/riscv-1.0-kgf/pulp-llvm-0.12.0
export PATH=$RISCV/bin:$PATH
export PATH="$HOME/local/bin:$PATH"

conda activate msc22f11
echo "Occamy simulation framework set up."