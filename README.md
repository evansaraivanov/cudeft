# cudeft
CUDA implementation of the EFT of large scale structure.

This requires conda and an Nvidia GPU.

I have provided in the externals folder a slightly modified version of m-CUBES (see arXiv:2202.01753) which I have independently modified to allow for parallel integrations over $k$-modes should your GPU have enough SM cores to launch them.

## Download and Install

To download and install cudeft, run the following commands:
```
git clone git@github.com:evansaraivanov/cudeft.git
cd cudeft
conda create -n (your env name) --file cudeft_env.yaml
conda activate (your env name)
mkdir build
cd build
cmake ..
make
cd ../
```
cmake should get your CUDA archecture correct.

The integrals can be evaluated from python, allowing one to easily make code to read or create linear power spectra however they like and pass it to the code. See example.py for an example with $P_L(k) = k^{-n}$ and enjoy the speed! You can run the example with
```
python example.py
```

