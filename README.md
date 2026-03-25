# cudeft
CUDA implementation of the EFT of large scale structure.

This requires conda, an Nvidia GPU and a slightly modified version of the m-CUBES integration package, which will be made available soon.

## Download and Install

To download and install cudeft, run the following commands:

```
git clone git@github.com:evansaraivanov/cudeft.git
cd cudeft
conda create -n (your env name) --file cudeft_env.yaml
cd build
cmake ..
make
cd ../
```

The integrals can be evaluated from python, allowing one to easily make code to read or create linear power spectra however they like and pass it to the code. See example.py for an example with $P_L(k) = k^{-n}$ and enjoy the speed!
