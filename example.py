import build.pycudeft as pycudeft
import numpy as np  

size = 100
logk = np.zeros(size)
logp = np.zeros(size)
step_size = (np.log(1e4) - np.log(1e-5))/size

for i in range(size):
	logk[i] = np.log(1e-5) + i*step_size
	logp[i] = -2.2*logk[i]

logk_eval = np.log(np.array([0.01, 0.1, 1.0, 10.0]))

result_mm1 = pycudeft.p_mm(logk_eval, len(logk_eval), logp, logk, size, 50.0, "1")
result_mm1 = pycudeft.p_mm(logk_eval, len(logk_eval), logp, logk, size, 50.0, "1")
result_mmq = pycudeft.p_mm(logk_eval, len(logk_eval), logp, logk, size, 50.0, "quad")
result_mm2 = pycudeft.p_mm(logk_eval, len(logk_eval), logp, logk, size, 50.0, "2")

print('RESULT = ')
for i in range(len(logk_eval)):
	print("{:2e}, {:3e}, {:3e}, {:3e}".format(
		np.exp(logk_eval[i]), 
		result_mm1[i], 
		result_mmq[i],
		result_mm2[i]
	)
)
#np.savetxt('test.txt',result_mm)