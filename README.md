# 8-methods-to-denoise
### 8 methods to denoise one dim signal with randn-noise.

## restore.py
![image](https://github.com/ZakiZhang168/8-methods-to-denoise/assets/130261283/67688336-6809-4445-be30-69dcf4f1e742)
- NS: noisy signal
- S: original siganl
- mean filter: ws = window size
- median filter:
- average filter: ns = number of noisy signal(different)
- bandpass filter: l = low cut-off frequency, h = high ...
- threshold filter: r = ratio(max abs(fft) / min ...)
- wavelet filter: a = threshold
- std filter: 
- NN: neural network

## restore2.py
![image](https://github.com/ZakiZhang168/8-methods-to-denoise/assets/130261283/a93aa7a1-40e4-40d9-8197-bf53a7c069bd)
- why not pair them two or three together ?!

### assessment
- 1.mean 2.median 3.std	->Useful but not effective
![image](https://github.com/ZakiZhang168/8-methods-to-denoise/assets/130261283/c8c9a0a0-2258-4d86-beb1-65a985a160d4)
- (double noise) 1.mean 2.median 3.std -> unsatisfactory
![image](https://github.com/ZakiZhang168/8-methods-to-denoise/assets/130261283/389deb4a-adb6-43d4-8d5e-03c2311187bf)
- 4.Average -> better but unpractical
![image](https://github.com/ZakiZhang168/8-methods-to-denoise/assets/130261283/bb064b93-71ed-488a-ba13-f5eb4630d172)
- 5.Bandpass -> not bad, more smooth
![image](https://github.com/ZakiZhang168/8-methods-to-denoise/assets/130261283/ac7b1057-9c69-47ac-841c-2c01e9cd2a80)
- 6.Wavelet -> somewhat better than 5
![image](https://github.com/ZakiZhang168/8-methods-to-denoise/assets/130261283/3cab451e-fc1e-4976-b5b4-00438cdc6baf)
- 7.Threshold -> good
![image](https://github.com/ZakiZhang168/8-methods-to-denoise/assets/130261283/1ed80713-13c5-4571-9504-15a01ea2ee8f)
- 8. NN -> not good, Difficult to evaluate
![image](https://github.com/ZakiZhang168/8-methods-to-denoise/assets/130261283/2c0ea358-a310-4c23-a03d-a4b078a474e5)
- Combination -> give up md,mn,std, but nn is useful ! 
![image](https://github.com/ZakiZhang168/8-methods-to-denoise/assets/130261283/2ca840a9-95b3-43d3-bbfc-d9bde4c8969e)



