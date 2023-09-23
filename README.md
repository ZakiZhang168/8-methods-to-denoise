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
- why not pair them two or three together ?!   <br>

## assessment
- 1.mean 2.median 3.std	->Useful but not effective
- ![3](https://github.com/ZakiZhang168/8-methods-to-denoise/assets/130261283/1604f251-af8f-4c70-a6ab-8fd16e3bcd95)

<br>
- (double noise) 1.mean 2.median 3.std -> unsatisfactory
![4](https://github.com/ZakiZhang168/8-methods-to-denoise/assets/130261283/fcd247ac-2ac5-4b5a-a08a-8bbc12904a4b)

<br>
- 4.Average -> better but unpractical
![5](https://github.com/ZakiZhang168/8-methods-to-denoise/assets/130261283/e10217e5-8e5f-47c2-b0a5-f43bed22307c)

<br>
- 5.Bandpass -> not bad, more smooth
![6](https://github.com/ZakiZhang168/8-methods-to-denoise/assets/130261283/82f973e5-283d-4109-b520-1b8417c77b7d)

<br>
- 6.Wavelet -> somewhat better than 5
![7](https://github.com/ZakiZhang168/8-methods-to-denoise/assets/130261283/64dcfc30-c5e4-4e4f-b2ff-647d31cda029)

<br>
- 7.Threshold -> good
![8](https://github.com/ZakiZhang168/8-methods-to-denoise/assets/130261283/fec776c9-e47a-4874-8ff8-79a99ba6ef42)

<br>
- 8. NN -> not good, Difficult to evaluate
![9](https://github.com/ZakiZhang168/8-methods-to-denoise/assets/130261283/9a2651da-ffdd-4c85-8121-d4c5bc8c31af)

<br>
- Combination -> give up md,mn,std, but nn is useful ! 
![10](https://github.com/ZakiZhang168/8-methods-to-denoise/assets/130261283/65c11d36-0dcd-4e01-9530-06b761616787)

<br>



