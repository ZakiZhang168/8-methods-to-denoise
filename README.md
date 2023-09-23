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
<br>
- (double noise) 1.mean 2.median 3.std -> unsatisfactory

<br>
- 4.Average -> better but unpractical

<br>
- 5.Bandpass -> not bad, more smooth

<br>
- 6.Wavelet -> somewhat better than 5

<br>
- 7.Threshold -> good

<br>
- 8. NN -> not good, Difficult to evaluate

<br>
- Combination -> give up md,mn,std, but nn is useful ! 

<br>



