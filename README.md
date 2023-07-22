# Generating 4bar drum midi file with Music VAE

## 개요
- **논문 정보**

이 논문은 google의 magenta project의 일부분으로 많은 양의 midi data를 학습해서 새로운 악곡을 generate하는 것이 목적이다. 

- **Music VAE란 무엇인가?**

기존에 사용하던 autoencoder의 형식의 generator를 개선해서 더 나은 형태의 모델을 만들고자 한다. 개선방안으로는 음악의 앞뒤 맥락을 익히게 하려고 encoder 에 bidirectional LSTM network를 채용하였고 decoder은 계층적 구조를 가지는 RNN을 사용하였다. 이 두가지를 이어붙여 만든 auto-encoder 형태를 여러가지 악기들로 이루어진 multi-stream 데이터들을 가공하여 training 시킨 모델이다.

- **목표**

논문의 목표는 이전의 모델보다 더 나은 sampling과 interpolation을 제공하는것이다.
이 논문에서 내가 얻고자 하는 목표는 4마디의 drum midi file을 생산하는 것이다.

## 논문 분석
- **1. Music VAE 모델 개요**

  - **1.1 인코더 구조**
  
  인코더의 구조는 이전까지의 연구와는 크게 달라진점이 없다고 한다. input sequence를 2겹의 bidirectional LSTM을 사용하였고 그 결과물을 2겹을 Fully-connected layer로 latent distribution parameter, 즉 잠재공간의 확률분포를 만들어 낸다. encoder 자체로는 이전 연구에 비해서 특별한것이 없지만 음악이라는 데이터적 측면에서 과거와 미래, 양쪽 방향 모두의 context를 학습하기 위한 bidirectional LSTM을 사용한것은 주목할만하다.

  - **1.2 디코더 구조**
  
  논문의 가장 주요한 부분인 decoder 구조에서는 기존의 VAE에서 사용하던 RNN layer로 이루어진 decoder에서 벗어나 hierarcical decoder를 구성했다. 이는 연구자들이 RNN layer로는 잠재상태의 소실 현상으로 인한것으로 판단하고 이를 보완하기 위함이다. 방법론적으로는 잠재공간에서 나오는 벡터들을 2겹의 unidirectional LSTM을 통과시켜 conductor이라는 중간 단계를 만들어, 1겹의 fully-connected layer을 통과시킨뒤 기존의 RNN decoder 형태를 뒷단에 붙인 형태이다. 이는 지속적으로 RNN으로 한번에 decode하는 방식이 아니라 시퀀스 데이터를 쪼개서 서브시퀀스를 생성할때마다 잠재 공간에서 나온 conductor를 가져와서 RNN에 넣어주었다. conductor 의 자가회귀 방식도 실험해보았지만 성능이 그렇게 좋지 않았고 잠재공간에 있는 벡터를 보존해 최종 RNN 디코더에 입력하는것이 더 좋았다고 한다.

  - **1.3 다중 흐름 모델링**

  이 모델을 학습하기 위한 데이터로 연구자들은 trio model 즉 3개의 data 종류를 한꺼번에 학습시켰다고 한다. (drum, bass, melody) 이 3개의 악기들은 서로 orthognal 한 관계를 이루고 있다고 가정하며 각각 한개의 차원을 이루고 있다고 생각하였다. 또한 결과 비교를 위해 baseline 모델을 기존의 non-hierarchical decoder를 사용하였다.

- **2. 실험 결과 및 분석**

  - **2.1 평가 방법**
  음악 생성이라는 영역 답게 정량적인 평가가 거의 불가능할 정도에 가깝기 때문에 정성적인 평가를 할수 밖에 없었다고 한다. 총 192개의 평가가 수집되었고 약 30초쯤 되는 생성된 음악을 들었다고 한다. 

  - **2.2 결과 해석**

  결과적으로 모든 영역에서 flat한 decoder를 사용한것보다 논문에서 사용한 hierachical decoder를 사용한 것이 더 높게 rating 되었다. 또 특이하게도 drum 에서는 실제 음악보다는 generate 한 음악이 조금더 rating이 높은 특징을 보였다. 또 음악을 test하기 위한 몇가지 테스트에서 (Kruskal-Wallis H test, Wilcoxon signed rank test) 상당히 높은 성능을 보였다고 한다.

- **3. 응용 및 한계**

  - **3.1 한계와 개선 방향**

  음악을 생성하기는 하지만 16마디 정도 되는 아주 긴 시간의 음악을 생성하는데는 실패하였다고 한다. 연구자들이 차후에는 다른 형태의 sequential data를 자신들의 모델에 테스트하는것이 그 다음 목표라고 한다.

- **4. 결론**

  - **4.1 논문 요약**

  음악을 생성하기 위해서 예전에 사용하던 모델을 개선시킨 논문이다. 잠재공간에 있는 벡터들을 더 효과적으로 활용하기 위해서 많은 노력들을 들인듯 하다. 비록 long-term의 music context를 생성하는데는 실패했지만 그래도 짦은 크기의 음악을 효과적으로 생성할수 있었다는게 이 논문의 의의라고 생각한다.

## 설치 및 실행
- **환경 설정**

`pip install magenta`(bash)

conda로 가상환경을 생성한뒤에 magenta 패키지를 설정하는것으로 끝났다.
현재 나의 로컬 환경에는 Nvidia geforce 2060 을 활용할수 있는 GPU 설정이 되어 있었기 때문에 GPU를 사용해서 training을 시킬수 있었다.

- **데이터 다운로드**
코드를 살펴보던중 정확하게 목적에 부합하는 config map을 찾았다.
"groovae_4bar" config는 groovae midi dataset을 tfds에서 가져와서 학습하도록 이미 세팅되어 있었기에 그대로 config map을 사용하였다.
명령어는 아래쪽에 기록하였다.

- **모델 학습 방법**

`python music_vae_train.py --config='groovae_4bar' --run_dir='./checkpoint' --mode='train'`(bash)

위의 명령어에 이미 dataset download, grooveconverter(전처리), training까지 한번에 가능하도록 설정되어 있었기때문에 root 디렉토리에서 위의 명령어를 쳐서 training 하였다.
지속해서 training을 하다가 더이상 loss가 줄어드는것 같지 않을때 자체적으로 학습을 멈춰서 가장 마지막에 있는 checkpoint를 가져와서 model generate 하였다.

- **음악 생성 방법**
`python music_vae_generate.py --config='groovae_4bar' --checkpoint_file='./newest_checkpoint/model.ckpt-53482' --output_dir='./output' --mode='sample' --num_outputs=5`

위의 명령어로 model을 generate 하였다. 학습한 그대로 config를 가져왔으며 가장 최신의 checkpoint 파일을 가져다가 newest_checkpoint 디렉토리에 넣은후 실행하였다.
총 5개의 sample을 generate하도록 하였으며 그 결과는 ouput 디렉토리에 midi file로 저장되었다.


