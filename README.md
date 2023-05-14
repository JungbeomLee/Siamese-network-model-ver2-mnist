# Siamese-network-model-ver2-mnist
> Siamenetwork 구조를 적용한 모델 학습 프로젝트이다.

> mnist dataset을 활용하여 한쌍의 숫자 이미지에 대한 유사도를 판별한다.


# 사전 학습 모델 사용 방법
```
Only_Siamese_network_ver2_testing_code.ipynb 
```
> 파일을 사용하여 사전 학습된 모델을 사용하는 예시 코드를 볼 수 있다.

```
create_Siamese_network_ver2.py
```
> 위 파일을 사용하여 모델을 생성한 뒤, 모델의 가중치 파일을 적용하여 사용할 수 있다.


# 데이터셋 mnist
![Alt text](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

> 손으로 쓴 숫자들로 구성된 데이터셋이다.

> 0~9까지(10개의 클래스)의 숫자를 손글씨로 쓴 이미지 샘플로 구성되어 있다.

> Training dataset과 Test dataset이 포함되어 있다.
> Training dataset에는 60,000개의 이미지가 있다.
> Test dataset에는 10,000개의 이미지가 있다.

> 각 이미지는 28x28 픽셀의 흑백 이미지로 구성되어 있다.


# 모델
### 1. Siamese network 논문
> https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf


### 2. What is Siamese network?
> 두 개의 입력 이미지를 비교하여 유사도를 측정하는 모델 구조이다.

> 새로운 이미지에 대한 유사성을 예측할 수 있다.

> 두 개의 동일한 네트워크 공유 가중치로 구성되며 각 네트워크는 입력 이미지를 임베딩 하는 역할을 수행한다.

> 각각의 입력 이미지는 네트워크를 통과한 후 이미지의 핵심 특징을 나타내는 임베딩 벡터로 변환된다.

> 두 임베딩 벡터의 유클리디란 거리를 계산 후, 거리의 값이 작을수록 두 이미지가 유사하다고 판단한다.

> Siamese Network는 손실 함수를 통해 학습된다.
>> Training label에는 이미지 쌍의 유사성(유사할 경우1, 다른 경우 0)이 포함된다. 모델은 입력 이미지 쌍을 임베딩 하여 거리를 계산하고, 이를 기반으로 유사성을 예측한다. 손실 함수는 예측된 label과 실제 label의 차이를 최소화하기 위해 모델의 가중치를 조정한다.


