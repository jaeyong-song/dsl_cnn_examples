# CNN 실습 사전 준비

## 0. 기본 라이브러리 설치

최신 파이썬, pytorch, 자신이 좋아하는 IDE에서 실습 코드 돌아가는지 확인

(또는 CNN을 자주 하실 일이 없거나 바쁘신 분은 실습과 과제를 모두 캐글 데이터셋으로 하기 때문에 캐글에 커널 생성해서 하셔도 됩니다.)

## 1. NVIDIA 그래픽 카드 유무에 따른 준비

nvidia 그래픽 카드가 있으시다면 training 속도가 차원이 다르게 빨라집니다. 

https://mickael-k.tistory.com/18

https://m.blog.naver.com/PostView.nhn?blogId=hschoi237&logNo=221655297017&proxyReferer=https:%2F%2Fwww.google.com%2F

이 예시를 따라서 설치하시면 됩니다.

## 2. 확인

`prepare_test.py`파일의 코드가 정상적으로 실행되는지 확인하시면 됩니다. 또한 nvidia 그래픽카드에 맞추어 환경설정을 하신 분들은 `cuda:n`또는 `cuda`가 나오면 정상이고 다른 분들은 `cpu`가 나오면 정상입니다.