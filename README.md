# 🚀 LiDAR 기반 딥러닝 실내 장애물 탐지 및 스무딩 매핑
(LiDAR-based Obstacle Detection & Smoothing Mapping)

## 1. 프로젝트 개요
자율주행 로봇이 실내를 이동할 때, 라이다(LiDAR) 센서의 2D 격자 지도(Grid Map)는 노이즈가 많고 시각적으로 직관적이지 않습니다. 
본 프로젝트는 라이다 원본 지도와 사용자가 지정한 실제 장애물 경계(Ground Truth)를 CNN 모델에 학습시켜, **어떤 실내 지도가 주어지더라도 진짜 장애물의 윤곽을 스스로 예측하고, 사람이 보기 편한 매끄러운 곡선(Smooth Boundary)으로 도면화**하는 딥러닝 파이프라인입니다.

## 2. 주요 기능
- **Custom Ground Truth 자동 추출:** 사용자가 이미지 편집기(예: 그림판)로 붉은색 테두리만 그리면, 내부 면적을 자동으로 채워 AI 학습용 장애물 라벨(Tensor)로 변환합니다.
- **2-Class CNN 분할(Segmentation):** 0(이동 가능 구역), 1(장애물)로 공간을 이진 분류하여 실질적인 회피 대상만 추적합니다.
- **형태학적 스무딩(Morphological Smoothing):** AI가 예측한 계단식 픽셀 결과물에 Gaussian Blur와 `cv2.approxPolyDP` 알고리즘을 적용하여 부드럽고 매끄러운 윤곽선을 생성합니다.
- **실무 환경(Ubuntu/ROS) 최적화:** 시각화 모듈을 경량화하고 `argparse`를 도입하여 리눅스 터미널 환경에서 즉각적인 Inference가 가능하도록 최적화했습니다.

## 3. 기술 스택 (Tech Stack)
- **Language:** Python
- **Deep Learning:** PyTorch
- **Computer Vision:** OpenCV, Numpy

## 4. 파일 구성
- `my_room_map.png`: 원본 LiDAR 2D 지도 (Input)
- `my_room_map_obs.png`: 사용자 지정 장애물 윤곽선 데이터 (Ground Truth)
- `run_obstacle_inference.py`: 우분투 터미널용 최종 학습 및 추론 통합 스크립트

## 5. 실행 방법 (Usage)
```bash
# 기본 실행 (에폭 13,000회 최적화 적용)
python3 run_obstacle_inference.py --epochs 13000

# 파일 경로 직접 지정 실행
python3 run_obstacle_inference.py --map my_room_map.png --obs my_room_map_obs.png --output result.png
