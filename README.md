# Object-Detection-Autonomous-Driving

[🔗 Detail](https://docs.ros.org/en/humble/Installation.html).
[![Video Link](https://i.postimg.cc/43dWpm2k/Untitled.png)](https://www.youtube.com/watch?v=OlB29rsbV0o&ab_channel=%EA%B9%80%EB%8C%80%EC%99%84)

실제 차의 10분의1 크기의 모형 자동차를 통해 실제도로와 유사한 모형도로 상황을 인식하고 그에 맞게 제어해야 한다. 미션은 아래와 같다.

1. 차선 인식 후 차선을 따라 주행
2. 정지 및 횡단보도 표지판 인식시 정지선에서 5초 정지 및 주행
3. 횡단보도 신호에 맞게 주행
4. 교차로에서 표지판 신호에 따라 주행
5. 정적 장애물 회피하여 주행
6. 돌발 장애물 인식시 정지 후 주행
7. 도로 현황을 bird-eye-view로 시각화

여기 깃허브 README를 위한 프로젝트 설명이 있습니다:

---

## 프로젝트 개요

이 프로젝트는 실제 도로 환경과 유사한 모의 도로 상황에서 1:10 크기의 모형 자동차가 차선 및 교통 표지판을 인식하고 이를 통해 주행하는 것을 목표로 합니다. 특히, 차선 인식의 안정성을 강화하기 위해 가려지거나 끊어진 차선을 연결하고 보정하는 칼만 필터를 사용하였습니다.

## 주요 기능

- **칼만 필터를 이용한 차선 모델링**: 끊어지거나 가려진 차선을 보정하여 지속적인 차선 추적을 보장합니다.

- **교통 표지판 우선순위 관리**: maxHeap 자료 구조를 사용하여 교통 표지판의 우선순위를 효율적으로 관리하고 빠르게 인식합니다.

- **최적의 차선 탐지**: 직선의 위치와 각도를 기반으로 클러스터링을 수행하여 모형 자동차에 대한 최적의 차선 경로를 식별합니다.

- **객체 탐지 및 예외 처리**: LiDAR 데이터 처리를 통한 클러스터링과 추적을 통합하며, 자동차 주변의 측면 객체에 대한 예외 처리도 구현합니다.

- **YOLOv3와 TensorRT를 이용한 딥러닝**: 객체 인식을 위해 YOLOv3 알고리즘을 채택하고, 모형 자동차의 Jetson 시리즈 하드웨어에 최적화하기 위해 TensorRT를 사용합니다.

- **Bird-Eye-View를 통한 시각화**: 호모그래피 변환을 적용하여 모형 자동차 주변 객체와의 거리와 위치를 시각화합니다.

## 사용 기술

- **칼만 필터**: 차선 모델링 및 보정을 위해 사용됩니다.
- **MaxHeap 자료 구조**: 교통 표지판 우선순위 관리에 사용됩니다.
- **클러스터링 알고리즘**: 최적의 차선 경로를 결정하기 위해 사용됩니다.
- **LiDAR 기술**: 객체 클러스터링 및 추적을 위해 사용됩니다.
- **YOLOv3와 TensorRT**: Jetson 시리즈 하드웨어에서의 딥러닝 최적화를 위해 사용됩니다.
- **호모그래피 변환**: Bird-Eye-View 시각화를 위해 사용됩니다.

이 저장소는 이러한 시스템의 구현을 상세히 설명하며, 테스트 환경 내에서 모형 자동차의 작동에 대한 문서를 제공합니다.

---
