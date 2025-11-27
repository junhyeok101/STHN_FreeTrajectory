# ============================================================================
# 데이터 로딩 및 전처리 모듈 (NVIDIA FlowNet2-PyTorch 기반)
# GitHub: https://github.com/NVIDIA/flownet2-pytorch
# 위성 이미지(지형도)와 항공 이미지(쿼리) 간의 호모그래피 변환을 통해
# 위치 기반 이미지 매칭을 위한 광학 흐름 라벨을 생성하는 데이터셋
# ============================================================================

import numpy as np  # 수치 계산 라이브러리
import torch  # PyTorch 텐서 및 기본 연산
import torch.utils.data as data  # PyTorch 데이터셋 및 DataLoader
import kornia.geometry.transform as tgm  # 기하학적 변환 (호모그래피, 워핑 등)

import random  # 무작위 샘플링
from glob import glob  # 파일 경로 패턴 매칭
import os  # 운영체제 관련 작업
import cv2  # OpenCV (perspectiveTransform 등)
from os.path import join  # 파일 경로 결합
import h5py  # HDF5 파일 포맷 (데이터 저장/로드)
from sklearn.neighbors import NearestNeighbors  # KNN 알고리즘 (근처 샘플 찾기)
import logging  # 로깅
from PIL import Image, ImageFile  # 이미지 로드 및 처리
import torchvision.transforms as transforms  # 이미지 전처리 (정규화, 리사이즈 등)
import torchvision.transforms.functional as F  # 함수형 이미지 처리
from tqdm import tqdm  # 진행률 표시 (현재 코드에서는 사용 안 함)
import math  # 수학 함수 (현재 코드에서는 사용 안 함)

# ============================================================================
# 이미지 로드 설정
# ============================================================================
Image.MAX_IMAGE_PIXELS = None  # 매우 큰 이미지 로드 시 PIL 경고 무시
marginal = 0  # 미사용 변수

# ============================================================================
# 이미지 정규화 상수 (ImageNet 사전학습 모델 기준)
# ============================================================================
# 각 채널(R, G, B)의 평균값 - 이미지에서 뺌
imagenet_mean = [0.485, 0.456, 0.406]
# 각 채널의 표준편차 - 이미지를 이 값으로 나눔
imagenet_std = [0.229, 0.224, 0.225]
# Tartan Bridge 데이터셋의 검증 영역 좌표 [left, right, top, bottom]
# 특정 지역의 이미지만 검증에 사용
TB_val_region = [2650, 5650, 5100, 9500]

# ============================================================================
# 이미지 변환 조합 (역정규화)
# ============================================================================
# 모델 출력을 시각화하기 위해 정규화를 역으로 수행
inv_base_transforms = transforms.Compose(
    [ 
        # 이론적으로는: Normalize(mean=[m/s for m,s in ...], std=[1/s for s in ...])
        # 하지만 현재는 단순히 텐서를 이미지로 변환만 함
        transforms.ToPILImage(),  # 텐서(0~1) → PIL Image
    ]
)

# ============================================================================
# 이미지 변환 조합 (기본)
# ============================================================================
# 모델 입력 전 이미지 정규화
base_transforms = transforms.Compose(
    [ 
        # transforms.ToTensor() 제거됨 (나중에 수동으로 적용)
        # ImageNet 평균/표준편차로 정규화: (I - mean) / std
        transforms.Normalize(mean=imagenet_mean,
                             std=imagenet_std),
    ]
)


# ============================================================================
# 1. homo_dataset 클래스 - 호모그래피 기반 데이터셋 기본 클래스
# ============================================================================
class homo_dataset(data.Dataset):
    """
    쿼리 이미지와 데이터베이스 이미지 간의 호모그래피 변환을 통해
    광학 흐름(Optical Flow) 라벨을 생성하는 데이터셋.
    
    동작 방식:
    1. 쿼리 이미지(지상): 그레이스케일 → 3채널 텐서
    2. DB 이미지(위성): 대비 조정 후 리사이즈
    3. 호모그래피 계산: UTM 좌표 차이로부터
    4. 광학 흐름 계산: 모든 픽셀을 호모그래피로 변환한 후 이동량 계산
    5. 데이터 증강: 회전, 크기조정, 원근감 왜곡
    """
    
    def __init__(self, args, augment=False):
        """
        호모그래피 데이터셋 초기화
        
        Parameters:
            args: 설정 파라미터 (Namespace)
                - resize_width: 이미지 리사이즈 크기 (기본 256)
                - crop_width: 크롭 크기 (기본 224)
                - database_size: 위성 이미지 원본 크기 (512~2560)
                - perspective_max: 원근감 왜곡 최대값 (픽셀)
                - rotate_max: 회전 각도 최대값 (라디안)
                - resize_max: 크기조정 비율 최대값
                - eval_model: 평가 모드 여부
                - multi_aug_eval: 평가 시 다중 증강 여부
            augment: 데이터 증강 활성화 여부
        """
        self.args = args
        self.is_test = False  # 현재 사용 안 함
        self.image_list_img1 = []  # 쿼리 이미지 목록 (현재 사용 안 함)
        self.image_list_img2 = []  # DB 이미지 목록 (현재 사용 안 함)
        self.dataset = []  # 데이터셋 (현재 사용 안 함)
        self.augment = augment  # 데이터 증강 사용 여부
        
        # ===== 데이터 증강 설정 =====
        if self.augment:  # 증강이 활성화된 경우만 설정
            # 평가 모드(eval_model이 지정된 경우): 결정론적 난수 생성기 사용
            if self.args.eval_model is not None:
                self.rng = None  # 워커 초기화 시 seed 설정됨
            
            # 사용 가능한 증강 타입 리스트 ("no"는 증강 없음을 의미)
            self.augment_type = ["no"]
            
            # 설정된 값이 0보다 크면 해당 증강 타입 추가
            if self.args.perspective_max > 0:
                self.augment_type.append("perspective")  # 원근 왜곡: 코너 포인트에 무작위 오프셋
            if self.args.rotate_max > 0:
                self.augment_type.append("rotate")  # 회전: 2D 회전 행렬
            if self.args.resize_max > 0:
                self.augment_type.append("resize")  # 크기조정: 스케일 인수로 리사이즈
        
        # ===== 기본 이미지 변환 파이프라인 =====
        # 최종 이미지를 resize_width x resize_width로 리사이즈
        #방식만 적어둔 것. 사용은 나중에 img1에 적용 
        self.base_transform = transforms.Compose(
            [
                transforms.Resize([self.args.resize_width, self.args.resize_width]),
            ]
        )
        
        # ===== 쿼리 이미지 변환 파이프라인 =====
        # 쿼리 이미지(항공 사진, 보통 그레이스케일)를 3채널 텐서로 변환
        self.query_transform = transforms.Compose(
            [
                # 그레이스케일 이미지를 3채널로 변환 (RGB로 통일)
                #transforms.Grayscale(num_output_channels=3),
                # PIL Image를 PyTorch 텐서로 변환 (값 범위: 0~1)
                transforms.ToTensor()
            ]
        )
        
    # =========================================================================
    # 2-1. 회전 변환 함수
    # =========================================================================
    
    def rotate_transform(self, rotation, four_point_org, four_point_1, four_point_org_augment, four_point_1_augment):
        """
        네 코너 포인트를 회전 변환.
        2D 회전 공식: P' = R(θ) * (P - C) + C
        여기서 R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]], C는 회전 중심
        
        Parameters:
            rotation: 회전 각도 (라디안). 양수 = 반시계방향
            four_point_org: 원본 이미지의 4 코너 좌표, 형태: (1, 4, 2)
                - [0,0] = 좌상단 (top-left)
                - [0,1] = 우상단 (top-right)
                - [0,2] = 좌하단 (bottom-left)
                - [0,3] = 우하단 (bottom-right)
            four_point_1: 변환된(위성) 이미지의 4 코너 좌표, 형태: (1, 4, 2)
            four_point_org_augment: 증강 후 원본 이미지 4 코너 (출력, in-place 수정)
            four_point_1_augment: 증강 후 변환 이미지 4 코너 (출력, in-place 수정)
        
        Returns:
            증강된 두 세트의 4 포인트 좌표
        """
        # 원본 이미지의 회전 중심: 이미지 중앙
        # resize_width-1은 픽셀 좌표의 마지막 유효 인덱스 (0-indexed)
        center_x_org = torch.tensor((self.args.resize_width - 1) / 2)
        
        # 변환 이미지의 회전 중심: 4 코너의 중점
        # 일반적으로 이미지 중앙이지만, 호모그래피로 변환된 이미지는 왜곡될 수 있음
        # 원본 코드의 방식: 좌상단[0,0] + 우하단[0,3]의 평균
        center_x_1 = (four_point_1[0, 0, :] + four_point_1[0, 3, :]) / 2
        
        # ===== 원본 이미지의 4 코너 회전 변환 =====
        # 각 코너마다 2D 회전 공식 적용
        
        # 코너 0: 좌상단 (top-left)
        # x' = (x - cx) * cos θ - (y - cy) * sin θ + cx
        # y' = (x - cx) * sin θ + (y - cy) * cos θ + cy
        four_point_org_augment[0, 0, 0] = (four_point_org[0, 0, 0] - center_x_org) * torch.cos(rotation) - (four_point_org[0, 0, 1] - center_x_org) * torch.sin(rotation) + center_x_org
        four_point_org_augment[0, 0, 1] = (four_point_org[0, 0, 0] - center_x_org) * torch.sin(rotation) + (four_point_org[0, 0, 1] - center_x_org) * torch.cos(rotation) + center_x_org
        
        # 코너 1: 우상단 (top-right)
        four_point_org_augment[0, 1, 0] = (four_point_org[0, 1, 0] - center_x_org) * torch.cos(rotation) - (four_point_org[0, 1, 1] - center_x_org) * torch.sin(rotation) + center_x_org
        four_point_org_augment[0, 1, 1] = (four_point_org[0, 1, 0] - center_x_org) * torch.sin(rotation) + (four_point_org[0, 1, 1] - center_x_org) * torch.cos(rotation) + center_x_org
        
        # 코너 2: 좌하단 (bottom-left)
        four_point_org_augment[0, 2, 0] = (four_point_org[0, 2, 0] - center_x_org) * torch.cos(rotation) - (four_point_org[0, 2, 1] - center_x_org) * torch.sin(rotation) + center_x_org
        four_point_org_augment[0, 2, 1] = (four_point_org[0, 2, 0] - center_x_org) * torch.sin(rotation) + (four_point_org[0, 2, 1] - center_x_org) * torch.cos(rotation) + center_x_org
        
        # 코너 3: 우하단 (bottom-right)
        four_point_org_augment[0, 3, 0] = (four_point_org[0, 3, 0] - center_x_org) * torch.cos(rotation) - (four_point_org[0, 3, 1] - center_x_org) * torch.sin(rotation) + center_x_org
        four_point_org_augment[0, 3, 1] = (four_point_org[0, 3, 0] - center_x_org) * torch.sin(rotation) + (four_point_org[0, 3, 1] - center_x_org) * torch.cos(rotation) + center_x_org
        
        # ===== 변환 이미지의 4 코너 회전 변환 =====
        # 원본 이미지와 동일하지만 중심이 다름
        
        # 코너 0: 좌상단
        four_point_1_augment[0, 0, 0] = (four_point_1[0, 0, 0] - center_x_1[0]) * torch.cos(rotation) - (four_point_1[0, 0, 1] - center_x_1[1]) * torch.sin(rotation) + center_x_1[0]
        four_point_1_augment[0, 0, 1] = (four_point_1[0, 0, 0] - center_x_1[0]) * torch.sin(rotation) + (four_point_1[0, 0, 1] - center_x_1[1]) * torch.cos(rotation) + center_x_1[1]
        
        # 코너 1: 우상단
        four_point_1_augment[0, 1, 0] = (four_point_1[0, 1, 0] - center_x_1[0]) * torch.cos(rotation) - (four_point_1[0, 1, 1] - center_x_1[1]) * torch.sin(rotation) + center_x_1[0]
        four_point_1_augment[0, 1, 1] = (four_point_1[0, 1, 0] - center_x_1[0]) * torch.sin(rotation) + (four_point_1[0, 1, 1] - center_x_1[1]) * torch.cos(rotation) + center_x_1[1]
        
        # 코너 2: 좌하단
        four_point_1_augment[0, 2, 0] = (four_point_1[0, 2, 0] - center_x_1[0]) * torch.cos(rotation) - (four_point_1[0, 2, 1] - center_x_1[1]) * torch.sin(rotation) + center_x_1[0]
        four_point_1_augment[0, 2, 1] = (four_point_1[0, 2, 0] - center_x_1[0]) * torch.sin(rotation) + (four_point_1[0, 2, 1] - center_x_1[1]) * torch.cos(rotation) + center_x_1[1]
        
        # 코너 3: 우하단
        four_point_1_augment[0, 3, 0] = (four_point_1[0, 3, 0] - center_x_1[0]) * torch.cos(rotation) - (four_point_1[0, 3, 1] - center_x_1[1]) * torch.sin(rotation) + center_x_1[0]
        four_point_1_augment[0, 3, 1] = (four_point_1[0, 3, 0] - center_x_1[0]) * torch.sin(rotation) + (four_point_1[0, 3, 1] - center_x_1[1]) * torch.cos(rotation) + center_x_1[1]
        
        # 디버깅 출력 (현재 주석 처리됨)
        # print("ori:", four_point_org[0, 0, 0], four_point_org[0, 0, 1], four_point_1[0, 0, 0], four_point_1[0, 0, 1])
        # print("now:", four_point_org_augment[0, 0, 0], four_point_org_augment[0, 0, 1], four_point_1_augment[0, 0, 0], four_point_1_augment[0, 0, 1])
        # print("center:", center_x_1, four_point_1[0, 0, :], four_point_1[0, 3, :])
        
        return four_point_org_augment, four_point_1_augment

    # =========================================================================
    # 2-2. 크기 조정 변환 함수
    # =========================================================================
    
    def resize_transform(self, scale_factor, beta, alpha, four_point_org_augment, four_point_1_augment):
        """
        이미지 크기 조정에 따른 4 코너 포인트 좌표 변환.
        크기 조정은 본질적으로 이미지의 가장자리에 빈 공간을 추가하는 것.
        
        Parameters:
            scale_factor: 스케일 인수
                - 1.0 = 원본 크기
                - > 1.0 = 확대 (이미지가 더 큼)
                - < 1.0 = 축소 (이미지가 더 작음)
            beta: 크롭 너비 / 리사이즈 너비 비율 = crop_width / resize_width
            alpha: DB 크기 / 리사이즈 너비 비율 = database_size / resize_width
            four_point_org_augment: 원본 이미지 4 포인트 (수정됨)
            four_point_1_augment: 변환 이미지 4 포인트 (수정됨)
        
        Returns:
            크기 조정된 좌표
        """
        # 가장자리 오프셋 계산
        # 스케일이 1보다 작으면 (축소) offset이 양수 → 이미지 크기 감소
        # 스케일이 1보다 크면 (확대) offset이 음수 → 가장자리 확장
        offset = self.args.resize_width * (1 - scale_factor) / 2
        
        # ===== 원본 이미지의 4 코너 좌표 조정 =====
        # 크기 조정은 이미지 중심을 기준으로 진행됨
        # 따라서 좌측/상측 코너는 offset을 더하고, 우측/하측 코너는 offset을 뺌
        
        # 코너 0: 좌상단 (top-left) → 좌측, 상측 모두 offset 추가
        four_point_org_augment[0, 0, 0] += offset  # x 증가 (좌측으로)
        four_point_org_augment[0, 0, 1] += offset  # y 증가 (상측으로)
        
        # 코너 1: 우상단 (top-right) → 우측 offset 감소, 상측 offset 추가
        four_point_org_augment[0, 1, 0] -= offset  # x 감소 (우측으로)
        four_point_org_augment[0, 1, 1] += offset  # y 증가 (상측으로)
        
        # 코너 2: 좌하단 (bottom-left) → 좌측 offset 추가, 하측 offset 감소
        four_point_org_augment[0, 2, 0] += offset  # x 증가 (좌측으로)
        four_point_org_augment[0, 2, 1] -= offset  # y 감소 (하측으로)
        
        # 코너 3: 우하단 (bottom-right) → 우측, 하측 모두 offset 감소
        four_point_org_augment[0, 3, 0] -= offset  # x 감소 (우측으로)
        four_point_org_augment[0, 3, 1] -= offset  # y 감소 (하측으로)
        
        # ===== 변환 이미지의 4 코너 좌표 조정 =====
        # 변환 이미지에는 스케일 조정된 offset을 적용
        # offset * beta / alpha 이유:
        # - beta: 크롭/리사이즈 비율 (변환 이미지 기준)
        # - alpha: DB/리사이즈 비율 (변환 이미지의 원본 DB 크기 기준)
        # - 결합: 변환 이미지 좌표계에 맞춤
        
        # 코너 0: 좌상단
        four_point_1_augment[0, 0, 0] += offset * beta / alpha
        four_point_1_augment[0, 0, 1] += offset * beta / alpha
        
        # 코너 1: 우상단
        four_point_1_augment[0, 1, 0] -= offset * beta / alpha
        four_point_1_augment[0, 1, 1] += offset * beta / alpha
        
        # 코너 2: 좌하단
        four_point_1_augment[0, 2, 0] += offset * beta / alpha
        four_point_1_augment[0, 2, 1] -= offset * beta / alpha
        
        # 코너 3: 우하단
        four_point_1_augment[0, 3, 0] -= offset * beta / alpha
        four_point_1_augment[0, 3, 1] -= offset * beta / alpha
        
        return four_point_org_augment, four_point_1_augment

    # =========================================================================
    # 2-3. 메인 데이터 로딩 함수
    # =========================================================================
    
    def __getitem__(self, query_PIL_image, database_PIL_image, query_utm, database_utm, index, neg_img2=None):
        """
        단일 데이터 샘플 반환.
        
        전체 파이프라인:
        1. 쿼리 이미지 → 그레이스케일 → 3채널 텐서 (256x256)
        2. UTM 좌표 차이 계산 → 호모그래피 계산
        3. (선택) 데이터 증강: 회전/크기조정/원근감
        4. 쿼리 이미지에 호모그래피 적용 (왜곡)
        5. 최종 크롭 및 리사이즈
        6. 모든 픽셀의 호모그래피 변환 → 광학 흐름 계산
        
        Parameters:
            query_PIL_image: 쿼리 이미지 (PIL Image), 보통 항공 사진
            database_PIL_image: 데이터베이스 이미지 (PIL Image), 보통 위성 사진
            query_utm: 쿼리 이미지의 UTM 좌표 (torch tensor), 형태: (1, 2)
                - [0]: 동쪽(Easting)
                - [1]: 북쪽(Northing)
            database_utm: 데이터베이스 이미지의 UTM 좌표 (torch tensor), 형태: (1, 2)
            index: 쿼리 이미지의 인덱스 (정수)
            pos_index: 양성(positive) 데이터베이스 이미지의 인덱스 (정수)
            neg_img2: 음성(negative) 이미지 (선택사항, 현재 사용 안 함)
        
        Returns:
            img2: 데이터베이스 이미지 (PIL Image 또는 텐서, 미정규화)
            img1: 쿼리 이미지 (torch tensor), 형태: (3, resize_width, resize_width), 정규화됨
            flow: 광학 흐름 (torch tensor), 형태: (2, resize_width, resize_width)
                - flow[0]: x 방향 이동량
                - flow[1]: y 방향 이동량
            H: 호모그래피 행렬 (torch tensor), 형태: (3, 3)
            query_utm, database_utm, index, pos_index: 메타데이터
        """
        
        # ===== 난수 생성기 초기화 =====
        # 멀티프로세싱 환경에서 각 워커가 독립적인 난수 시드를 가져야 함
        if hasattr(self, "rng") and self.rng is None:
            # 현재 워커 정보 가져오기
            worker_info = torch.utils.data.get_worker_info()
            # 워커 ID를 시드로 하는 난수 생성기 생성
            self.rng = np.random.default_rng(seed=worker_info.id)

        # 이미지 할당
        img1 = query_PIL_image  # 쿼리 이미지
        img2 = database_PIL_image  # DB 이미지 (img1을 이것으로 왜곡)

        # 이미지 크기 가져오기 (PIL Image.size = (width, height))
        height, width = img1.size
        
        # ===== UTM 좌표 차이 계산 =====
        # 호모그래피는 두 이미지 간의 기하학적 변환을 나타냄
        # 근사적으로는 위치 차이(displacement)로 모델링 가능
        t = np.float32(np.array(query_utm - database_utm))  # (1, 2) 배열
        # UTM 좌표는 (Easting, Northing) 순서이지만,
        # 이미지 좌표는 (x, y)이고 x는 Northing(북쪽)에 대응, y는 Easting(동쪽)에 대응
        # 따라서 좌표 스왑 필요
        t[0][0], t[0][1] = t[0][1], t[0][0]  # Swap!
        
        # ===== 쿼리 이미지 전처리 =====
        # 그레이스케일 → 3채널 텐서 (0~1 범위)
        img1 = self.query_transform(img1)  # 출력 형태: (3, H, W)
        
        # 변환값 정규화
        # DB 이미지는 database_size 크기로 저장되어 있지만,
        # 호모그래피는 resize_width 크기로 계산되므로 정규화 필요
        # alpha = database_size / resize_width (확대 비율)
        # t를 alpha로 나누면 resize_width 기준 좌표가 됨
        alpha = self.args.database_size / self.args.resize_width
        t = t / alpha  # 리사이즈된 이미지에 맞게 변환값 정규화

        if isinstance(database_PIL_image, Image.Image):
            img2 = transforms.ToTensor()(database_PIL_image)
        else:
            img2 = database_PIL_image
        
        # 텐서로 변환 및 배치 차원 제거
        t_tensor = torch.Tensor(t).squeeze(0)  # (2,)
        
        # ===== 격자점 생성 =====
        # 모든 픽셀 위치를 나열하여 나중에 호모그래피로 변환하고 광학 흐름 계산
        # mgrid: 다차원 격자 생성
        y_grid, x_grid = np.mgrid[0:self.args.resize_width, 0:self.args.resize_width]
        # x_grid[i,j] = j, y_grid[i,j] = i
        # 평탄화 및 결합: [[x0, y0], [x1, y1], ...]
        point = np.vstack((x_grid.flatten(), y_grid.flatten())).transpose()  # (N, 2), N = resize_width^2
        
        # ===== 원본 이미지의 4 코너 포인트 설정 =====
        # 호모그래피는 4개 대응점으로부터 계산됨
        # 4 코너를 사용하면 이미지 전체의 변환을 제약할 수 있음
        four_point_org = torch.zeros((2, 2, 2))  # (2, 2, 2) 형태
        # [0]: 원본 이미지 기준 코너
        # [1]: 나중에 증강된 코너로 업데이트됨 (나중에 사용)
        
        # 4 코너 좌표 정의 (픽셀 좌표, 0-indexed)
        top_left = torch.Tensor([0, 0])  # (0, 0)
        top_right = torch.Tensor([self.args.resize_width - 1, 0])  # (255, 0) for 256x256
        bottom_left = torch.Tensor([0, self.args.resize_width - 1])  # (0, 255)
        bottom_right = torch.Tensor([self.args.resize_width - 1, self.args.resize_width - 1])  # (255, 255)
        
        # 4 코너 배치 할당
        four_point_org[:, 0, 0] = top_left
        four_point_org[:, 0, 1] = top_right
        four_point_org[:, 1, 0] = bottom_left
        four_point_org[:, 1, 1] = bottom_right
        
        # ===== 변환된(DB) 이미지의 4 코너 포인트 설정 =====
        # DB 이미지는 위성 이미지이고, 쿼리 이미지보다 훨씬 크기 때문에
        # 일반적으로 중앙 부분만 추출하여 사용됨
        # 따라서 database_size에 따라 다른 오프셋 적용
        four_point_1 = torch.zeros((2, 2, 2))
        
        # 각 database_size에 대해 다른 오프셋을 적용하는 이유:
        # 더 큰 DB 이미지는 더 많이 중앙 부분만 사용하므로 오프셋이 더 큼
        
        if self.args.database_size == 512:
            # 512x512: 전체 사용 (오프셋 없음)
            four_point_1[:, 0, 0] = t_tensor + top_left
            four_point_1[:, 0, 1] = t_tensor + top_right
            four_point_1[:, 1, 0] = t_tensor + bottom_left
            four_point_1[:, 1, 1] = t_tensor + bottom_right
            
        elif self.args.database_size == 1024:
            # 1024x1024: 중앙 절반 사용 (1/4씩 오프셋)
            # 리사이즈 후 256x256인 이미지 기준으로 중앙 128x128만 추출하면
            # 원본 1024x1024에서 중앙 512x512를 추출하는 것과 동일
            top_left_resize = torch.Tensor([self.args.resize_width/4, self.args.resize_width/4])
            top_right_resize = torch.Tensor([self.args.resize_width - self.args.resize_width/4 - 1, self.args.resize_width/4])
            bottom_left_resize = torch.Tensor([self.args.resize_width/4, self.args.resize_width - self.args.resize_width/4 - 1])
            bottom_right_resize = torch.Tensor([self.args.resize_width - self.args.resize_width/4 - 1, self.args.resize_width - self.args.resize_width/4 - 1])
            four_point_1[:, 0, 0] = t_tensor + top_left_resize
            four_point_1[:, 0, 1] = t_tensor + top_right_resize
            four_point_1[:, 1, 0] = t_tensor + bottom_left_resize
            four_point_1[:, 1, 1] = t_tensor + bottom_right_resize

        elif self.args.database_size == 98:
            # 1024x1024: 중앙 절반 사용 (1/4씩 오프셋)
            # 리사이즈 후 256x256인 이미지 기준으로 중앙 128x128만 추출하면
            # 원본 1024x1024에서 중앙 512x512를 추출하는 것과 동일
            top_left_resize = torch.Tensor([self.args.resize_width/4, self.args.resize_width/4])
            top_right_resize = torch.Tensor([self.args.resize_width - self.args.resize_width/4 - 1, self.args.resize_width/4])
            bottom_left_resize = torch.Tensor([self.args.resize_width/4, self.args.resize_width - self.args.resize_width/4 - 1])
            bottom_right_resize = torch.Tensor([self.args.resize_width - self.args.resize_width/4 - 1, self.args.resize_width - self.args.resize_width/4 - 1])
            four_point_1[:, 0, 0] = t_tensor + top_left_resize
            four_point_1[:, 0, 1] = t_tensor + top_right_resize
            four_point_1[:, 1, 0] = t_tensor + bottom_left_resize
            four_point_1[:, 1, 1] = t_tensor + bottom_right_resize
        
        elif self.args.database_size == 294:
            # 294x294: 1/4씩 오프셋 (1024와 동일한 비율)
            top_left_resize = torch.Tensor([self.args.resize_width/4, self.args.resize_width/4])
            top_right_resize = torch.Tensor([self.args.resize_width - self.args.resize_width/4 - 1, self.args.resize_width/4])
            bottom_left_resize = torch.Tensor([self.args.resize_width/4, self.args.resize_width - self.args.resize_width/4 - 1])
            bottom_right_resize = torch.Tensor([self.args.resize_width - self.args.resize_width/4 - 1, self.args.resize_width - self.args.resize_width/4 - 1])
            four_point_1[:, 0, 0] = t_tensor + top_left_resize
            four_point_1[:, 0, 1] = t_tensor + top_right_resize
            four_point_1[:, 1, 0] = t_tensor + bottom_left_resize
            four_point_1[:, 1, 1] = t_tensor + bottom_right_resize
            
        elif self.args.database_size == 1200:
            # 1200x1200: 중앙 2/3 사용 (1/3씩 오프셋)
            top_left_resize2 = torch.Tensor([self.args.resize_width/3, self.args.resize_width/3])
            top_right_resize2 = torch.Tensor([self.args.resize_width - self.args.resize_width/3 - 1, self.args.resize_width/3])
            bottom_left_resize2 = torch.Tensor([self.args.resize_width/3, self.args.resize_width - self.args.resize_width/3 - 1])
            bottom_right_resize2 = torch.Tensor([self.args.resize_width - self.args.resize_width/3 - 1, self.args.resize_width - self.args.resize_width/3 - 1])
            four_point_1[:, 0, 0] = t_tensor + top_left_resize2
            four_point_1[:, 0, 1] = t_tensor + top_right_resize2
            four_point_1[:, 1, 0] = t_tensor + bottom_left_resize2
            four_point_1[:, 1, 1] = t_tensor + bottom_right_resize2
            
        elif self.args.database_size == 1400:
            # 1400x1400: 중앙 2/3 사용 (1/3씩 오프셋)
            top_left_resize2 = torch.Tensor([self.args.resize_width/3, self.args.resize_width/3])
            top_right_resize2 = torch.Tensor([self.args.resize_width - self.args.resize_width/3 - 1, self.args.resize_width/3])
            bottom_left_resize2 = torch.Tensor([self.args.resize_width/3, self.args.resize_width - self.args.resize_width/3 - 1])
            bottom_right_resize2 = torch.Tensor([self.args.resize_width - self.args.resize_width/3 - 1, self.args.resize_width - self.args.resize_width/3 - 1])
            four_point_1[:, 0, 0] = t_tensor + top_left_resize2
            four_point_1[:, 0, 1] = t_tensor + top_right_resize2
            four_point_1[:, 1, 0] = t_tensor + bottom_left_resize2
            four_point_1[:, 1, 1] = t_tensor + bottom_right_resize2            
            
        elif self.args.database_size == 1536:
            # 1536x1536: 중앙 2/3 사용 (1/3씩 오프셋)
            top_left_resize2 = torch.Tensor([self.args.resize_width/3, self.args.resize_width/3])
            top_right_resize2 = torch.Tensor([self.args.resize_width - self.args.resize_width/3 - 1, self.args.resize_width/3])
            bottom_left_resize2 = torch.Tensor([self.args.resize_width/3, self.args.resize_width - self.args.resize_width/3 - 1])
            bottom_right_resize2 = torch.Tensor([self.args.resize_width - self.args.resize_width/3 - 1, self.args.resize_width - self.args.resize_width/3 - 1])
            four_point_1[:, 0, 0] = t_tensor + top_left_resize2
            four_point_1[:, 0, 1] = t_tensor + top_right_resize2
            four_point_1[:, 1, 0] = t_tensor + bottom_left_resize2
            four_point_1[:, 1, 1] = t_tensor + bottom_right_resize2
            
        elif self.args.database_size == 2100:
            # 2100x2100: 중앙 2/3 사용 (1/3씩 오프셋)
            top_left_resize2 = torch.Tensor([self.args.resize_width/3, self.args.resize_width/3])
            top_right_resize2 = torch.Tensor([self.args.resize_width - self.args.resize_width/3 - 1, self.args.resize_width/3])
            bottom_left_resize2 = torch.Tensor([self.args.resize_width/3, self.args.resize_width - self.args.resize_width/3 - 1])
            bottom_right_resize2 = torch.Tensor([self.args.resize_width - self.args.resize_width/3 - 1, self.args.resize_width - self.args.resize_width/3 - 1])
            four_point_1[:, 0, 0] = t_tensor + top_left_resize2
            four_point_1[:, 0, 1] = t_tensor + top_right_resize2
            four_point_1[:, 1, 0] = t_tensor + bottom_left_resize2
            four_point_1[:, 1, 1] = t_tensor + bottom_right_resize2

        elif self.args.database_size == 2048:
            # 2048x2048: 중앙 3/4 사용 (3/8씩 오프셋)
            top_left_resize3 = torch.Tensor([self.args.resize_width/8*3, self.args.resize_width/8*3])
            top_right_resize3 = torch.Tensor([self.args.resize_width - self.args.resize_width/8*3 - 1, self.args.resize_width/8*3])
            bottom_left_resize3 = torch.Tensor([self.args.resize_width/8*3, self.args.resize_width - self.args.resize_width/8*3 - 1])
            bottom_right_resize3 = torch.Tensor([self.args.resize_width - self.args.resize_width/8*3 - 1, self.args.resize_width - self.args.resize_width/8*3 - 1])
            four_point_1[:, 0, 0] = t_tensor + top_left_resize3
            four_point_1[:, 0, 1] = t_tensor + top_right_resize3
            four_point_1[:, 1, 0] = t_tensor + bottom_left_resize3
            four_point_1[:, 1, 1] = t_tensor + bottom_right_resize3
            
        elif self.args.database_size == 2560:
            # 2560x2560: 중앙 3/5 사용 (2/5씩 오프셋)
            top_left_resize4 = torch.Tensor([self.args.resize_width/5*2, self.args.resize_width/5*2])
            top_right_resize4 = torch.Tensor([self.args.resize_width - self.args.resize_width/5*2 - 1, self.args.resize_width/5*2])
            bottom_left_resize4 = torch.Tensor([self.args.resize_width/5*2, self.args.resize_width - self.args.resize_width/5*2 - 1])
            bottom_right_resize4 = torch.Tensor([self.args.resize_width - self.args.resize_width/5*2 - 1, self.args.resize_width - self.args.resize_width/5*2 - 1])
            four_point_1[:, 0, 0] = t_tensor + top_left_resize4
            four_point_1[:, 0, 1] = t_tensor + top_right_resize4
            four_point_1[:, 1, 0] = t_tensor + bottom_left_resize4
            four_point_1[:, 1, 1] = t_tensor + bottom_right_resize4
            
        else:
            # 지원하지 않는 database_size
            raise NotImplementedError()
        
        # ===== 포인트 배열 형태 변환 =====
        # 호모그래피 계산을 위해 (2, 2, 2) → (1, 4, 2) 형태로 변환
        # kornia의 get_perspective_transform은 (batch, 4, 2) 입력을 기대함
        four_point_org = four_point_org.flatten(1).permute(1, 0).unsqueeze(0).contiguous()
        # (2, 2, 2) → flatten(1) → (2, 4) → permute(1, 0) → (4, 2) → unsqueeze(0) → (1, 4, 2)
        
        four_point_1 = four_point_1.flatten(1).permute(1, 0).unsqueeze(0).contiguous()
        
        # ===== 데이터 증강 =====
        if self.augment:
            # 증강할 코너 포인트 복제
            four_point_org_augment = four_point_org.clone()
            four_point_1_augment = four_point_1.clone()
            
            # beta: 크롭 비율
            # 최종 이미지는 크롭 후 리사이즈되므로,
            # 광학 흐름 계산 시 크롭 영역만 고려해야 함
            beta = self.args.crop_width / self.args.resize_width
            
            # 훈련 모드와 평가 모드 분기
            if self.args.eval_model is None or self.args.multi_aug_eval:
                # 훈련 모드 또는 평가 시 다중 증강: 무작위 증강 선택
                augment_type_single = random.choice(self.augment_type)
                
                if augment_type_single == "rotate":
                    # 회전 각도 무작위 샘플링
                    # [-rotate_max/2, rotate_max/2] 범위에서 균일 분포
                    rotation = torch.tensor(random.random() - 0.5) * 2 * self.args.rotate_max
                    # 호모그래피는 256x256 이미지 기준
                    four_point_org_augment, four_point_1_augment = self.rotate_transform(rotation, four_point_org, four_point_1, four_point_org_augment, four_point_1_augment)
                    
                elif augment_type_single == "resize":
                    # 크기 조정 비율 무작위 샘플링
                    # [1-resize_max/2, 1+resize_max/2] 범위에서 균일 분포
                    scale_factor = 1 + (random.random() - 0.5) * 2 * self.args.resize_max
                    # 스케일 인수는 양수여야 함 (축소될 수 없음)
                    assert scale_factor > 0
                    # 호모그래피는 256x256 이미지 기준
                    four_point_org_augment, four_point_1_augment = self.resize_transform(scale_factor, beta, alpha, four_point_org_augment, four_point_1_augment)
                    
                elif augment_type_single == "perspective":
                    # 원근감 왜곡: 각 코너에 무작위 오프셋 추가
                    for p in range(4):  # 4개 코너
                        for xy in range(2):  # x, y 좌표
                            # [-perspective_max, perspective_max] 범위의 정수 오프셋
                            t1 = random.randint(-self.args.perspective_max, self.args.perspective_max)
                            four_point_org_augment[0, p, xy] += t1  # 원본 이미지 기준 256
                            # 변환 이미지에는 스케일 조정된 오프셋 적용
                            four_point_1_augment[0, p, xy] += t1 * beta / alpha  # 변환 이미지 기준
                            
                elif augment_type_single == "no":
                    # 증강 없음: 아무 작업도 하지 않음
                    pass
                else:
                    raise NotImplementedError()
                    
            else:
                # 평가 모드: 결정론적 증강 (재현 가능)
                # 난수 생성기는 워커마다 다른 시드를 가짐
                if self.args.rotate_max != 0:
                    # 회전 증강만 적용
                    rotation = torch.tensor(self.rng.random() - 0.5) * 2 * self.args.rotate_max
                    four_point_org_augment, four_point_1_augment = self.rotate_transform(rotation, four_point_org, four_point_1, four_point_org_augment, four_point_1_augment)
                    
                elif self.args.resize_max != 0:
                    # 크기 조정 증강만 적용
                    scale_factor = 1 + (self.rng.random() - 0.5) * 2 * self.args.resize_max
                    assert scale_factor > 0
                    four_point_org_augment, four_point_1_augment = self.resize_transform(scale_factor, beta, alpha, four_point_org_augment, four_point_1_augment)
                    
                elif self.args.perspective_max != 0:
                    # 원근감 왜곡 증강만 적용
                    for p in range(4):
                        for xy in range(2):
                            # 정수 오프셋 (난수 생성기 사용)
                            t1 = self.rng.integers(-self.args.perspective_max, self.args.perspective_max)
                            four_point_org_augment[0, p, xy] += t1
                            four_point_1_augment[0, p, xy] += t1 * beta / alpha
                else:
                    raise NotImplementedError()
            
            # ===== 호모그래피 변환 적용 =====
            # 원본 이미지에 호모그래피 적용하여 데이터 증강
            # H: 원본 이미지 → 증강 이미지로의 변환
            H = tgm.get_perspective_transform(four_point_org, four_point_org_augment)
            H_inverse = torch.inverse(H)
            
            # 쿼리 이미지의 현재 너비
            img1_width = img1.shape[1]
            
            # 원본 이미지 크기의 4 코너 (아직 리사이즈되기 전)
            four_point_raw = torch.zeros((2, 2, 2))
            top_left_raw = torch.Tensor([0, 0])
            top_right_raw = torch.Tensor([img1_width - 1, 0])
            bottom_left_raw = torch.Tensor([0, img1_width - 1])
            bottom_right_raw = torch.Tensor([img1_width - 1, img1_width - 1])
            four_point_raw[:, 0, 0] = top_left_raw
            four_point_raw[:, 0, 1] = top_right_raw
            four_point_raw[:, 1, 0] = bottom_left_raw
            four_point_raw[:, 1, 1] = bottom_right_raw
            four_point_raw = four_point_raw.flatten(1).permute(1, 0).unsqueeze(0).contiguous()
            
            # H_1: 원본 이미지 크기 → resize_width 크기로의 변환 (리사이즈 효과)
            H_1 = tgm.get_perspective_transform(four_point_raw, four_point_org)
            H_1_inverse = torch.inverse(H_1)
            
            # H_total: 원본 → resize_width → 증강 으로의 총 변환
            # 역순으로 적용: (원본 이미지) → H_1_inverse (리사이즈됨) → H_inverse (증강됨)
            H_total = H_1_inverse @ H_inverse @ H_1
            
            # 호모그래피를 이미지에 적용 (워핑)
            img1 = tgm.warp_perspective(img1.unsqueeze(0), H_total, (img1_width, img1_width)).squeeze(0)
            
            # 증강된 포인트로 업데이트 (나중에 광학 흐름 계산에 사용)
            four_point_1 = four_point_1_augment

        # ===== 최종 이미지 전처리 =====
        # 쿼리 이미지 처리
        img1 = transforms.CenterCrop(self.args.crop_width)(img1)  # 중앙 crop_width x crop_width 크롭
        img1 = self.base_transform(img1)  # resize_width x resize_width로 리사이즈

        # ===== 호모그래피 행렬 계산 =====
        # 원본 이미지의 4 코너 → 변환 이미지의 4 코너
        H = tgm.get_perspective_transform(four_point_org, four_point_1)
        H = H.squeeze()  # (1, 3, 3) → (3, 3)
        
        # ===== 광학 흐름 계산 =====
        # 모든 픽셀을 호모그래피로 변환하고, 변환 전후의 이동량을 계산
        
        # cv2.perspectiveTransform은 입력 형태가 (1, N, 2)여야 함
        # point: (N, 2) → [point]: (1, N, 2)
        point_transformed_branch1 = cv2.perspectiveTransform(np.array([point], dtype=np.float64), H.numpy()).squeeze()
        # 출력: (N, 2) 또는 (1, N, 2) → squeeze() → (N, 2)
        
        # 광학 흐름: 변환 후 - 변환 전
        diff_branch1 = point_transformed_branch1 - np.array(point, dtype=np.float64)
        # x 방향 이동: diff_branch1[:, 0]
        # y 방향 이동: diff_branch1[:, 1]
        diff_x_branch1 = diff_branch1[:, 0]
        diff_y_branch1 = diff_branch1[:, 1]

        # 1D 배열을 2D 이미지로 변형
        # resize_width x resize_width 크기의 광학 흐름 맵
        diff_x_branch1 = diff_x_branch1.reshape((img1.shape[1], img1.shape[2]))  # img1.shape = (C, H, W)
        diff_y_branch1 = diff_y_branch1.reshape((img1.shape[1], img1.shape[2]))
        
        # 광학 흐름을 하나의 배열로 결합
        pf_patch = np.zeros((self.args.resize_width, self.args.resize_width, 2))
        pf_patch[:, :, 0] = diff_x_branch1  # x 채널
        pf_patch[:, :, 1] = diff_y_branch1  # y 채널
        
        # numpy 배열 → PyTorch 텐서
        flow = torch.from_numpy(pf_patch).permute(2, 0, 1).float()  # (2, H, W)
        
        # H를 다시 squeeze (불필요하지만 원본 코드 유지)
        H = H.squeeze()
        
        # 반환: (img2, img1, flow, H, query_utm, database_utm, index, pos_index)
        # ⚠️ 주의: img2는 정규화되지 않은 PIL Image 또는 텐서 (원본 코드 유지)
        return img2, img1, flow, H, query_utm, database_utm, index


# ============================================================================
# 2. MYDATA 클래스 - 실제 데이터셋 구현
# ============================================================================

class MYDATA(homo_dataset):
    """
    homo_dataset을 상속받아 HDF5 파일로부터 실제 데이터를 로드하는 클래스.
    
    데이터 구조:
    - Database: 위성 이미지 (대규모 지형도)
    - Queries: 항공 이미지 (작은 쿼리 이미지)
    - UTM 좌표: 각 이미지의 지리적 위치
    
    핵심 기능:
    - KNN으로 거리 기반 양성 샘플 찾기
    - 위성 맵에서 ROI 추출
    - HDF5 파일에서 이미지 로드
    """
    
    def __init__(self, args, datasets_folder="datasets", dataset_name="pitts30k", split="train"):
        """
        MYDATA 초기화
        
        Parameters:
            args: 설정 파라미터
            datasets_folder: 데이터셋 루트 폴더 경로 (기본값: "datasets")
            dataset_name: 데이터셋 이름 (기본값: "pitts30k")
            split: 데이터 분할 (기본값: "train", "val", "test", "extended" 등)
        """
        # 부모 클래스 초기화 (homo_dataset)
        # augment = (args.augment == "img"): img 증강 활성화 여부
        super(MYDATA, self).__init__(args, augment=(args.augment == "img"))
        
        #인자 정리 
        self.args = args
        self.dataset_name = dataset_name
        self.split = split  # train/val/test/extended
        self.pred_utm = None
        
        # ===== 파일 경로 설정 =====
        # HDF5 파일: 이미지와 메타데이터 저장
        self.database_folder_nameh5_path = join(
            datasets_folder, dataset_name, split + "_database.h5"
        )
        # 위성 맵 이미지 (매우 큰 이미지)
        self.database_folder_map_path = join(
            datasets_folder, "maps/satellite/20201117_BingSatellite.png"
        )
        # 맵을 텐서로 로드
        self.database_folder_map_df = F.to_tensor(Image.open(self.database_folder_map_path))
        # 쿼리 이미지 HDF5 파일
        self.queries_folder_h5_path = join(
            datasets_folder, dataset_name, split + "_queries.h5"
        )
        
        # ===== HDF5 파일 열기 (읽기 전용, 병렬 접근 지원) =====
        # swmr=True: 싱글 라이터-멀티 리더 모드 (여러 프로세스에서 동시 읽기 가능)
        database_folder_nameh5_df = h5py.File(self.database_folder_nameh5_path, "r", swmr=True)
        queries_folder_h5_df = h5py.File(self.queries_folder_h5_path, "r", swmr=True)

        # ===== 이미지 이름 → 인덱스 매핑 딕셔너리 생성 =====
        # HDF5 파일의 이미지 데이터는 인덱스로 접근되므로,
        # 이미지 파일명(이름)에서 인덱스를 찾기 위한 매핑 필요
        self.database_name_dict = {}
        self.queries_name_dict = {}
        #251113_수정_순서맞추기
        self.queries_paths =[]
        self.database_paths =[]

        # ===== 데이터베이스 이미지 매핑 =====
        for index, database_image_name in enumerate(database_folder_nameh5_df["image_name"]):
            # HDF5는 바이트 문자열로 저장하므로 디코딩 필요
            database_image_name_decoded = database_image_name.decode("UTF-8")
            
            # 중복 이름 처리: 같은 이름이 여러 번 나타날 수 있음
            # 이 경우 northing(북쪽 좌표)에 작은 노이즈 추가하여 구별
            while database_image_name_decoded in self.database_name_dict:
                northing = [str(float(database_image_name_decoded.split("@")[2]) + 0.00001)]
                database_image_name_decoded = "@".join(
                    list(database_image_name_decoded.split("@")[:2]) + northing + 
                    list(database_image_name_decoded.split("@")[3:])
                )
            self.database_name_dict[database_image_name_decoded] = index
            self.database_paths.append(database_image_name_decoded)
        # ===== 쿼리 이미지 매핑 =====
        for index, queries_image_name in enumerate(queries_folder_h5_df["image_name"]):
            queries_image_name_decoded = queries_image_name.decode("UTF-8")
            
            # 중복 이름 처리
            while queries_image_name_decoded in self.queries_name_dict:
                northing = [str(float(queries_image_name_decoded.split("@")[2]) + 0.00001)]
                queries_image_name_decoded = "@".join(
                    list(queries_image_name_decoded.split("@")[:2]) + northing + 
                    list(queries_image_name_decoded.split("@")[3:])
                )
            self.queries_name_dict[queries_image_name_decoded] = index
            #251113_수정_매칭 순서 맞추기 용 
            self.queries_paths.append(queries_image_name_decoded)
        # ===== 정렬된 이미지 경로 생성 =====
        # 일관성 있는 순서 보장
        # self.database_paths = sorted(self.database_name_dict)
        # self.queries_paths = sorted(self.queries_name_dict)
        #이미지 저장 순서대로 이미지를 저장하기 때문에 순서가 일정해진다. 


        # ===== 이미지 이름에서 UTM 좌표 추출 =====
        # 이미지 이름 형식: "ID@easting@northing@..."
        # "@"로 split하면: [ID, easting, northing, ...]
        self.database_utms = np.array(
            [(path.split("@")[1], path.split("@")[2])
             for path in self.database_paths]
        ).astype(float)  # (N_db, 2) 배열
        
        self.queries_utms = np.array(
            [(path.split("@")[1], path.split("@")[2])
             for path in self.queries_paths]
        ).astype(float)  # (N_q, 2) 배열

        # 여기까지는 대충 인자 정리, 부모 호출 그리고 이미지 query datbase의 이미지 좌표를 정ㄹ리 하는 과정이다. 



        # ===== soft_positives 찾기 (거리 기반 양성 샘플) =====
        # soft positive: val_positive_dist_threshold (기본 25m) 이내의 DB 이미지
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)  # DB 이미지 좌표로 KNN 모델 학습
        
        # 각 쿼리에 대해 threshold 거리 이내의 DB 이미지 찾기
        self.soft_positives_per_query = knn.radius_neighbors(
            self.queries_utms,  # 쿼리 좌표
            radius=args.val_positive_dist_threshold,  # 거리 threshold (기본 25m)
            return_distance=False,  # 인덱스만 반환 (거리는 불필요)
        )  # 결과: (N_q,) 배열, 각 요소는 양성 인덱스 배열

        # ===== hard_negatives 찾기 (거리 기반 음성 샘플) =====
        # hard negative: prior_location_threshold 거리 이상의 DB 이미지
        # -1일 경우 음성 샘플을 사용하지 않음
        if args.prior_location_threshold != -1:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.database_utms)
            
            # 각 쿼리에 대해 threshold 거리 이내의 DB 이미지 찾기
            # (이것이 hard negative임)
            self.hard_negatives_per_query = knn.radius_neighbors(
                self.queries_utms,
                radius=args.prior_location_threshold,
                return_distance=False,
            )

        # ===== 데이터베이스/쿼리 프리픽스 추가 =====
        # 경로 앞에 "database_" 또는 "queries_" 프리픽스 추가
        # 나중에 이미지 타입 구별할 때 사용
        for i in range(len(self.database_paths)):
            self.database_paths[i] = "database_" + self.database_paths[i]
        for i in range(len(self.queries_paths)):
            self.queries_paths[i] = "queries_" + self.queries_paths[i]

        # 전체 이미지 경로 (DB + 쿼리)
        self.images_paths = list(self.database_paths) + list(self.queries_paths)

        # 데이터셋 크기
        self.database_num = len(self.database_paths)
        self.queries_num = len(self.queries_paths)

        # ===== HDF5 파일 닫기 =====
        # __getitem__에서 재열기 (멀티프로세싱 안정성)
        self.database_folder_nameh5_df = None
        self.queries_folder_h5_df = None
        database_folder_nameh5_df.close()
        queries_folder_h5_df.close()
        
        # ===== 양성 샘플이 없는 쿼리 제거 =====
        # 훈련 목적상 모든 쿼리는 최소 하나의 양성 샘플을 가져야 함
        queries_without_any_soft_positive = np.where(
            np.array([len(p) for p in self.soft_positives_per_query], dtype=object) == 0
        )[0]
        
        # 로깅
        if len(queries_without_any_soft_positive) != 0:
            logging.info(
                f"There are {len(queries_without_any_soft_positive)} queries without any positives "
                + "within the training set. They won't be considered as they're useless for training."
            )
        
        # 양성 샘플이 없는 쿼리 제거
        self.soft_positives_per_query = np.delete(self.soft_positives_per_query, queries_without_any_soft_positive)
        self.queries_paths = np.delete(self.queries_paths, queries_without_any_soft_positive)
        self.queries_utms = np.delete(self.queries_utms, queries_without_any_soft_positive, axis=0)

        # 이미지 경로와 쿼리 개수 업데이트
        self.images_paths = list(self.database_paths) + list(self.queries_paths)
        self.queries_num = len(self.queries_paths)
    
        # ===== 테스트 페어 로드 (선택사항) =====
        # 테스트 페어를 미리 계산하면 배치 크기가 달라도 일관된 결과 얻음
        if args.load_test_pairs is not None:
            # 사용자가 지정한 파일 사용
            load_test_pairs_file = args.load_test_pairs
        else:
            # 기본 위치
            load_test_pairs_file = f"cache/{self.split}_{args.val_positive_dist_threshold}_pairs.pth"
        if not os.path.isfile(load_test_pairs_file):
            # 테스트 페어 파일이 없으면 온라인에서 생성
            logging.info("Using online test pairs or generating test pairs. It is possible that different batch size can generate different test pairs.")
            print("Using online test pairs or generating test pairs. It is possible that different batch size can generate different test pairs.")
            self.test_pairs = None
        else:
            # 테스트 페어 파일 로드
            logging.info("Loading cached test pairs to make sure that the test pairs will not change for different batch size.")
            print("Loading cached test pairs to make sure that the test pairs will not change for different batch size.")
            self.test_pairs = torch.load(load_test_pairs_file)

    def get_positive_indexes(self, query_index):
        """
        특정 쿼리의 양성 샘플 인덱스 반환
        
        Parameters:
            query_index: 쿼리 인덱스
        
        Returns:
            양성 샘플 DB 이미지 인덱스 배열
        """
        positive_indexes = self.soft_positives_per_query[query_index]
        return positive_indexes
     
    def __len__(self):
        """데이터셋 크기 반환 (쿼리 개수)"""
        return self.queries_num

    def junhyeok_set_pred_utm(self, pred_utm_x, pred_utm_y):
        """evaluate에서 호출"""
        self.pred_utm = (pred_utm_x, pred_utm_y)

    def __getitem__(self, index):
        """
        단일 데이터 샘플 반환
        
        Parameters:
            index: 쿼리 이미지 인덱스
        
        Returns:
            부모 클래스(homo_dataset)의 __getitem__에서 반환한 값
        """
        
        # ===== HDF5 파일 재열기 =====
        # __init__에서 닫은 파일을 다시 열어야 함
        # 이는 멀티프로세싱 환경에서 각 워커가 독립적으로 파일을 열도록 보장
        if self.queries_folder_h5_df is None:
            self.database_folder_nameh5_df = h5py.File(
                self.database_folder_nameh5_path, "r", swmr=True)
            self.queries_folder_h5_df = h5py.File(
                self.queries_folder_h5_path, "r", swmr=True)
            
        # ===== 쿼리 이미지 로드 =====
        # 명시적 대비 조정 옵션 (일부 데이터셋에서 필요)
        if self.args.G_contrast != "none" and self.split != "extended":
            # 대비 조정 방법 선택
            if self.args.G_contrast == "manual":
                # 수동 대비 조정: contrast_factor=3 (3배)
                img = transforms.functional.adjust_contrast(
                    self._find_img_in_h5(index, database_queries_split="queries"), 
                    contrast_factor=3
                )
            elif self.args.G_contrast == "autocontrast":
                # 자동 대비 조정
                img = transforms.functional.autocontrast(
                    self._find_img_in_h5(index, database_queries_split="queries")
                )
            elif self.args.G_contrast == "equalize":
                # 히스토그램 균등화
                img = transforms.functional.equalize(
                    self._find_img_in_h5(index, database_queries_split="queries")
                )
            else:
                raise NotImplementedError()
        else:
            # 대비 조정 없음
            img = self._find_img_in_h5(index, database_queries_split="queries")
        
        # # ===== 양성 샘플(DB 이미지) 선택 =====
        # if self.test_pairs is not None and not self.args.generate_test_pairs:
        #     # 캐시된 테스트 페어 사용
        #     pos_index = self.test_pairs[index]
            
        # else:
        #     # 양성 샘플 중 무작위 선택
        #     pos_index = random.choice(self.get_positive_indexes(index))

        
        #251112_수정_박준혁
        if self.pred_utm is not None:
            x, y = self.pred_utm
            pos_img = self.junhyeok_find_img_in_map(x, y)
            database_utm = torch.tensor([x, y]).unsqueeze(0)
        else:
            # 첫 배치: 일반 positive 샘플 사용
            #             
            x, y =  44643, 1863
            pos_img = self.junhyeok_find_img_in_map(x, y)
            database_utm = torch.tensor([x, y]).unsqueeze(0)
        
        # if index == 36:
        #      x, y = 1229, 741
        #      pos_img = self.junhyeok_find_img_in_map(x, y)
        #      database_utm = torch.tensor([x, y]).unsqueeze(0)

        # if index == 100:
        #     x, y = 1224, 1031
        #     pos_img = self.junhyeok_find_img_in_map(x, y)
        #     database_utm = torch.tensor([x, y]).unsqueeze(0)


        # #case1
        # if index == 418:
        #      x, y = 1346, 968
        #      pos_img = self.junhyeok_find_img_in_map(x, y)
        #      database_utm = torch.tensor([x, y]).unsqueeze(0)

        # #case2
        # if index == 512:
        #      x, y = 885, 952
        #      pos_img = self.junhyeok_find_img_in_map(x, y)
        #      database_utm = torch.tensor([x, y]).unsqueeze(0)


        # #case3
        # if index == 1364:
        #      x, y = 858, 629
        #      pos_img = self.junhyeok_find_img_in_map(x, y)
        #      database_utm = torch.tensor([x, y]).unsqueeze(0)

        # # if index == 1366:
        # #      x, y = 849, 629
        # #      pos_img = self.junhyeok_find_img_in_map(x, y)
        # #      database_utm = torch.tensor([x, y]).unsqueeze(0)


        # # if index == 1370:
        # #      x, y = 831, 629
        # #      pos_img = self.junhyeok_find_img_in_map(x, y)
        # #      database_utm = torch.tensor([x, y]).unsqueeze(0)


        # # if index == 1391:
        # #      x, y = 736, 626
        # #      pos_img = self.junhyeok_find_img_in_map(x, y)
        # #      database_utm = torch.tensor([x, y]).unsqueeze(0)


        # #case 4     
        # if index == 1740:
        #      x, y = 1406, 652
        #      pos_img = self.junhyeok_find_img_in_map(x, y)
        #      database_utm = torch.tensor([x, y]).unsqueeze(0)



        # #case 5     
        # if index == 1822:
        #      x, y = 1776, 667
        #      pos_img = self.junhyeok_find_img_in_map(x, y)
        #      database_utm = torch.tensor([x, y]).unsqueeze(0)

        # #case 5     
        # if index == 1856:
        #      x, y = 1925, 674
        #      pos_img = self.junhyeok_find_img_in_map(x, y)
        #      database_utm = torch.tensor([x, y]).unsqueeze(0)

        # #case      
        # if index == 1903:
        #      x, y = 1863, 540
        #      pos_img = self.junhyeok_find_img_in_map(x, y)
        #      database_utm = torch.tensor([x, y]).unsqueeze(0)
        # if index == 1950:
        #      x, y = 1651, 530
        #      pos_img = self.junhyeok_find_img_in_map(x, y)
        #      database_utm = torch.tensor([x, y]).unsqueeze(0)


        # if index == 1766:
        #      x, y = 1523, 657
        #      pos_img = self.junhyeok_find_img_in_map(x, y)
        #      database_utm = torch.tensor([x, y]).unsqueeze(0)

        # ===== UTM 좌표 가져오기 =====
        query_utm = torch.tensor(self.queries_utms[index]).unsqueeze(0)  # (1, 2)
        # database_utm = torch.tensor(self.database_utms[pos_index]).unsqueeze(0)  # (1, 2)
    
        # ===== 부모 클래스의 __getitem__ 호출 =====
        # 호모그래피 계산 및 광학 흐름 라벨 생성
        return super(MYDATA, self).__getitem__(img, pos_img, query_utm, database_utm, index)

    def __repr__(self):
        """데이터셋 정보 문자열 표현"""
        return f"< {self.__class__.__name__}, {self.dataset_name} - #database: {self.database_num}; #queries: {self.queries_num} >"
    
    def _find_img_in_h5(self, index, database_queries_split=None):
        """
        HDF5 파일에서 이미지 로드
        
        Parameters:
            index: 이미지 인덱스
            database_queries_split: "database" 또는 "queries"
        
        Returns:
            PIL Image
        """
        # ===== 이미지 이름 추출 =====
        if database_queries_split is None:
            # 경로에서 프리픽스 제거 ("database_" 또는 "queries_")
            image_name = "_".join(self.images_paths[index].split("_")[1:])
            # 프리픽스에서 DB/쿼리 타입 판단
            database_queries_split = self.images_paths[index].split("_")[0]
        else:
            if database_queries_split == "database":
                # 데이터베이스 이미지 경로에서 이름 추출
                image_name = "_".join(self.database_paths[index].split("_")[1:])
            elif database_queries_split == "queries":
                # 쿼리 이미지 경로에서 이름 추출
                image_name = "_".join(self.queries_paths[index].split("_")[1:])
            else:
                raise KeyError("Dont find correct database_queries_split!")

        # ===== HDF5에서 이미지 로드 =====
        if database_queries_split == "database":
            # 데이터베이스 이미지 로드
            img = Image.fromarray(
                self.database_folder_nameh5_df["image_data"][
                    self.database_name_dict[image_name]  # 이름에서 인덱스 조회
                ]
            )
        elif database_queries_split == "queries":
            # 쿼리 이미지 로드
            img = Image.fromarray(
                self.queries_folder_h5_df["image_data"][
                    self.queries_name_dict[image_name]  # 이름에서 인덱스 조회
                ]
            )
        else:
            raise KeyError("Dont find correct database_queries_split!")

        return img
    
    def _find_img_in_map(self, index, database_queries_split=None):
        """
        위성 맵 이미지에서 ROI(Region of Interest) 추출
        
        Parameters:
            index: 이미지 인덱스
            database_queries_split: "database"만 지원
        
        Returns:
            PIL Image (ROI 부분)
        """
        # 데이터베이스 이미지만 지원
        if database_queries_split != 'database':
            raise NotImplementedError()
        
        # ===== 이미지 이름에서 UTM 좌표 추출 =====
        image_name = "_".join(self.database_paths[index].split("_")[1:])
        # 이미지 이름 형식: "ID@easting@northing@..."
        img = self.database_folder_map_df  # 전체 위성 맵 (매우 큼, 예: 20000x20000)
        
        # UTM 중심 좌표
        center_cood = (float(image_name.split("@")[1]), float(image_name.split("@")[2]))
        # center_cood[0]: easting (x 좌표)
        # center_cood[1]: northing (y 좌표)
        
        # ===== ROI 영역 정의 =====
        # 중심을 기준으로 정사각형 영역 추출
        # database_size: ROI 크기 (예: 512x512, 1024x1024 등)
        area = (
            int(center_cood[1]) - self.args.database_size // 2,  # left (맵 위의 x 시작)
            int(center_cood[0]) - self.args.database_size // 2,   # top (맵 위의 y 시작)
            int(center_cood[1]) + self.args.database_size // 2,   # right (맵 위의 x 끝)
            int(center_cood[0]) + self.args.database_size // 2    # bottom (맵 위의 y 끝)
        )
        
        # ROI 크롭
        img = F.crop(
            img=img, 
            top=area[1],  # 시작 y
            left=area[0],  # 시작 x
            height=area[3] - area[1],  # 높이 (= database_size)
            width=area[2] - area[0]  # 너비 (= database_size)
        )
        
        return img


    #251113
    def junhyeok_find_img_in_map(self, input1, input2):
        """
        UTM 좌표로 직접 ROI 추출
        
        Parameters:
            input1: easting (x 좌표)
            input2: northing (y 좌표)
        
        Returns:
            PIL Image (ROI 부분)
        """
        
        # 전체 위성 맵 가져오기
        img = self.database_folder_map_df
        
        # ROI 영역 정의
        area = (
            int(input1) - self.args.database_size // 2,  # left
            int(input2) - self.args.database_size // 2,  # top
            int(input1) + self.args.database_size // 2,  # right
            int(input2) + self.args.database_size // 2   # bottom
        )
        
        # ROI 크롭
        img = F.crop(
            img=img, 
            top=area[1],                      # 시작 y
            left=area[0],                     # 시작 x
            height=area[3] - area[1],         # 높이
            width=area[2] - area[0]           # 너비
        )
        
        return img





# ============================================================================
# 3. 데이터로더 생성 함수
# ============================================================================

def fetch_dataloader(args, split='train'):
    """
    PyTorch DataLoader 생성
    
    Parameters:
        args: 설정 파라미터
        split: 데이터 분할 ('train', 'val', 'test', 'extended')
    
    Returns:
        PyTorch DataLoader
    """
    # 데이터셋 생성
    train_dataset = MYDATA(args, args.datasets_folder, args.dataset_name, split)
    
    # ===== 훈련/확장 세트 =====
    if split == 'train' or split == 'extended':
        train_loader = data.DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            pin_memory=True,  # GPU 메모리로 고정 (PCIe 전송 가속화)
            shuffle=True,  # 에포크마다 순서 변경 (훈련 안정성 향상)
            num_workers=args.num_workers,  # 병렬 데이터 로드 워커 개수
            drop_last=True,  # 마지막 불완전 배치 제거 (배치 정규화 안정성)
            worker_init_fn=seed_worker  # 각 워커 난수 시드 설정
        )
    
    # ===== 검증/테스트 세트 =====
    elif split == 'val' or split == 'test':
        # 난수 생성기 고정 (재현성 보장)
        g = torch.Generator()
        g.manual_seed(0)
        
        train_loader = data.DataLoader(
            train_dataset, 
            batch_size=args.batch_size,
            pin_memory=True,
            shuffle=False,  # 순서 유지 (일관된 평가)
            num_workers=args.num_workers,
            drop_last=False,  # 모든 샘플 포함 (정확한 평가)
            worker_init_fn=seed_worker, 
            generator=g
        )
    
    # 로깅
    logging.info(f"{split} set: {train_dataset}")
    return train_loader


# ============================================================================
# 4. 데이터로더 워커 초기화 함수
# ============================================================================

def seed_worker(worker_id):
    """
    각 데이터로더 워커의 난수 생성기 시드 설정
    
    목적: 멀티프로세싱 환경에서 각 워커가 독립적이면서도 
          결정론적인 난수를 생성하도록 보장
    
    Parameters:
        worker_id: 워커 ID (PyTorch에서 자동 전달)
    """
    # 메인 프로세스의 난수 상태에서 유일한 값 추출
    worker_seed = torch.initial_seed() % 2**32
    # numpy 난수 생성기 시드 설정
    np.random.seed(worker_seed)
    # 파이썬 표준 random 모듈 시드 설정
    random.seed(worker_seed)