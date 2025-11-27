# 기존 IQR 식 trajectory 뽑는 과정에서 -> median 기준 IQR을 더하기 빼는 과정으로 변경
# trajectory의 확대 
# 


import numpy as np
import os
import torch
import argparse
from model.network import STHN
from utils import setup_seed
import datasets_4cor_img_NewYork_251126 as datasets
from tqdm import tqdm
import cv2
import parser
from datetime import datetime
from os.path import join
import commons
import logging
import matplotlib.pyplot as plt
from collections import defaultdict



def test(args, wandb_log):
    if not args.identity:
        model = STHN(args)

        model_med = torch.load(args.eval_model, map_location='cuda:0')
        for key in list(model_med['netG'].keys()):
            model_med['netG'][key.replace('module.', '')] = model_med['netG'][key]
        for key in list(model_med['netG'].keys()):
            if key.startswith('module'):
                del model_med['netG'][key]
        model.netG.load_state_dict(model_med['netG'], strict=False)

        if args.two_stages:
            model_med = torch.load(args.eval_model, map_location='cuda:0')
            for key in list(model_med['netG_fine'].keys()):
                model_med['netG_fine'][key.replace('module.', '')] = model_med['netG_fine'][key]
            for key in list(model_med['netG_fine'].keys()):
                if key.startswith('module'):
                    del model_med['netG_fine'][key]
            model.netG_fine.load_state_dict(model_med['netG_fine'])

        model.setup()
        model.netG.eval()
        if args.two_stages:
            model.netG_fine.eval()
    else:
        model = None

    if args.test:
        val_dataset = datasets.fetch_dataloader(args, split='test')
    else:
        val_dataset = datasets.fetch_dataloader(args, split='val')

    evaluate_SNet(model, val_dataset, batch_size=args.batch_size, args=args, wandb_log=wandb_log)


def filter_outliers_iqr(points, window_size=10, threshold=0.8, debug_file="debug.txt"):
    """IQR 기반 outlier 제거 및 평균 계산 - 처음만 14개, 그 다음부턴 10개씩"""
    filtered_points = []
    debug_log = []
    
    points_arr = np.array(points)
    min_valid_points = 6
    
    i = 0
    first_window = True
    trajectory_idx = 0
    
    while i < len(points_arr):
        trajectory_idx += 1
        
        # 처음만 14개, 그 다음부턴 10개씩
        if first_window:
            window = points_arr[i:i+14]
            window_size_curr = 14
            i += 14
            first_window = False
        else:
            window = points_arr[i:i+window_size]
            window_size_curr = window_size
            i += window_size
        
        if len(window) == 0:
            continue
        
        # ============================================================
        # 디버깅: 원본 데이터
        # ============================================================
        debug_log.append(f"\n{'='*70}")
        debug_log.append(f"Trajectory {trajectory_idx}")
        debug_log.append(f"{'='*70}")
        debug_log.append(f"Window Size: {window_size_curr} points")
        debug_log.append(f"\n[원본 예측값 (0~{len(window)-1})]")
        for idx, pt in enumerate(window):
            debug_log.append(f"  Point {idx}: ({pt[0]:.2f}, {pt[1]:.2f})")
        
        # IQR 계산
        q1_r, q3_r = np.percentile(window[:, 0], [25, 75])
        q1_c, q3_c = np.percentile(window[:, 1], [25, 75])
        iqr_r = q3_r - q1_r
        iqr_c = q3_c - q1_c
        
        # ============================================================
        # 디버깅: IQR 계산 결과
        # ============================================================
        debug_log.append(f"\n[IQR 계산]")
        debug_log.append(f"  Row: Q1={q1_r:.2f}, Q3={q3_r:.2f}, IQR={iqr_r:.2f}")
        debug_log.append(f"  Col: Q1={q1_c:.2f}, Q3={q3_c:.2f}, IQR={iqr_c:.2f}")
        

        #수정
        median_r = np.median(window[:, 0])
        median_c = np.median(window[:, 1])
        
        lower_r = median_r - threshold * iqr_r
        upper_r = median_r + threshold * iqr_r
        lower_c = median_c - threshold * iqr_c
        upper_c = median_c + threshold * iqr_c
        
        # ============================================================
        # 디버깅: Threshold 범위
        # ============================================================
        debug_log.append(f"\n[Threshold 범위 (threshold={threshold})]")
        debug_log.append(f"  Row: [{lower_r:.2f}, {upper_r:.2f}]")
        debug_log.append(f"  Col: [{lower_c:.2f}, {upper_c:.2f}]")
        
        # Outlier 제거
        valid_mask = (
            (window[:, 0] >= lower_r) & (window[:, 0] <= upper_r) &
            (window[:, 1] >= lower_c) & (window[:, 1] <= upper_c)
        )
        valid_points = window[valid_mask]
        outlier_indices = [idx for idx in range(len(window)) if not valid_mask[idx]]
        
        # ============================================================
        # 디버깅: 이상치 정보
        # ============================================================
        debug_log.append(f"\n[이상치 탐지]")
        if len(outlier_indices) > 0:
            debug_log.append(f"  이상치 개수: {len(outlier_indices)}개")
            debug_log.append(f"  이상치 인덱스: {outlier_indices}")
            for outlier_idx in outlier_indices:
                outlier_pt = window[outlier_idx]
                reason = []
                if outlier_pt[0] < lower_r or outlier_pt[0] > upper_r:
                    reason.append(f"Row={outlier_pt[0]:.2f} (범위: [{lower_r:.2f}, {upper_r:.2f}])")
                if outlier_pt[1] < lower_c or outlier_pt[1] > upper_c:
                    reason.append(f"Col={outlier_pt[1]:.2f} (범위: [{lower_c:.2f}, {upper_c:.2f}])")
                debug_log.append(f"    Point {outlier_idx}: ({outlier_pt[0]:.2f}, {outlier_pt[1]:.2f}) - {', '.join(reason)}")
        else:
            debug_log.append(f"  이상치 없음")
        
        # ============================================================
        # 디버깅: 유효한 점
        # ============================================================
        debug_log.append(f"\n[유효한 점 ({len(valid_points)}개)]")
        for idx, pt in enumerate(valid_points):
            debug_log.append(f"  Valid Point {idx}: ({pt[0]:.2f}, {pt[1]:.2f})")
        
        # 조건 상관없이 평균 계산 - 모든 점을 빨간색으로 표시
        if len(valid_points) >= min_valid_points:
            mean_point = valid_points.mean(axis=0)
            calc_type = f"유효한 점 평균 ({len(valid_points)}개 ≥ {min_valid_points}개)"
        else:
            mean_point = window.mean(axis=0)
            calc_type = f"전체 평균 ({len(valid_points)}개 < {min_valid_points}개)"
        
        # ============================================================
        # 디버깅: 최종 평균
        # ============================================================
        debug_log.append(f"\n[평균 계산]")
        debug_log.append(f"  방식: {calc_type}")
        debug_log.append(f"  최종 점: ({mean_point[0]:.2f}, {mean_point[1]:.2f})")
        debug_log.append(f"  결과 좌표: ({int(mean_point[0])}, {int(mean_point[1])})")
        
        filtered_points.append(tuple(map(int, mean_point)))
    
    # ============================================================
    # 최종 요약
    # ============================================================
    debug_log.append(f"\n\n{'='*70}")
    debug_log.append("최종 요약")
    debug_log.append(f"{'='*70}")
    debug_log.append(f"생성된 Trajectory 점: {len(filtered_points)}개")
    
    if len(filtered_points) == 0:
        debug_log.append(f"\n⚠️ 경고: 예측점이 0개입니다! 확인이 필요합니다.")
    
    # 디버그 파일 저장
    with open(debug_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(debug_log))
    
    print(f"\n✓ 디버그 로그 저장: {debug_file}")
    
    return filtered_points

def convert_four_pred_to_utm(four_pred, database_utm, args, batch_idx=0):
    """
    four_pred (픽셀 오프셋) → 예측 UTM 좌표로 변환
    
    Args:
        four_pred: 모델 예측 (B, 2, 2, 2) - 픽셀 오프셋
        database_utm: Database 이미지 UTM (B, 1, 2)  ← 변경!
        args: 설정 정보
        batch_idx: 배치 인덱스 (기본값 0)
    
    Returns:
        tuple: (predicted_utm_x, predicted_utm_y)
    """
    
    S = int(args.resize_width)
    img_center_pixels = S / 2
    
    # ===== 예측 네 코너 계산 =====
    four_point_org = torch.zeros((1, 2, 2, 2))
    four_point_org[:, :, 0, 0] = torch.tensor([0, 0])
    four_point_org[:, :, 0, 1] = torch.tensor([S - 1, 0])
    four_point_org[:, :, 1, 0] = torch.tensor([0, S - 1])
    four_point_org[:, :, 1, 1] = torch.tensor([S - 1, S - 1])
    
    dst_pts_pred = (four_pred[batch_idx].cpu().detach().unsqueeze(0) + four_point_org) \
                   .flatten(2).permute(0, 2, 1)[0].numpy()
    center_pred_x = np.mean(dst_pts_pred[:, 0])
    center_pred_y = np.mean(dst_pts_pred[:, 1])
    
    # ===== 픽셀 오프셋 → 미터 변환 =====
    offset_pred_pixels_x = center_pred_x - img_center_pixels
    offset_pred_pixels_y = center_pred_y - img_center_pixels
    
    original_pixel_size_m = 0.25
    database_actual_size_m = args.database_size * original_pixel_size_m
    alpha = database_actual_size_m / args.resize_width
    
    offset_pred_meters_x = offset_pred_pixels_x * alpha
    offset_pred_meters_y = offset_pred_pixels_y * alpha
    
    # ===== 예측 UTM 좌표 (Database UTM + 오프셋) =====
    database_utm_x = database_utm[batch_idx, 0, 0].item()
    database_utm_y = database_utm[batch_idx, 0, 1].item()
    
    predicted_utm_x = database_utm_x + offset_pred_meters_x  # ← Database UTM!
    predicted_utm_y = database_utm_y + offset_pred_meters_y  # ← Database UTM!
    
    return predicted_utm_x, predicted_utm_y


def evaluate_SNet(model, val_dataset, batch_size=0, args=None, wandb_log=False):
    assert batch_size > 0, "batchsize > 0"

    pred_centers, gt_centers = [], []
    torch.cuda.empty_cache()

    for i_batch, data_blob in enumerate(tqdm(val_dataset)):
        if i_batch > 2100:
            continue

        img1, img2, flow_gt, H, query_utm, database_utm, index = [x for x in data_blob]
        current_batch_size = img1.shape[0] 

        if not args.identity:
            model.set_input(img1, img2, flow_gt)
            with torch.no_grad():
                model.forward()
            four_pred = model.four_pred

            # ✅ 배치 내 모든 샘플에 대해 처리 (개선됨)
            for b_idx in range(current_batch_size):
                # 정답값 (GT)
                gt_center = query_utm[b_idx].cpu().numpy()
                database_utm_gt= database_utm[b_idx].cpu().numpy()
                # 예측값 (Prediction)
                pred_utm_x, pred_utm_y = convert_four_pred_to_utm(
                    four_pred,
                    database_utm,
                    args,
                    batch_idx=b_idx
                )


                pred_center_utm = np.array([pred_utm_x, pred_utm_y])
                
                # logging.info(f"match : {index}")
                # logging.info(f"pred_center_utm {pred_center_utm}")
                # logging.info(f"gt_center_utm {gt_center}")
                # logging.info(f"Database UTM : {database_utm_gt}")
 

                pred_centers.append(pred_center_utm)
                gt_centers.append(gt_center)
                
                # ✅ 다음 iteration을 위해 현재 예측값을 데이터셋에 저장
                val_dataset.dataset.junhyeok_set_pred_utm(pred_utm_x, pred_utm_y)




    # ============================================================
    # 데이터 정렬 및 필터링
    # ============================================================
    pred_arr = np.squeeze(np.array(pred_centers)).astype(float)
    gt_arr = np.squeeze(np.array(gt_centers)).astype(float)
    
    # 배열 형태 보정
    if pred_arr.ndim == 1:
        pred_arr = pred_arr.reshape(1, -1)
        gt_arr = gt_arr.reshape(1, -1)

    pred_list = [tuple(map(int, pt)) for pt in pred_arr.tolist()]

    #reverse 1127 이게 킥이었누. 
    pred_list = [(int(pt[1]), int(pt[0])) for pt in pred_arr.tolist()]

    gt_list = [tuple(map(int, pt)) for pt in gt_arr.tolist()]
        
    print(f"\n[데이터 수집 완료]")
    print(f"GT trajectory points: {len(gt_list)}")
    print(f"Predicted trajectory points (raw): {len(pred_list)}")

    # 위성 이미지 불러오기
    sat_img_path = os.path.join(args.datasets_folder, "maps", "satellite", "20201117_BingSatellite.png")
    
    if not os.path.exists(sat_img_path):
        print(f"오류: 이미지 파일을 찾을 수 없습니다 - {sat_img_path}")
        return
    
    sat_img = cv2.imread(sat_img_path)
    sat_img = cv2.cvtColor(sat_img, cv2.COLOR_BGR2RGB)
    
    print(f"원본 이미지 크기: {sat_img.shape}")
    
    # 좌표 범위 분석 (GT + Pred 모두 고려)
    all_coords = gt_list + pred_list
    if all_coords:
        min_col = min(c for (r, c) in all_coords)
        max_col = max(c for (r, c) in all_coords)
        min_row = min(r for (r, c) in all_coords)
        max_row = max(r for (r, c) in all_coords)
        
        print(f"궤적 좌표 범위: row {min_row}~{max_row}, col {min_col}~{max_col}")
        
        margin = 700
        crop_top = max(0, min_row - margin)
        crop_bottom = min(sat_img.shape[0], max_row + margin)
        crop_left = max(0, min_col - margin)
        crop_right = min(sat_img.shape[1], max_col + margin)
        
        print(f"Crop 범위: top={crop_top}, bottom={crop_bottom}, left={crop_left}, right={crop_right}")
        
        # 이미지 자르기
        cropped_img = sat_img[crop_top:crop_bottom, crop_left:crop_right]
        print(f"자른 후 이미지 크기: {cropped_img.shape}")
        
        # 좌표 오프셋 조정
        adjusted_gt = [(r - crop_top, c - crop_left) for (r, c) in gt_list]
        adjusted_pred = [(r - crop_top, c - crop_left) for (r, c) in pred_list]
        
        # ============================================================
        # 5개씩 묶어서 필터링
        # ============================================================
        os.makedirs("outputs_jeju_median_trajectory", exist_ok=True)
        debug_file_path = os.path.join("outputs_jeju_median_trajectory", "debug.txt")
        filtered_pred = filter_outliers_iqr(np.array(adjusted_pred), debug_file=debug_file_path)
        
        print(f"\n[필터링 결과]")
        print(f"원본 점: {len(adjusted_pred)}개 → 필터링된 점: {len(filtered_pred)}개")
        
        # ============================================================
        # Progressive visualization - GT 고정 + 필터링 점 누적
        # ============================================================
        num_matches = len(filtered_pred)
        
        # 최대 이미지 생성 수 제한
        max_images = getattr(args, 'max_trajectory_images', None)
        if max_images is not None:
            num_matches = min(num_matches, max_images)
            print(f"[제한] 최대 {max_images}개 이미지만 생성합니다.")
        
        print(f"\n{num_matches + 1}개의 이미지 생성 중...")  # +1은 빨간 점 없는 초기 이미지
        os.makedirs("outputs_jeju_median_trajectory", exist_ok=True)

        # ============================================================
        # 0번째: 빨간 점 없는 초기 상태
        # ============================================================

        plt.figure(figsize=(15, 10), dpi=300)
        plt.imshow(cropped_img)
        
        # GT Trajectory - 노란색 선과 점
        if adjusted_gt:
            cols_gt = [c for (r, c) in adjusted_gt]
            rows_gt = [r for (r, c) in adjusted_gt]
            # GT 선
            plt.plot(cols_gt, rows_gt, color="yellow", linewidth=2.0, alpha=0.8, zorder=3)
            # GT 점
            plt.scatter(cols_gt, rows_gt, c="yellow", s=8, alpha=0.8, zorder=4, 
                       edgecolors='yellow', linewidth=0.5, label="GT Trajectory")
        
        plt.gca().invert_yaxis()
        
        # 범례
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=8, label='GT Trajectory'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Predicted Trajectory'),
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.title(f"GT vs Predicted Trajectory - Initial (0 predicted points)", 
                 fontsize=12, pad=20)
        plt.axis("off")
        
        # 저장
        save_path = f"outputs_jeju_median_trajectory/trajectory_match_0000.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  [0/0] 저장됨 - 초기 상태 (빨간 점 없음)")

        # ============================================================
        # 1번째 이후: 빨간 점 누적 추가
        # ============================================================
        for match_idx in range(num_matches):
            plt.figure(figsize=(15, 10), dpi=300)
            plt.imshow(cropped_img)
            
            # ★ GT Trajectory - 노란색 선과 점 (뒤에 배치)
            if adjusted_gt:
                cols_gt = [c for (r, c) in adjusted_gt]
                rows_gt = [r for (r, c) in adjusted_gt]
                # GT 선 먼저 그리기
                plt.plot(cols_gt, rows_gt, color="yellow", linewidth=2.0, alpha=0.8, zorder=3)
                # GT 점들
                plt.scatter(cols_gt, rows_gt, c="yellow", s=8, alpha=0.8, zorder=4, 
                           edgecolors='yellow', linewidth=0.5, label="GT Trajectory")
            
            # ★ Filtered Predicted Trajectory - 빨간색 선 (GT 위에)
            current_filtered_pred = filtered_pred[:match_idx + 1]
            
            if current_filtered_pred:
                # 점들을 선으로 먼저 연결 (GT 위에)
                if len(current_filtered_pred) > 1:
                    cols_line = [c for (r, c) in current_filtered_pred]
                    rows_line = [r for (r, c) in current_filtered_pred]
                    # Predicted 선 (GT 위에 그리기)
                    plt.plot(cols_line, rows_line, color="red", linewidth=2.5, alpha=1.0, zorder=6)
                
                # 누적된 점들 표시
                for i, (r, c) in enumerate(current_filtered_pred):
                    alpha = 1.0
                    size = 20
                    
                    # ★ 모든 점을 빨간색으로 표시 (선 위에)
                    plt.scatter(c, r, c="red", s=size, alpha=alpha, zorder=7, 
                               edgecolors='darkred', linewidth=0.5)
                
                # 최신 점 강조 (별표)
                latest_r, latest_c = current_filtered_pred[-1]
                
                plt.scatter(latest_c, latest_r, c="red", s=120, marker="*", 
                           zorder=10, edgecolors='black', linewidth=1.5)
            
            plt.gca().invert_yaxis()
            
            # ★ 범례
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markersize=8, label='GT Trajectory'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Predicted Trajectory'),
            ]
            plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
            
            plt.title(f"GT vs Predicted Trajectory - Match {match_idx + 1}/{num_matches} ({len(current_filtered_pred)} accumulated)", 
                     fontsize=12, pad=20)
            plt.axis("off")
            
            # 저장
            save_path = f"outputs_jeju_median_trajectory/trajectory_match_{match_idx + 1:04d}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            if (match_idx + 1) % 50 == 0:
                print(f"  [{match_idx + 1}/{num_matches}] 저장됨")
        
        print(f"\n✓ 총 {num_matches + 1}개의 이미지 생성 완료!")
        print(f"저장 위치: outputs_jeju_median_trajectory/trajectory_match_*.png")
        print(f"  - trajectory_match_0000.png: 초기 상태 (빨간 점 없음)")
        print(f"  - trajectory_match_0001.png ~ {num_matches:04d}.png: 점 누적")
    else:
        print("궤적 데이터가 없습니다.")


if __name__ == '__main__':
    args = parser.parse_arguments()
    start_time = datetime.now()
    if not args.identity:
        args.save_dir = join(
            "test",
            args.save_dir,
            args.eval_model.split("/")[-2] if args.eval_model is not None else "none",
            f"{args.dataset_name}-{start_time.strftime('%Y-%m-%d_%H-%M-%S')}",
        )
        commons.setup_logging(args.save_dir, console='info')
    setup_seed(0)
    wandb_log = False
    test(args, wandb_log)

