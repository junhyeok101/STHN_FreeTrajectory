from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
import threading

# 폴더 경로 설정
match_folder = "output_NewYork_251125/match_images"
trajectory_folder = "outputs_NewYork_median_trajectory_251127"
output_folder = "outputs_NewYork_total"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def extract_number(filename):
    """파일명에서 숫자 추출"""
    name_without_ext = os.path.splitext(filename)[0]
    import re
    numbers = re.findall(r'\d+', name_without_ext)
    return int(numbers[-1]) if numbers else 0

match_images = sorted(os.listdir(match_folder), key=extract_number)
trajectory_images = sorted(os.listdir(trajectory_folder), key=extract_number)

num_sets = len(match_images) // 10

# trajectory 캐시: 한 번만 로드하고 리사이징된 버전 저장
traj_cache = {}

def process_single_match(traj_idx, match_img, traj_img_resized):
    """단일 match 이미지 처리"""
    try:
        match_path = os.path.join(match_folder, match_img)
        if not os.path.isfile(match_path):
            return None
        
        # match 이미지 열기
        img_match = Image.open(match_path)
        
        # match 리사이징
        max_width = traj_img_resized.width
        img_match_resized = img_match.resize(
            (max_width, int(img_match.height * max_width / img_match.width)),
            Image.Resampling.LANCZOS
        )
        
        # 전체 높이 계산
        total_height = img_match_resized.height + traj_img_resized.height
        
        # 새로운 이미지 생성
        merged_img = Image.new('RGB', (max_width, total_height))
        
        # 이미지 붙이기
        merged_img.paste(img_match_resized, (0, 0))
        merged_img.paste(traj_img_resized, (0, img_match_resized.height))
        
        # 저장
        output_path = os.path.join(output_folder, match_img)
        merged_img.save(output_path, quality=95)
        
        return f"✓ {match_img}"
    except Exception as e:
        return f"✗ {match_img}: {e}"

# trajectory별로 처리
for traj_idx in range(num_sets):
    match_batch = match_images[traj_idx*10 : traj_idx*10 + 10]
    
    if traj_idx >= len(trajectory_images):
        break
    
    try:
        # trajectory 이미지 한 번만 로드 및 리사이징
        traj_path = os.path.join(trajectory_folder, trajectory_images[traj_idx])
        if not os.path.isfile(traj_path):
            continue
        
        traj_img = Image.open(traj_path)
        
        # 기본 너비 설정 (match 이미지들과 비교하여 최적화)
        max_width = 1024  # 또는 실제 match 이미지 너비로 설정
        traj_resized = traj_img.resize(
            (max_width, int(traj_img.height * max_width / traj_img.width)),
            Image.Resampling.LANCZOS
        )
        
        # trajectory 변환 (RGB로 한 번만)
        if traj_resized.mode != 'RGB':
            traj_resized = traj_resized.convert('RGB')
        
        # 멀티스레딩으로 match 이미지들 병렬 처리
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = [
                executor.submit(process_single_match, traj_idx, match_img, traj_resized)
                for match_img in match_batch
            ]
            
            for future in results:
                result = future.result()
                if result:
                    print(result)
        
    except Exception as e:
        print(f"✗ trajectory 처리 실패: {e}")

print("모든 이미지 통합 완료!")