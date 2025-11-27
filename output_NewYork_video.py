import cv2
import os
import natsort
import sys
import subprocess

# === 경로 설정 ===
image_dir = "outputs_NewYork_total_251127"
output_dir = "outputs_NewYork_video_251127"
os.makedirs(output_dir, exist_ok=True)

avi_video = os.path.join(output_dir, "uav_NY_video_temp.avi")
mp4_video = os.path.join(output_dir, "uav_NY_video.mp4")

# === 이미지 정렬 ===
images = [f for f in os.listdir(image_dir) if f.endswith(".png")]
images = natsort.natsorted(images)

if len(images) == 0:
    raise ValueError(f"{image_dir} 안에 PNG 이미지가 없습니다!")

print(f"✅ {len(images)}개 이미지 발견")

# === 첫 이미지 크기 확인 ===
first_img = cv2.imread(os.path.join(image_dir, images[0]))
h, w, c = first_img.shape
print(f"✅ 이미지 해상도: {w}x{h}")

# === 비디오 저장 설정 (AVI + MJPG) ===
fps = 5
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(avi_video, fourcc, fps, (w, h))

if not out.isOpened():
    print("❌ VideoWriter 초기화 실패!")
    sys.exit(1)

print("⏳ AVI 비디오 생성 중...")

for i, img_name in enumerate(images):
    img = cv2.imread(os.path.join(image_dir, img_name))
    
    if img is None:
        print(f"⚠️ 이미지 로드 실패: {img_name}")
        continue
    
    # 크기 불일치 시 리사이즈
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h))
    
    out.write(img)
    
    if (i + 1) % 10 == 0:
        print(f"  {i + 1}/{len(images)} frames written")

out.release()
print(f"✅ AVI 파일 생성 완료: {avi_video}")

# === FFmpeg를 사용하여 MP4로 변환 ===
print("⏳ MP4 변환 중 (FFmpeg)...")

try:
    subprocess.run([
        "ffmpeg", 
        "-i", avi_video,
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        mp4_video,
        "-y"  # 기존 파일 덮어쓰기
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    print(f"✅ MP4 변환 완료: {mp4_video}")
    
    # 임시 AVI 파일 삭제
    if os.path.exists(avi_video):
        os.remove(avi_video)
        print(f"✅ 임시 파일 삭제")
    
    print(f"\n🎉 최종 영상 저장 완료!")
    print(f"📁 경로: {mp4_video}")
    print(f"📊 정보: {len(images)}개 프레임, {fps}fps, {w}x{h} 해상도")

except FileNotFoundError:
    print("❌ FFmpeg가 설치되지 않았습니다!")
    print("설치 방법:")
    print("  - Windows: https://ffmpeg.org/download.html 또는 'choco install ffmpeg'")
    print("  - Mac: brew install ffmpeg")
    print("  - Linux: sudo apt install ffmpeg")
    print(f"\n⚠️ AVI 파일은 생성되었습니다: {avi_video}")
    print("   이 파일을 직접 사용하거나 온라인 변환기로 MP4로 변환하세요.")

except subprocess.CalledProcessError as e:
    print(f"❌ FFmpeg 변환 실패: {e}")
    print(f"⚠️ AVI 파일은 생성되었습니다: {avi_video}")