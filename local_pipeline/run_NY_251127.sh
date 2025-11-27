#251112_knn 매 매칭마다 수행 

# chmod +x run.sh

# #251127 작성, 
# 1. 자유경로 매칭 시각화 코드 +디버깅
# 2. trajectory + log
# 3. 이미지 합치기
# 4. 비디오 저장. 


# # newyork trajectory
    python3 local_pipeline/eval_NewYork_trajectory_251126.py \
    --datasets_folder datasets_NewYork \
    --dataset_name NewYork \
    --eval_model 1536_two_stages/Finetuning_NewYork_1536.pth \
    --val_positive_dist_threshold 100 \
    --lev0 \
    --database_size 1400 \
    --corr_level 4 \
    --test \
    --num_workers 0 \
    --batch_size 1 \
    --output_dir output_newyork_251125 

#newyork match


    python3 local_pipeline/eval_NewYork_251126.py \
    --datasets_folder datasets_NewYork \
    --dataset_name NewYork \
    --eval_model 1536_two_stages/Finetuning_NewYork_1536.pth \
    --val_positive_dist_threshold 100 \
    --lev0 \
    --database_size 1400 \
    --corr_level 4 \
    --test \
    --num_workers 0 \
    --batch_size 1 \
    --output_dir output_NewYork_251125 \


    python3 output_NewYork_make.py

    python3 output_NewYork_video.py