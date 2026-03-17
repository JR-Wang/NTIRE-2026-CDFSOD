python3 ./main.py \
--data_dir ./dataset \
--test_dir ./dataset/CDFSOD/dataset2 \
--json_name_list 1_shot.json 5_shot.json 10_shot.json \
--radio_model_path ./checkpoints/c-radio_v4-h_half.pth.tar \
--save_dir ./results \
--min_threshold 0.01 \
--diffusion_steps 30 \
--alp 0.3 \
--lamb 0.5 

