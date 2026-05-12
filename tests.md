

# 16k sample 160 sim 8 roll
./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=generate \
  --game='carcassonne(max_turns=10)' \
  --dataset=/tmp/car10_puremcts_16k_train.nop \
  --holdout_dataset=/tmp/car10_puremcts_4k_holdout.nop \
  --samples=16384 \
  --holdout_samples=4096 \
  --teacher=pure_mcts \
  --max_simulations=160 \
  --rollout_count=8 \
  --mcts_policy_temperature=1 \
  --num_workers=16 \
  --seed=1



./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=train \
  --game='carcassonne(max_turns=10)' \
  --dataset=/tmp/car10_puremcts_16k_train.nop \
  --holdout_dataset=/tmp/car10_puremcts_4k_holdout.nop \
  --student_path=/tmp/az10_pure_mcts_pretrain_from_file_12 \
  --init_from_checkpoint=false \
  --nn_model=resnet \
  --nn_width=64 \
  --nn_depth=12 \
  --train_steps=2000 \
  --batch_size=512 \
  --learning_rate=0.001 \
  --weight_decay=0.0001 \
  --device=/cuda:0 \
  --report_every=100 \
  --save_final_checkpoint=true \
  --save_best_holdout_checkpoint=true \
  --save_checkpoint_every_report=true
# 64x4
  az mcts  mcts az
  49:38 46:42
  ckpt 1100
# 64x8
  50:43
  1200
# 64x12
  64:43
  

# 64k sample 640 sim 8 roll
./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=generate \
  --game='carcassonne(max_turns=10)' \
  --dataset=/tmp/car10_puremcts_train2.nop \
  --holdout_dataset=/tmp/car10_puremcts_holdout2.nop \
  --samples=65536 \
  --holdout_samples=16384 \
  --teacher=pure_mcts \
  --max_simulations=640 \
  --rollout_count=8 \
  --mcts_policy_temperature=1 \
  --num_workers=16 \
  --seed=1