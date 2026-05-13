

# gen mcts dataset
./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=generate \
  --game='carcassonne(max_turns=10)' \
  --dataset=/tmp/car10_puremcts_train3.nop \
  --holdout_dataset=/tmp/car10_puremcts_holdout3.nop \
  --samples=65536 \
  --holdout_samples=16384 \
  --teacher=pure_mcts \
  --max_simulations=1600 \
  --rollout_count=8 \
  --mcts_policy_temperature=1 \
  --num_workers=16 \
  --seed=1


# train mcts dataset
./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=train \
  --game='carcassonne(max_turns=10)' \
  --dataset=/tmp/car10_puremcts_train3.nop \
  --holdout_dataset=/tmp/car10_puremcts_holdout3.nop \
  --student_path=/tmp/az10_pure_mcts_pretrain_from_file_v3_32x4 \
  --init_from_checkpoint=false \
  --nn_model=resnet \
  --nn_width=32 \
  --nn_depth=4 \
  --train_steps=2000 \
  --batch_size=512 \
  --learning_rate=0.001 \
  --weight_decay=0.0001 \
  --device=/cuda:0 \
  --report_every=100 \
  --save_final_checkpoint=true \
  --save_best_holdout_checkpoint=true \
  --save_checkpoint_every_report=true

# test
./build/examples/alpha_zero_torch_game_example \
  --game='carcassonne(max_turns=10)' \
  --player1=az \
  --player2=mcts \
  --az_path=/tmp/az10_pure_mcts_pretrain_from_file_v3_32x4 \
  --az_checkpoint=-3 \
  --az_device=/cuda:0 \
  --max_simulations=160 \
  --num_games=100 \
  --quiet=true


# gen model dataset
./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=generate \
  --teacher=az \
  --game='carcassonne(max_turns=10)' \
  --az_path=/tmp/az10_pure_mcts_pretrain_from_file_v2_64x8 \
  --az_checkpoint=-3 \
  --device=/cuda:0 \
  --dataset=$PWD/datasets/car10_azmcts_train2.nop \
  --holdout_dataset=$PWD/datasets/car10_azmcts_holdout2.nop \
  --samples=65536 \
  --holdout_samples=16384 \
  --max_simulations=640 \
  --mcts_policy_temperature=1 \
  --policy_alpha=0.15 \
  --policy_epsilon=0.25 \
  --temperature=1 \
  --temperature_drop=1 \
  --num_workers=32 \
  --seed=1 \
  --inference_batch_size=32 \
  --inference_threads=16

# new train from model dataset
./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=train \
  --game='carcassonne(max_turns=10)' \
  --dataset=$PWD/datasets/car10_azmcts_train.nop \
  --holdout_dataset=$PWD/datasets/car10_azmcts_holdout.nop \
  --student_path=/tmp/az10_model_pretrain_64x12_v3 \
  --init_from_checkpoint=false \
  --nn_model=resnet \
  --nn_width=64 \
  --nn_depth=12 \
  --train_steps=2000 \
  --batch_size=512 \
  --learning_rate=0.0001 \
  --weight_decay=0.0001 \
  --device=/cuda:0 \
  --report_every=100 \
  --save_final_checkpoint=true \
  --save_best_holdout_checkpoint=true \
  --save_checkpoint_every_report=true


  ./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=train \
  --game='carcassonne(max_turns=10)' \
  --dataset=$PWD/datasets/car10_azmcts_train.nop \
  --holdout_dataset=$PWD/datasets/car10_azmcts_holdout.nop \
  --student_path=/tmp/az10_model_pretrain_64x12_conti \
  --init_from_checkpoint=true \
  --az_path=/tmp/az10_pure_mcts_pretrain_from_file_v2_64x12 \
  --az_checkpoint=-3 \
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

# continue training
  ./build/examples/alpha_zero_torch_dataset_pretrain   --mode=train   \
  --game='carcassonne(max_turns=10)'   \
  --dataset=/tmp/car10_puremcts_train3.nop   \
  --holdout_dataset=/tmp/car10_puremcts_holdout3.nop   \
  --student_path=/tmp/az10_pure_mcts_pretrain_from_file_v3_32x4  \
  --init_from_checkpoint=true   \
  --az_path=/tmp/az10_pure_mcts_pretrain_from_file_v3_32x4 \
  --az_checkpoint=-1 \
   --train_steps=4000   --batch_size=512   --learning_rate=0.001   --weight_decay=0.0001   --device=/cuda:0   --report_every=100   --save_final_checkpoint=true   --save_best_holdout_checkpoint=true   --save_checkpoint_every_report=true


  # log
  generate low noise data from az10_pure_mcts_pretrain_from_file_v2_64x8
  --policy_alpha=0.3   --policy_epsilon=0
    suspended
  generate 1600 sims pure mcts data car10_puremcts_train3
    finished
  train az10_model_pretrain_64x12_v3 from car10_azmcts_train
  --learning_rate=0.0001
    failed
  train az10_pure_mcts_pretrain_from_file_v3_64x12 from car10_puremcts_train3
    not good enough(+20)
  train az10_pure_mcts_pretrain_from_file_v3_32x4 from car10_puremcts_train3
    not good enough(+24)
  