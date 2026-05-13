

# gen mcts dataset
./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=generate \
  --game='carcassonne(max_turns=10)' \
  --dataset=/tmp/car10_puremcts_train2.nop \
  --holdout_dataset=/tmp/car10_puremcts_holdout2.nop \
  --samples=65536 \
  --holdout_samples=16384 \
  --teacher=pure_mcts \
  --max_simulations=1600 \
  --rollout_count=64 \
  --mcts_policy_temperature=1 \
  --num_workers=32 \
  --seed=1


# train mcts dataset
./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=train \
  --game='carcassonne(max_turns=10)' \
  --dataset=/tmp/car10_puremcts_train.nop \
  --holdout_dataset=/tmp/car10_puremcts_holdout.nop \
  --student_path=/tmp/az10_pure_mcts_pretrain_64x8 \
  --init_from_checkpoint=false \
  --nn_model=resnet \
  --nn_width=64 \
  --nn_depth=8 \
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
  --player1=mcts \
  --player2=az \
  --az_path=/tmp/az10_pure_mcts_pretrain_64x8 \
  --az_checkpoint=-3 \
  --az_device=/cuda:0 \
  --max_simulations=160 \
  --num_games=1000 \
  --quiet=true \
  --num_workers=4


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
  --dataset=/tmp/car10_puremcts_train.nop   \
  --holdout_dataset=/tmp/car10_puremcts_holdout.nop   \
  --student_path=/tmp/az10_pure_mcts_pretrain_64x8  \
  --init_from_checkpoint=true   \
  --az_path=/tmp/az10_pure_mcts_pretrain_64x8 \
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
  changed play test and value
  generate 1600 sims pure mcts data car10_puremcts_train4(changed value)
    finished
  train az10_pure_mcts_pretrain_from_file_v4_64x8 from car10_puremcts_train4
    failed badly
  train az10_pure_mcts_pretrain_from_file_v4_32x4 from car10_puremcts_train4
    failed badly
  generate 160 roll pure mcts data car10_puremcts_train5
  train az10_pure_mcts_pretrain_from_file_v4_128x16 from car10_puremcts_train4


./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=generate \
  --game="$GAME" \
  --dataset="$DATA" \
  --holdout_dataset="$HOLD" \
  --samples=32768 \
  --holdout_samples=8192 \
  --teacher=az_prior_rollout_value \
  --az_path="$CUR" \
  --az_checkpoint=$CKPT \
  --az_graph_def=vpnet.pb \
  --max_simulations=400 \
  --rollout_count=8 \
  --mcts_policy_temperature=1.0 \
  --policy_epsilon=0 \
  --policy_alpha=0.03 \
  --temperature=1 \
  --temperature_drop=2 \
  --value_target=terminal \
  --value_is_current_player=$VAL \
  --num_workers=16 \
  --inference_batch_size=64 \
  --inference_threads=8 \
  --inference_cache=65536 \
  --device=/cuda:0 \
  --seed=31

  car10_puremcts_train: 65536 1600 16
  az10_pure_mcts_pretrain_32x4
    3300 step mcts,az:42, 46 44, 42
  car10_puremcts_train2: 65536 1600 64
  az10_pure_mcts_selfplay_smoke_64x8
    step0 一次50, 43(mcts,az) 一次52, 33 50, 45