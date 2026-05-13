# current-player value fix

# 1) regenerate data with value targets matching current-player observations
./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=generate \
  --game='carcassonne(max_turns=10)' \
  --dataset=/tmp/car10_puremcts_train_cpvalue.nop \
  --holdout_dataset=/tmp/car10_puremcts_holdout_cpvalue.nop \
  --samples=131072 \
  --holdout_samples=32768 \
  --teacher=pure_mcts \
  --max_simulations=1600 \
  --rollout_count=8 \
  --mcts_policy_temperature=0.5 \
  --value_is_current_player=true \
  --num_workers=16 \
  --seed=11

# 2) supervised warm start with the same current-player value semantics
./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=train \
  --game='carcassonne(max_turns=10)' \
  --dataset=/tmp/car10_puremcts_train_cpvalue.nop \
  --holdout_dataset=/tmp/car10_puremcts_holdout_cpvalue.nop \
  --student_path=/tmp/az10_cpvalue_puremcts_64x8 \
  --init_from_checkpoint=false \
  --nn_model=resnet \
  --nn_width=64 \
  --nn_depth=8 \
  --train_steps=8000 \
  --batch_size=512 \
  --learning_rate=0.001 \
  --weight_decay=0.0001 \
  --value_is_current_player=true \
  --device=/cuda:0 \
  --report_every=100 \
  --save_final_checkpoint=true \
  --save_best_holdout_checkpoint=true \
  --save_checkpoint_every_report=true

# 3) real training: AlphaZero self-play from the warm start, same value semantics
./build/examples/alpha_zero_torch_example \
  --game='carcassonne(max_turns=10)' \
  --path=/tmp/az10_cpvalue_selfplay_64x8 \
  --init_checkpoint=/tmp/az10_cpvalue_puremcts_64x8/checkpoint-2000 \
  --nn_model=resnet \
  --nn_width=64 \
  --nn_depth=8 \
  --learning_rate=0.0001 \
  --weight_decay=0.0001 \
  --max_simulations=160 \
  --actors=16 \
  --evaluators=2 \
  --train_batch_size=512 \
  --replay_buffer_size=65536 \
  --replay_buffer_reuse=3 \
  --policy_alpha=0.15 \
  --policy_epsilon=0.25 \
  --temperature=1 \
  --temperature_drop=10 \
  --value_is_current_player=true \
  --devices=/cuda:0 \
  --checkpoint_freq=5 \
  --max_steps=40

# 4) evaluate only with the matching current-player value evaluator
./build/examples/alpha_zero_torch_game_example \
  --game='carcassonne(max_turns=10)' \
  --player1=az \
  --player2=mcts \
  --az_path=/tmp/az10_cpvalue_selfplay_64x8 \
  --az_checkpoint=-1 \
  --az_device=/cuda:0 \
  --az_value_is_current_player=true \
  --max_simulations=160 \
  --num_games=100 \
  --num_workers=8 \
  --quiet=true

# v8: pure-MCTS policy prior -> prior-guided rollout teacher -> final model

# 1) clean pure MCTS policy labels; do not reuse terminal-return datasets
./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=generate \
  --game='carcassonne(max_turns=10)' \
  --dataset=/tmp/car10_puremcts_train8.nop \
  --holdout_dataset=/tmp/car10_puremcts_holdout8.nop \
  --samples=131072 \
  --holdout_samples=32768 \
  --teacher=pure_mcts \
  --max_simulations=1600 \
  --rollout_count=8 \
  --mcts_policy_temperature=0.5 \
  --num_workers=16 \
  --seed=8

# 2) train a policy prior only
./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=train \
  --game='carcassonne(max_turns=10)' \
  --dataset=/tmp/car10_puremcts_train8.nop \
  --holdout_dataset=/tmp/car10_puremcts_holdout8.nop \
  --student_path=/tmp/az10_puremcts_v8_policy_64x8 \
  --init_from_checkpoint=false \
  --nn_model=resnet \
  --nn_width=64 \
  --nn_depth=8 \
  --train_steps=8000 \
  --batch_size=512 \
  --learning_rate=0.001 \
  --weight_decay=0.0001 \
  --policy_loss_weight=1 \
  --value_loss_weight=0 \
  --l2_loss_weight=1 \
  --device=/cuda:0 \
  --report_every=100 \
  --save_final_checkpoint=true \
  --save_best_holdout_checkpoint=true \
  --save_checkpoint_every_report=true

# 3) generate new labels using the learned prior, but rollout value at leaves
./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=generate \
  --teacher=az_prior_rollout_value \
  --game='carcassonne(max_turns=10)' \
  --az_path=/tmp/az10_puremcts_v8_policy_64x8 \
  --az_checkpoint=-1 \
  --device=/cuda:0 \
  --dataset=/tmp/car10_azprior_rollout_train8.nop \
  --holdout_dataset=/tmp/car10_azprior_rollout_holdout8.nop \
  --samples=131072 \
  --holdout_samples=32768 \
  --max_simulations=640 \
  --rollout_count=8 \
  --mcts_policy_temperature=0.5 \
  --policy_alpha=0.15 \
  --policy_epsilon=0.25 \
  --temperature=1 \
  --temperature_drop=10 \
  --num_workers=16 \
  --seed=9 \
  --inference_batch_size=32 \
  --inference_threads=16

# 4) train the final model on policy-guided rollout data
./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=train \
  --game='carcassonne(max_turns=10)' \
  --dataset=/tmp/car10_azprior_rollout_train8.nop \
  --holdout_dataset=/tmp/car10_azprior_rollout_holdout8.nop \
  --student_path=/tmp/az10_puremcts_v8_azprior_rollout_64x8 \
  --init_from_checkpoint=true \
  --az_path=/tmp/az10_puremcts_v8_policy_64x8 \
  --az_checkpoint=-1 \
  --train_steps=8000 \
  --batch_size=512 \
  --learning_rate=0.0003 \
  --weight_decay=0.0001 \
  --policy_loss_weight=1 \
  --value_loss_weight=0.1 \
  --l2_loss_weight=1 \
  --device=/cuda:0 \
  --report_every=100 \
  --save_final_checkpoint=true \
  --save_best_holdout_checkpoint=true \
  --save_checkpoint_every_report=true

# 5) evaluate the exact v8 final model
./build/examples/alpha_zero_torch_game_example \
  --game='carcassonne(max_turns=10)' \
  --player1=az \
  --player2=mcts \
  --az_path=/tmp/az10_puremcts_v8_azprior_rollout_64x8 \
  --az_checkpoint=-1 \
  --az_device=/cuda:0 \
  --max_simulations=160 \
  --num_games=100 \
  --num_workers=8 \
  --quiet=true

  

# gen mcts dataset
./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=generate \
  --game='carcassonne(max_turns=10)' \
  --dataset=/tmp/car10_puremcts_train6.nop \
  --holdout_dataset=/tmp/car10_puremcts_holdout6.nop \
  --samples=65536 \
  --holdout_samples=16384 \
  --teacher=pure_mcts \
  --max_simulations=1600 \
  --rollout_count=160 \
  --mcts_policy_temperature=1 \
  --num_workers=32 \
  --seed=1


# train mcts dataset
./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=train \
  --game='carcassonne(max_turns=10)' \
  --dataset=/tmp/car10_puremcts_train4.nop \
  --holdout_dataset=/tmp/car10_puremcts_holdout4.nop \
  --student_path=/tmp/az10_pure_mcts_pretrain_from_file_v4_128x16 \
  --init_from_checkpoint=false \
  --nn_model=resnet \
  --nn_width=128 \
  --nn_depth=16 \
  --train_steps=8000 \
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
  --num_workers=16


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
  changed play test and value
  generate 1600 sims pure mcts data car10_puremcts_train4(changed value)
    finished
  train az10_pure_mcts_pretrain_from_file_v4_64x8 from car10_puremcts_train4
    failed badly
  train az10_pure_mcts_pretrain_from_file_v4_32x4 from car10_puremcts_train4
    failed badly
  generate 160 roll pure mcts data car10_puremcts_train5
  train az10_pure_mcts_pretrain_from_file_v4_128x16 from car10_puremcts_train4
