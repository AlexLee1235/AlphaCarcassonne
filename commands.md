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



./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=train \
  --game='carcassonne(max_turns=10)' \
  --dataset=/tmp/car10_puremcts_train2.nop \
  --holdout_dataset=/tmp/car10_puremcts_holdout2.nop \
  --student_path=/tmp/az10_pure_mcts_pretrain_from_file_v2_64x12 \
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


./build/examples/alpha_zero_torch_game_example \
  --game='carcassonne(max_turns=10)' \
  --player1=az \
  --player2=mcts \
  --az_path=/tmp/az10_pure_mcts_pretrain_from_file_v2_64x12 \
  --az_checkpoint=-3 \
  --az_device=/cuda:0 \
  --max_simulations=160 \
  --num_games=100 \
  --quiet=true



./build/examples/alpha_zero_torch_dataset_pretrain \
  --mode=generate \
  --teacher=az \
  --game='carcassonne(max_turns=10)' \
  --az_path=/tmp/az10_pure_mcts_pretrain_from_file_v2_64x8 \
  --az_checkpoint=-3 \
  --device=/cuda:0 \
  --dataset=$PWD/datasets/car10_azmcts_train.nop \
  --holdout_dataset=$PWD/datasets/car10_azmcts_holdout.nop \
  --samples=65536 \
  --holdout_samples=16384 \
  --max_simulations=640 \
  --mcts_policy_temperature=1 \
  --policy_alpha=1 \
  --policy_epsilon=0.25 \
  --temperature=1 \
  --temperature_drop=10 \
  --num_workers=16 \
  --seed=1
