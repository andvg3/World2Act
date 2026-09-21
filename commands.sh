##run all below command to activate env, export MIOPEN and run code inference cosmos

conda activate tuan.wan.worldmodel.pytorch271

export MIOPEN_ENABLE_LOGGING=0
export MIOPEN_ENABLE_LOGGING_CMD=0
mkdir -p ~/miopen_db_cosmos_idm
mkdir -p ~/miopen_cache_cosmos_idm

export MIOPEN_USER_DB_PATH=$HOME/miopen_db_cosmos_idm
export MIOPEN_SYSTEM_DB_PATH=$HOME/miopen_db_cosmos_idm        
export MIOPEN_CUSTOM_CACHE_DIR=$HOME/miopen_cache_cosmos_idm

export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

python -m examples.video2world_gr00t \
  --model_size 2B \
  --gr00t_variant droid \
  --prompt "press the button on the coffee machine to serve coffee" \
  --input_path /vast/users/tianyu.wang/tuanvvv_workspace/GR00T-Dreams/cosmos-predict2/datasets/benchmark_train/test_new_robocasa/demo_1_grid_2x2.jpg \
  --prompt_prefix "" \
  --disable_guardrail \
  --save_path "output/generated_video_robocasa_model_retrained_4views_PnPpresscoffee_0.mp4"


python -m examples.video2world_gr00t \
  --model_size 2B \
  --gr00t_variant droid \
  --prompt "pick the hot dog from the cabinet" \
  --input_path /vast/users/tianyu.wang/tuanvvv_workspace/GR00T-Dreams/cosmos-predict2/datasets/benchmark_train/test_new_robocasa/demo_1_grid_2x2.jpg \
  --prompt_prefix "" \
  --disable_guardrail \
  --save_path "output/generated_video_robocasa_model_retrained_4views_PnPpickhotdog_0_8000.mp4"


python -m examples.video2world_gr00t \
  --model_size 2B \
  --gr00t_variant droid \
  --prompt "close the right cabinet door" \
  --input_path /vast/users/tianyu.wang/tuanvvv_workspace/GR00T-Dreams/cosmos-predict2/datasets/benchmark_train/test_new_robocasa/demo_1_grid_2x2.jpg \
  --prompt_prefix "" \
  --disable_guardrail \
  --save_path "output/generated_video_robocasa_model_retrained_4views_PnPclose_door_0_8000.mp4"



###traning
cd /vast/users/tianyu.wang/tuanvvv_workspace/GR00T-Dreams/test_code/wan2.1-cosmos2

EXP=predict2_video2world_training_2b_groot_gr1_480
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_predict2/configs/base/config.py -- experiment=${EXP}





### test full task

python -m examples.video2world_gr00t \
  --model_size 2B \
  --gr00t_variant droid \
  --prompt "close the cabinet doors" \
  --input_path /vast/users/tianyu.wang/tuanvvv_workspace/GR00T-Dreams/robomimic-main/first_frame/CloseDoubleDoor/seed_22/eps_00/first_frame.png \
  --prompt_prefix "" \
  --disable_guardrail \
  --save_path "output/cosmos2_fulltask_closthecabinetdoor.mp4"



### test full task-libero

python -m examples.video2world_gr00t_libero \
  --model_size 2B \
  --gr00t_variant droid \
  --prompt "place moka pot on stove" \
  --input_path /vast/users/tianyu.wang/tuanvvv_workspace/GR00T-Dreams/cosmos-predict2/datasets/benchmark_train/RoboCasa_mg/train/first_frames_from_videos_libero/demo_0-turn_on_the_stove_and_put_the_moka_pot_on_it-place_moka_pot_on_stove-order3/frame0.jpg\
  --prompt_prefix "" \
  --disable_guardrail \
  --save_path "output/cosmos2_libero_place_moca.mp4"


#### test for un-seen task robocasa
cd /vast/users/tianyu.wang/tuanvvv_workspace/GR00T-Dreams/test_code/wan2.1-cosmos2

source /vast/users/tianyu.wang/anaconda3/etc/profile.d/conda.sh
unset LD_PRELOAD
unset LD_LIBRARY_PATH
source /vast/users/tianyu.wang/tuanvvv_workspace/GR00T-Dreams/cosmos-predict2/setup_rocm70.sh

CUDA_VISIBLE_DEVICES=0 python -m examples.video2world_gr00t_robocasa_unseen_task_single \
  --model_size 2B \
  --gr00t_variant droid \
  --prompt "The robot arm is performing a task. A multi-view video shows that a robot pick the lime from the cabinet. The video is split into four views: The top-left view shows the robotic arm from the left side, the top-right view shows it from the right side, the bottom-left view shows a first-person perspective from the robot's end-effector (gripper), and the bottom-right view is a black screen (inactive view). The robot pick the lime from the cabinet" \
  --input_path /vast/users/tianyu.wang/tuanvvv_workspace/GR00T-Dreams/cosmos-predict2/output/robocasa_atomicgeneralize_unseen_iter_000018800/PnPCabToCounter/PnPCabToCounter_mg_demo_0_aa_0/first_frame.png \
  --prompt_prefix "" \
  --disable_guardrail \
  --num_gpus 1 \
  --save_path /vast/users/tianyu.wang/tuanvvv_workspace/GR00T-Dreams/cosmos-predict2/output/robocasa_atomicgeneralize_unseen_iter_000018800/PnPCabToCounter/PnPCabToCounter_mg_demo_0_aa_0/generated_iter_000018800_single.mp4