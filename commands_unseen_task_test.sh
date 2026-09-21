


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