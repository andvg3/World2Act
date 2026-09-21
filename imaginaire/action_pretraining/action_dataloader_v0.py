import torch
import h5py
import random
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict

class RoboCasaActionDataset(Dataset):
    def __init__(self, root_dir):
        self.root_path = Path(root_dir)
        self.episode_index = []
        self.lookup_map = {}
        # New: Helper map to group demo_nums by task for sampling
        self.task_to_demos = defaultdict(list)
        
        self._build_and_load_index()

    def _build_and_load_index(self):
        """Scans directory and loads ACTIONS into memory immediately."""
        all_hdf5_files = list(self.root_path.rglob("demo_gentex_im128_randcams.hdf5"))
        mg_files = [p for p in all_hdf5_files if "mg" in p.parts]
        
        # print(f"Indexing and caching {len(mg_files)} MG files...")
        
        for h5_path in tqdm(mg_files):
            try:
                with h5py.File(h5_path, 'r') as f:
                    if "data" not in f: continue
                    
                    # Assuming task_name is at a specific depth in the path
                    task_name = h5_path.parts[-4]
                    for demo_key in f["data"].keys():
                        demo_num = int(demo_key.split("_")[-1])
                        
                        # Load actions to RAM
                        actions = torch.from_numpy(f["data"][demo_key]["actions"][:]).float()
                        
                        entry = {
                            "demo_num": demo_num,
                            "task_name": task_name,
                            "actions": actions
                        }
                        
                        self.episode_index.append(entry)
                        self.lookup_map[(task_name, demo_num)] = entry
                        # Track which demo_nums belong to this task
                        self.task_to_demos[task_name].append(demo_num)
                        
            except Exception as e:
                print(f"Error indexing {h5_path}: {e}")

    def __len__(self):
        return len(self.episode_index)

    def __getitem__(self, idx):
        return self.episode_index[idx]

    def get_action(self, task_name, demo_num):
        key = (task_name, demo_num)
        if key not in self.lookup_map:
            raise KeyError(f"Task '{task_name}' Demo {demo_num} not found.")
        return self.lookup_map[key]
    
    def sample_negative_actions(self, anchor_task, anchor_demo_num, num_negatives=4, hard_ratio=0.5):
        """
        Samples a fixed number of negative action episodes.
        """
        negatives = []
        
        target_hard_count = int(num_negatives * hard_ratio)

        # --- 1. Hard Negatives (Intra-task variations) ---
        # Retrieve demo list from the newly created task_to_demos map
        all_demos_for_task = self.task_to_demos.get(anchor_task, [])
        potential_hard_demos = [d for d in all_demos_for_task if d != anchor_demo_num]
        
        actual_hard_samples = random.sample(
            potential_hard_demos, 
            k=min(len(potential_hard_demos), target_hard_count)
        )
        for d_num in actual_hard_samples:
            negatives.append(self.lookup_map[(anchor_task, d_num)])

        # --- 2. Easy Negatives (Inter-task variations) ---
        remaining_needed = num_negatives - len(negatives)
        other_tasks = [t for t in self.task_to_demos.keys() if t != anchor_task]
        
        if other_tasks and remaining_needed > 0:
            for _ in range(remaining_needed):
                neg_task = random.choice(other_tasks)
                neg_demo_num = random.choice(self.task_to_demos[neg_task])
                negatives.append(self.lookup_map[(neg_task, neg_demo_num)])

        return negatives

if __name__ == "__main__":
    # --- Usage ---
    dataset = RoboCasaActionDataset("/vast/users/tianyu.wang/anv_workspace/ThinkPlan/playground/RoboCasa_Data")
    action_dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=lambda b: b)

    # Direct access is now nearly instant
    sample = dataset.get_action("CoffeePressButton", 5)
    print(f"Retrieved {sample['task_name']} actions from RAM. Shape: {sample['actions'].shape}")