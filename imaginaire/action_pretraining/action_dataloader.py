import torch
import h5py
import random
import json
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
import re
import pandas as pd

class RoboCasaActionDataset(Dataset):
    def __init__(self, root_dir, metadata_path="/vast/users/tianyu.wang/anv_workspace/ThinkPlan/playground/playground/data/RoboCasa_mg/train/video_metadata.json"):
        self.root_path = Path(root_dir)
        self.metadata_path = metadata_path
        self.episode_index = []
        self.lookup_map = {}
        self.task_to_demos = defaultdict(list)
        
        # Load Metadata
        self._load_metadata()
        self._build_and_load_index()

    def _load_metadata(self):
        print(f"Loading metadata from {self.metadata_path}...")
        try:
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
            self.metadata = {}

    def _build_and_load_index(self):
        """Scans directory and loads ACTIONS into memory."""
        all_hdf5_files = list(self.root_path.rglob("demo_gentex_im128_randcams.hdf5"))
        mg_files = [p for p in all_hdf5_files if "mg" in p.parts]
        
        for h5_path in tqdm(mg_files):
            try:
                with h5py.File(h5_path, 'r') as f:
                    if "data" not in f: continue
                    
                    task_name = h5_path.parts[-4]
                    for demo_key in f["data"].keys():
                        demo_num = int(demo_key.split("_")[-1])
                        
                        # Optimization: Skip indexing if this demo isn't in our metadata JSON at all
                        # (Optional: remove this check if you want to allow non-metadata access for other purposes)
                        if task_name not in self.metadata or str(demo_num) not in self.metadata[task_name]:
                            continue

                        actions = torch.from_numpy(f["data"][demo_key]["actions"][:]).float()
                        
                        entry = {
                            "demo_num": demo_num,
                            "task_name": task_name,
                            "actions": actions
                        }
                        
                        self.episode_index.append(entry)
                        self.lookup_map[(task_name, demo_num)] = entry
                        self.task_to_demos[task_name].append(demo_num)
                        
            except Exception as e:
                print(f"Error indexing {h5_path}: {e}")

    def __len__(self):
        return len(self.episode_index)

    def __getitem__(self, idx):
        return self.episode_index[idx]

    def get_action(self, task_name, demo_num, aa_idx):
        """
        Retrieves atomic action, performs a Random Consecutive Crop (or pads) to EXACTLY 128 frames.
        """
        key = (task_name, demo_num)
        
        # 1. Retrieve full actions
        if key not in self.lookup_map:
            raise KeyError(f"Task '{task_name}' Demo {demo_num} not found (or filtered out).")
        
        full_actions = self.lookup_map[key]["actions"]

        # 2. Retrieve start/end from JSON
        try:
            meta_entry = self.metadata[task_name][str(demo_num)][str(aa_idx)]
            start = meta_entry["start"]
            end = meta_entry["end"]
        except KeyError:
             raise KeyError(f"Atomic Action {aa_idx} not found for {task_name} demo {demo_num}")

        # 3. Slice (end is inclusive in your JSON logic, so +1 for python slice)
        action_segment = full_actions[start : end + 1]

        # 4. Random Crop or Pad to 128
        seq_len = action_segment.shape[0]
        max_len = 128

        if seq_len >= max_len:
            # Random Consecutive Crop
            # Pick a random start index such that we fit exactly max_len frames
            diff = seq_len - max_len
            start_idx = torch.randint(0, diff + 1, (1,)).item()
            action_segment = action_segment[start_idx : start_idx + max_len]
            
        elif seq_len < max_len:
            # Pad (Repeat last frame)
            padding_needed = max_len - seq_len
            last_frame = action_segment[-1].unsqueeze(0)
            padding = last_frame.repeat(padding_needed, 1)
            action_segment = torch.cat([action_segment, padding], dim=0)

        return action_segment
    
    def sample_negative_actions(self, anchor_task, anchor_demo_num, num_negatives=4, hard_ratio=0.5):
        """
        Safely samples negatives, skipping demos missing from metadata.
        """
        negatives = []
        target_hard_count = int(num_negatives * hard_ratio)

        # --- Helper to safely get a negative ---
        def try_get_negative(task, demo_id):
            # Check if this specific demo has metadata for aa_idx=0
            if task in self.metadata and str(demo_id) in self.metadata[task]:
                if "0" in self.metadata[task][str(demo_id)]:
                    return self.get_action(task, demo_id, aa_idx=0)
            return None

        # --- 1. Hard Negatives (Same Task) ---
        all_demos = self.task_to_demos.get(anchor_task, [])
        # Filter out anchor and shuffle
        candidates = [d for d in all_demos if d != anchor_demo_num]
        random.shuffle(candidates)
        
        for d_num in candidates:
            if len(negatives) >= target_hard_count: break
            
            neg_tensor = try_get_negative(anchor_task, d_num)
            if neg_tensor is not None:
                negatives.append(neg_tensor)

        # --- 2. Easy Negatives (Different Task) ---
        other_tasks = list(self.task_to_demos.keys())
        if anchor_task in other_tasks: other_tasks.remove(anchor_task)
        
        while len(negatives) < num_negatives and other_tasks:
            rand_task = random.choice(other_tasks)
            if not self.task_to_demos[rand_task]: continue
            
            rand_demo = random.choice(self.task_to_demos[rand_task])
            
            neg_tensor = try_get_negative(rand_task, rand_demo)
            if neg_tensor is not None:
                negatives.append(neg_tensor)
                
        # Fill remaining with duplicates of existing negatives if we ran out of valid candidates
        # (Edge case safety)
        while len(negatives) < num_negatives:
            if negatives:
                negatives.append(negatives[0].clone())
            else:
                # Emergency fallback if NO negatives found (unlikely)
                negatives.append(torch.zeros((128, 12)))

        return negatives[:num_negatives]


class LIBEROActionDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): Path to LIBERO datasets (e.g., /.../LIBERO/datasets)
        """
        self.root_path = Path(root_dir)
        self.episode_index = []
        self.lookup_map = {}
        self.task_to_demos = defaultdict(list)
        
        # Regex to strip scene prefixes like "KITCHEN_SCENE3_" or "LIVING_ROOM_SCENE1_"
        # It looks for uppercase words + underscore + SCENE + digits + underscore at the start.
        self.scene_prefix_pattern = re.compile(r'^[A-Z_]+_SCENE\d+_')

        self._build_and_load_index()

    def _get_task_name_from_path(self, h5_path):
        """
        Extracts the clean task name from the filename.
        Ex: KITCHEN_SCENE3_turn_on_stove_demo.hdf5 -> turn_on_stove
        """
        filename = h5_path.stem # remove .hdf5
        
        # 1. Strip scene prefix (e.g., KITCHEN_SCENE3_)
        clean_name = self.scene_prefix_pattern.sub('', filename)
        
        # 2. FIX: Strip "_demo" suffix if it exists
        if clean_name.endswith("_demo"):
            clean_name = clean_name[:-5]  # remove last 5 chars
            
        return clean_name

    def _build_and_load_index(self):
        """Scans directory and loads ACTIONS into memory."""
        print(f"Scanning {self.root_path} for .hdf5 files...")
        
        # LIBERO usually organizes files directly in suite folders or root
        all_hdf5_files = list(self.root_path.rglob("*.hdf5"))
        
        print(f"Found {len(all_hdf5_files)} HDF5 files. Loading indices...")
        
        for h5_path in tqdm(all_hdf5_files):
            try:
                # Extract task name from filename
                task_name = self._get_task_name_from_path(h5_path)
                
                with h5py.File(h5_path, 'r') as f:
                    if "data" not in f: 
                        continue
                    
                    # Iterate through demos (demo_0, demo_1, ...)
                    for demo_key in f["data"].keys():
                        if not demo_key.startswith("demo_"):
                            continue
                            
                        demo_num = int(demo_key.split("_")[1])
                        
                        # Load actions (T, 7)
                        # Note: actions are numpy arrays, convert to FloatTensor
                        actions_np = f["data"][demo_key]["actions"][:]
                        actions = torch.from_numpy(actions_np).float()
                        
                        entry = {
                            "demo_num": demo_num,
                            "task_name": task_name,
                            "actions": actions,
                            "original_len": actions.shape[0]
                        }
                        
                        self.episode_index.append(entry)
                        self.lookup_map[(task_name, demo_num)] = entry
                        self.task_to_demos[task_name].append(demo_num)
                        
            except Exception as e:
                print(f"Error indexing {h5_path}: {e}")
                
        print(f"Loaded {len(self.episode_index)} total episodes across {len(self.task_to_demos)} tasks.")

    def __len__(self):
        return len(self.episode_index)

    def __getitem__(self, idx):
        return self.episode_index[idx]

    def get_action(self, task_name, demo_num, aa_idx=None):
        """
        Retrieves action sequence for a specific task and demo based on a segment index.
        
        Args:
            task_name (str): The clean task name
            demo_num (int): The demo index
            aa_idx (int, optional): The segment index (1-based). 
                                    Gets [128*(idx-1) : 128*idx].
                                    If end exceeds length, returns the last 128 frames.
        """
        key = (task_name, demo_num)
        if key not in self.lookup_map:
            print(f"Task '{task_name}' Demo {demo_num} not found.")
            return None
        
        action_segment = self.lookup_map[key]["actions"]
        seq_len = action_segment.shape[0]
        window_size = 128

        if aa_idx is not None:
            # Calculate the requested bounds
            start_idx = window_size * (aa_idx - 1)
            end_idx = window_size * aa_idx

            # If the requested window exceeds the sequence length, 
            # shift the window to capture the final 128 frames.
            if end_idx > seq_len:
                start_idx = max(0, seq_len - window_size)
                end_idx = seq_len
            
            action_segment = action_segment[start_idx : end_idx]

        return action_segment
    
    def sample_negative_actions(self, anchor_task, anchor_demo_num, hard_ratio=0.75, num_negatives=4):
        """
        Samples negatives:
        - Hard negatives: Same task, different demo
        - Easy negatives: Different task
        """
        negatives = []
        
        # --- 1. Hard Negatives (Same Task) ---
        all_demos = self.task_to_demos.get(anchor_task, [])
        candidates = [d for d in all_demos if d != anchor_demo_num]
        random.shuffle(candidates)
        
        # Take up to half as hard negatives
        target_hard = num_negatives // 2
        for d_num in candidates[:target_hard]:
            negatives.append(self.get_action(anchor_task, d_num))

        # --- 2. Easy Negatives (Different Task) ---
        other_tasks = list(self.task_to_demos.keys())
        if anchor_task in other_tasks: other_tasks.remove(anchor_task)
        
        while len(negatives) < num_negatives and other_tasks:
            rand_task = random.choice(other_tasks)
            if not self.task_to_demos[rand_task]: continue
            rand_demo = random.choice(self.task_to_demos[rand_task])
            
            negatives.append(self.get_action(rand_task, rand_demo))

        return negatives[:num_negatives]

class FrankaArmActionDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): Path to the folder containing the Franka .hdf5 files.
        """
        self.root_path = Path(root_dir)
        self.episode_index = []
        self.lookup_map = {}
        self.task_to_demos = defaultdict(list)
        
        # Refined mapping: Underscores instead of spaces to match video filenames later
        self.filename_to_task = {
            "closedoor": "close_the_cabinet_door",
            "pickbowl": "pick_the_bowl",
            "pickcup": "pick_the_cup",
            "placecup": "place_the_cup_on_the_cabinet",
            "placebowl": "place_the_bowl_on_the_plate",
            "placeonplate": "place_the_bowl_on_the_plate" 
        }

        self._build_and_load_index()

    def _get_task_name_from_path(self, h5_path):
        """
        Extracts the specific task description using the lookup map.
        """
        filename_stem = h5_path.stem.lower() # remove .hdf5 and ensure lowercase
        
        # Look up the task string, fallback to the filename if not explicitly mapped
        task_string = self.filename_to_task.get(filename_stem, filename_stem)
        return task_string

    def _build_and_load_index(self):
        """Scans directory and loads actions into memory."""
        print(f"Scanning {self.root_path} for .hdf5 files...")
        
        all_hdf5_files = list(self.root_path.rglob("*.hdf5"))
        print(f"Found {len(all_hdf5_files)} HDF5 files. Loading indices...")
        
        for h5_path in tqdm(all_hdf5_files):
            try:
                # Get the natural language task description
                task_name = self._get_task_name_from_path(h5_path)
                
                with h5py.File(h5_path, 'r') as f:
                    if "data" not in f: 
                        print(f"Skipping {h5_path}: No 'data' group found.")
                        continue
                    
                    # Iterate through demos (demo_0, demo_1, demo_9...)
                    for demo_key in f["data"].keys():
                        if not demo_key.startswith("demo_"):
                            continue
                            
                        demo_num = int(demo_key.split("_")[1])
                        
                        # Load actions (seq, 7) of dtype float32
                        actions_np = f["data"][demo_key]["actions"][:]
                        actions = torch.from_numpy(actions_np).float()
                        
                        entry = {
                            "demo_num": demo_num,
                            "task_name": task_name,
                            "actions": actions,
                            "original_len": actions.shape[0]
                        }
                        
                        self.episode_index.append(entry)
                        self.lookup_map[(task_name, demo_num)] = entry
                        self.task_to_demos[task_name].append(demo_num)
                        
            except Exception as e:
                print(f"Error indexing {h5_path}: {e}")
                
        print(f"Loaded {len(self.episode_index)} total episodes across {len(self.task_to_demos)} tasks.")

    def __len__(self):
        return len(self.episode_index)

    def __getitem__(self, idx):
        return self.episode_index[idx]

    def get_action(self, task_name, demo_num, aa_idx=None):
        """
        Retrieves action sequence for a specific task and demo.
        If aa_idx is provided, extracts a 128-frame segment.
        If the remaining sequence is shorter than 128, pads the end with 0s.
        """
        key = (task_name, demo_num)
        if key not in self.lookup_map:
            print(f"Task '{task_name}' Demo {demo_num} not found.")
            return None
        
        action_segment = self.lookup_map[key]["actions"]
        seq_len = action_segment.shape[0]
        action_dim = action_segment.shape[1] # This is 7 for your Franka setup
        window_size = 128

        if aa_idx is not None:
            # Calculate the requested bounds
            start_idx = window_size * (aa_idx - 1)
            end_idx = window_size * aa_idx

            # Edge case: If the segment requested is entirely past the end of the data
            if start_idx >= seq_len:
                return torch.zeros((window_size, action_dim), dtype=action_segment.dtype)

            # Slice whatever is actually available
            actual_end_idx = min(end_idx, seq_len)
            sliced_actions = action_segment[start_idx : actual_end_idx]

            # If the sliced segment is shorter than 128, pad it
            current_len = sliced_actions.shape[0]
            if current_len < window_size:
                # Create a zero tensor of shape (128, 7)
                padded_actions = torch.zeros((window_size, action_dim), dtype=sliced_actions.dtype)
                
                # Drop the actual actions into the beginning of the tensor
                padded_actions[:current_len, :] = sliced_actions
                
                action_segment = padded_actions
            else:
                action_segment = sliced_actions

        return action_segment
    
    def sample_negative_actions(self, anchor_task, anchor_demo_num, hard_ratio=0.75, num_negatives=4):
        """
        Samples negatives for contrastive learning/loss:
        - Hard negatives: Same task, different demo
        - Easy negatives: Different task
        """
        negatives = []
        
        # --- 1. Hard Negatives (Same Task) ---
        all_demos = self.task_to_demos.get(anchor_task, [])
        candidates = [d for d in all_demos if d != anchor_demo_num]
        random.shuffle(candidates)
        
        # Take up to half as hard negatives
        target_hard = num_negatives // 2
        for d_num in candidates[:target_hard]:
            negatives.append(self.get_action(anchor_task, d_num))

        # --- 2. Easy Negatives (Different Task) ---
        other_tasks = list(self.task_to_demos.keys())
        if anchor_task in other_tasks: 
            other_tasks.remove(anchor_task)
        
        while len(negatives) < num_negatives and other_tasks:
            rand_task = random.choice(other_tasks)
            if not self.task_to_demos[rand_task]: 
                other_tasks.remove(rand_task)
                continue
            rand_demo = random.choice(self.task_to_demos[rand_task])
            
            negatives.append(self.get_action(rand_task, rand_demo))

        return negatives[:num_negatives]


class SimplerEnvActionDataset(Dataset):
    def __init__(self, root_dir="/vast/users/tianyu.wang/tuanvvv_workspace/GR00T-Dreams/cosmos-predict2/datasets/benchmark_train/gr1"):
        self.root_path = Path(root_dir)
        if (self.root_path / "data").is_dir():
            self.root_path = self.root_path / "data"

        self.episode_index = []
        self.lookup_map = {}
        self.task_to_demos = defaultdict(list)
        self.cache = {}
        self.window_size = 128
        self.action_dim = 7

        self.episode_to_task = self._load_episode_tasks()
        self._build_and_load_index()

    def _load_episode_tasks(self):
        task_map = {}
        meta_path = self.root_path.parent / "meta" / "episodes.jsonl"
        if not meta_path.exists():
            return task_map

        with open(meta_path, "r") as f:
            for line in f:
                x = json.loads(line)
                eid = x.get("episode_index", x.get("episode_id", None))
                task = x.get("task", x.get("task_name", x.get("tasks", x.get("task_index", None))))
                if isinstance(task, list):
                    task = task[0] if task else "unknown"
                if eid is not None and task is not None:
                    task_map[int(eid)] = str(task)

        return task_map

    def _episode_id(self, x):
        if isinstance(x, int):
            return x
        m = re.search(r"episode_(\d+)", str(x))
        return int(m.group(1)) if m else int(x)

    def _parquet_path(self, episode_id):
        chunk_id = episode_id // 1000
        return self.root_path / f"chunk-{chunk_id:03d}" / f"episode_{episode_id:06d}.parquet"

    def _build_and_load_index(self):
        files = sorted(self.root_path.rglob("episode_*.parquet"))
        print(f"Found {len(files)} SimplerEnv parquet files.")

        for p in tqdm(files):
            eid = self._episode_id(p.name)
            task = self.episode_to_task.get(eid, f"chunk-{eid // 1000:03d}")

            entry = {
                "episode_id": eid,
                "demo_num": eid,
                "task_name": task,
                "path": p,
            }

            self.episode_index.append(entry)
            self.lookup_map[eid] = entry
            self.lookup_map[(task, eid)] = entry
            self.task_to_demos[task].append(eid)

        print(f"Indexed {len(self.episode_index)} SimplerEnv episodes.")

    def __len__(self):
        return len(self.episode_index)

    def __getitem__(self, idx):
        return self.episode_index[idx]

    def _read_actions(self, episode_id):
        episode_id = self._episode_id(episode_id)

        if episode_id in self.cache:
            return self.cache[episode_id]

        entry = self.lookup_map.get(episode_id)
        path = entry["path"] if entry is not None else self._parquet_path(episode_id)

        if not path.exists():
            print(f"Missing parquet: {path}")
            return None

        df = pd.read_parquet(path)

        for col in ["action", "actions"]:
            if col in df.columns:
                first = df[col].iloc[0]
                if hasattr(first, "__len__") and not isinstance(first, (str, bytes)):
                    actions = torch.tensor(df[col].tolist(), dtype=torch.float32)
                    actions = actions[:, :self.action_dim]
                    self.cache[episode_id] = actions
                    return actions

        action_cols = [c for c in df.columns if "action" in c.lower()]
        action_cols = sorted(action_cols, key=lambda c: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", c)])

        if len(action_cols) < self.action_dim:
            raise KeyError(f"No valid 7D action found in {path}. Columns: {df.columns.tolist()}")

        actions = torch.tensor(df[action_cols[:self.action_dim]].values, dtype=torch.float32)
        self.cache[episode_id] = actions
        return actions

    def get_action(self, task_name, demo_num=None, aa_idx=None):
        episode_id = self._episode_id(task_name if demo_num is None else demo_num)
        actions = self._read_actions(episode_id)

        if actions is None:
            return None

        if aa_idx is None:
            return actions

        start = int(aa_idx) * self.window_size
        end = start + self.window_size

        if start >= actions.shape[0]:
            start = max(0, actions.shape[0] - self.window_size)
            end = actions.shape[0]

        return actions[start:min(end, actions.shape[0])]

    def sample_negative_actions(self, anchor_task, anchor_demo_num=None, hard_ratio=0.75, num_negatives=4):
        anchor_eid = self._episode_id(anchor_task if anchor_demo_num is None else anchor_demo_num)
        entry = self.lookup_map.get(anchor_eid)
        anchor_task = entry["task_name"] if entry is not None else f"chunk-{anchor_eid // 1000:03d}"

        negatives = []
        target_hard = int(num_negatives * hard_ratio)

        hard = [e for e in self.task_to_demos.get(anchor_task, []) if e != anchor_eid]
        random.shuffle(hard)

        for eid in hard[:target_hard]:
            x = self.get_action(eid)
            if x is not None:
                negatives.append(x)

        other_tasks = [t for t in self.task_to_demos.keys() if t != anchor_task]

        tries = 0
        while len(negatives) < num_negatives and other_tasks and tries < num_negatives * 20:
            tries += 1
            task = random.choice(other_tasks)
            eid = random.choice(self.task_to_demos[task])
            x = self.get_action(eid)
            if x is not None:
                negatives.append(x)

        all_eps = [e["episode_id"] for e in self.episode_index if e["episode_id"] != anchor_eid]
        while len(negatives) < num_negatives and all_eps:
            x = self.get_action(random.choice(all_eps))
            if x is not None:
                negatives.append(x)

        while len(negatives) < num_negatives:
            negatives.append(torch.zeros((self.window_size, self.action_dim)))

        return negatives[:num_negatives]


if __name__ == "__main__":
    # --- Usage ---
    # Update this path to your actual path
    # data_path = "/vast/users/tianyu.wang/tuanvvv_workspace/GR00T-Dreams/libero_finetuning/LIBERO/datasets"
    
    # print("Initializing LIBERO Dataset...")
    # dataset = LIBEROActionDataset(root_dir=data_path)

    # # --- Test 1: Check a specific task name parsing ---
    # # We try to grab a task we know exists based on your description
    # # Example: 'turn_on_the_stove_and_put_the_moka_pot_on_it_demo'
    
    # # test_task = "turn_on_the_stove_and_put_the_moka_pot_on_it_demo"
    # test_task = "put_the_bowl_on_the_plate"
    # demo_num = 3
    # dataset.get_action(test_task, demo_num, 0)
    # if test_task in dataset.task_to_demos:
    #     print(f"\nFound task: {test_task}")
    #     demos = dataset.task_to_demos[test_task]
    #     print(f"Available demo IDs: {demos[:5]}...")
        
    #     # Get Action
    #     action = dataset.get_action(test_task, demos[0])
    #     print(f"Retrieved action shape for demo {demos[0]}: {action.shape}") # Should be (T, 7)
        
    #     # Test Negatives
    #     negs = dataset.sample_negative_actions(test_task, demos[0], num_negatives=2)
    #     print(f"Sampled {len(negs)} negatives.")
    # else:
    #     print(f"\nCould not find specific test task '{test_task}'.")
    #     print("Printing 5 random tasks found instead:")
    #     print(list(dataset.task_to_demos.keys())[:5])

    data_path = "/vast/users/tianyu.wang/tuanvvv_workspace/GR00T-Dreams/real-world-WM/Isaac-GR00T/real-world"
    dataset = FrankaArmActionDataset(data_path)
    action_segment = dataset.get_action("place_the_bowl_on_the_plate", 5)
    print(action_segment.shape)
