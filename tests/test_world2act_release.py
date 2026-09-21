"""CPU-only regression checks; run with python -m unittest discover -s tests."""

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock
import warnings

from examples.world2act_inference import parse_args, plan_generations

ROOT = Path(__file__).resolve().parents[1]


class InferencePlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "inputs"
        self.source.mkdir()

    def args(self, *extra):
        return parse_args(["--gr00t_variant", "droid", "--input_path", str(self.source),
                           "--save_path", str(self.root / "output.mp4"), *extra])

    def episode(self, name, frame="first_frame.png", prompts=None):
        directory = self.source / name
        directory.mkdir(parents=True)
        (directory / frame).touch()
        for filename, text in (prompts or {"prompt.txt": "pick the mango"}).items():
            (directory / filename).write_text(text)
        return directory

    def test_all_seeds_included_by_default(self):
        for seed in (12, 42, 999):
            self.episode(f"task/seed_{seed}/episode_0")
        jobs = plan_generations(self.args(), "robocasa")
        self.assertEqual(len(jobs), 3)
        self.assertTrue(all(str(j[0].output_path).startswith(str(self.root / "output")) for j in jobs))

    def test_disjoint_shards_cover_every_episode(self):
        for index in range(7):
            self.episode(f"task/episode_{index}")
        shards = []
        for index in range(3):
            jobs = plan_generations(self.args("--num_shards", "3", "--shard_index", str(index)), "robocasa_cp")
            shards.append({j[0].input_path for j in jobs})
        self.assertEqual(len(set.union(*shards)), 7)
        self.assertTrue(all(not shards[i] & shards[j] for i in range(3) for j in range(i)))

    def test_numeric_action_order_and_feedback_chain(self):
        self.episode("task/episode", prompts={"aa_10.txt": "close", "aa_2.txt": "pick"})
        sequence = plan_generations(self.args(), "libero")[0]
        self.assertEqual([j.prompt for j in sequence], ["pick", "close"])
        self.assertEqual(sequence[1].input_path, sequence[0].next_frame_path)
        self.assertIsNone(sequence[-1].next_frame_path)
        self.assertEqual(sequence[0].output_path.name, "output_aa_2.mp4")

    def test_full_task_prefers_prompt_file(self):
        self.episode("episode", prompts={"prompt.txt": "full task", "aa_0.txt": "atomic"})
        jobs = plan_generations(self.args("--prompt", "fallback"), "robocasa_cp")
        self.assertEqual(jobs[0][0].prompt, "full task")
        self.assertEqual(len(jobs[0]), 1)

    def test_franka_finds_nested_demo_frames(self):
        self.episode("session/pick-demo-1", frame="frame0.jpg")
        self.assertEqual(len(plan_generations(self.args(), "franka")), 1)

    def test_single_file_uses_exact_output(self):
        image = self.root / "image.png"
        image.touch()
        args = self.args("--input_path", str(image), "--prompt", "pick")
        jobs = plan_generations(args, "robocasa")
        self.assertEqual(jobs[0][0].output_path, self.root / "output.mp4")

    def test_output_cannot_be_inside_dataset(self):
        self.episode("episode")
        with self.assertRaisesRegex(ValueError, "outside"):
            plan_generations(self.args("--save_path", str(self.source / "outputs")), "libero")

    def test_missing_prompt_fails(self):
        episode = self.episode("episode")
        (episode / "prompt.txt").unlink()
        with self.assertRaisesRegex(ValueError, "prompt"):
            plan_generations(self.args(), "robocasa_cp")

    def test_empty_directory_fails(self):
        with self.assertRaisesRegex(ValueError, "No episodes"):
            plan_generations(self.args(), "libero")

    def test_directory_rejects_five_frame_conditioning(self):
        self.episode("episode")
        with self.assertRaisesRegex(ValueError, "num_conditional_frames 1"):
            plan_generations(self.args("--num_conditional_frames", "5"), "libero")


def isolated_method(path, class_name, method_name, namespace):
    """Load a method without importing unavailable GPU extensions in this CPU test."""
    tree = ast.parse((ROOT / path).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method], type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, path, "exec"), namespace)
    return namespace[method_name]


class WorldModelStepTests(unittest.TestCase):
    def run_steps(self, overflow=False):
        method = isolated_method("imaginaire/trainer.py", "ImaginaireTrainer", "training_step", {
            "distributed": SimpleNamespace(ddp_sync_grad=lambda *_: nullcontext())
        })
        trainer = SimpleNamespace(
            config=SimpleNamespace(trainer=SimpleNamespace(grad_accum_iter=2, distributed_parallelism="fsdp")),
            callbacks=Mock(), training_timer=lambda _: nullcontext(),
        )
        backward = Mock()

        class Loss:
            def __truediv__(self, divisor):
                self.divisor = divisor
                return self

            def backward(self):
                backward(self.divisor)

        loss = Loss()
        model = Mock()
        model.training_step.return_value = ({"prediction": "world model"}, loss)
        optimizer, scheduler = Mock(), Mock()
        scaler = Mock()
        scaler.scale.side_effect = lambda value: value
        scaler.get_scale.side_effect = [2.0, 1.0 if overflow else 2.0]
        _, returned_loss, count = method(trainer, model, optimizer, scheduler, scaler, {}, grad_accum_iter=0)
        self.assertIs(returned_loss, loss)
        self.assertEqual(count, 1)
        scaler.step.assert_not_called()
        _, _, count = method(trainer, model, optimizer, scheduler, scaler, {}, grad_accum_iter=count)
        self.assertEqual(count, 0)
        self.assertEqual(backward.call_count, 2)
        backward.assert_called_with(2)
        scaler.unscale_.assert_called_once_with(optimizer)
        scaler.step.assert_called_once_with(optimizer)
        optimizer.zero_grad.assert_called_once_with(set_to_none=True)
        model.eval.assert_not_called()
        self.assertEqual(scheduler.step.call_count, 0 if overflow else 1)

    def test_real_loss_accumulates_before_optimizer_step(self):
        self.run_steps()

    def test_scheduler_skips_amp_overflow(self):
        self.run_steps(overflow=True)


class ActionVAETrainingTests(unittest.TestCase):
    def test_stage1_freezes_world_model_and_checkpoints_action_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            optimizer = Mock()
            torch = SimpleNamespace(optim=SimpleNamespace(AdamW=Mock(return_value=optimizer)), no_grad=nullcontext)
            method = isolated_method("imaginaire/action_pretraining/trainer.py", "ActionVAETrainer", "train", {
                "torch": torch, "Path": Path,
                "distributed": SimpleNamespace(get_world_size=lambda: 1), "log": Mock(),
                "misc": SimpleNamespace(to=lambda value, **_: value),
            })
            trainer = SimpleNamespace(config=SimpleNamespace(trainer=SimpleNamespace(max_iter=3, memory_format=None)))
            trainer._get_action_data_batch = Mock(return_value=(SimpleNamespace(shape=(1, 128, 7)), "negatives"))
            def update(*args):
                trainer._action_counter += 1
                trainer.action_optimizer.step()
            trainer._train_action_model = update
            saved = []
            trainer._save_action_weight = lambda model: saved.append(trainer._action_counter)
            world = Mock()
            world.to.return_value = world
            world.training_step.return_value = ({"model_pred": SimpleNamespace(x0=Mock())}, None)
            action = Mock()
            action.float.return_value.to.return_value = action
            batch = {"video": SimpleNamespace(shape=(1, 3, 93, 480, 832))}
            method(trainer, world, [batch], None, ["episode"], action,
                   dataset_type="simpler_env", output_dir=directory, save_iter=2)
            world.requires_grad_.assert_called_once_with(False)
            world.eval.assert_called_once()
            world.init_optimizer_scheduler.assert_not_called()
            self.assertEqual(optimizer.step.call_count, 3)
            self.assertEqual(saved, [2, 3])


class VideoWindowTests(unittest.TestCase):
    def sample(self, frame_count):
        indices = []
        reader = Mock()
        reader.get_batch.side_effect = lambda selected: (indices.extend(selected) or SimpleNamespace(asnumpy=lambda: "frames"))
        reader.get_avg_fps.return_value = 16

        class Reader:
            def __len__(self):
                return frame_count

            get_batch = reader.get_batch
            get_avg_fps = reader.get_avg_fps
            seek = reader.seek

        numpy = SimpleNamespace(
            random=SimpleNamespace(randint=lambda low, high: high - 1),
            arange=lambda start, end: SimpleNamespace(tolist=lambda: list(range(start, end))),
        )
        method = isolated_method("cosmos_predict2/data/dataset_video.py", "Dataset", "_load_video", {
            "np": numpy, "warnings": warnings, "cpu": lambda _: None, "VideoReader": lambda *_, **__: Reader(),
        })
        result = method(SimpleNamespace(sequence_length=93), "sample.mp4")
        self.assertEqual(result, ("frames", 16))
        return indices

    def test_exact_length_video_is_valid(self):
        self.assertEqual(self.sample(93), list(range(93)))

    def test_last_possible_window_is_reachable(self):
        self.assertEqual(self.sample(94), list(range(1, 94)))


if __name__ == "__main__":
    unittest.main()
