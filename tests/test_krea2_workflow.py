from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tomllib
import types
import unittest
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(
    os.environ.get("KREA2_SCRIPTS_DIR", REPOSITORY_ROOT / "src" / "scripts")
)
TEMPLATES_DIR = SCRIPTS_DIR / "krea2" / "templates"


def load_trigger_words_module():
    spec = importlib.util.spec_from_file_location(
        "trigger_words_under_test", SCRIPTS_DIR / "trigger_words.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load trigger_words.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_glob_images(image_directory: str) -> list[str]:
    supported = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
    return sorted(
        str(path)
        for path in Path(image_directory).iterdir()
        if path.is_file() and path.suffix.lower() in supported
    )


class FakeTomlModule(types.ModuleType):
    def load(self, path: str):
        with Path(path).open("rb") as handle:
            return tomllib.load(handle)


class Krea2TemplateTests(unittest.TestCase):
    def load_template(self, name: str) -> dict[str, object]:
        with (TEMPLATES_DIR / name).open("rb") as handle:
            return tomllib.load(handle)

    def test_32gb_checkpoint_search_presets(self):
        expected = {
            "train.toml": {
                "steps": 8000,
                "save": 400,
                "sample": 400,
                "state": 800,
                "rank": 64,
                "alpha": 64,
                "lr": 0.00005,
                "has_network_args": True,
            },
            "train-baseline.toml": {
                "steps": 4000,
                "save": 200,
                "sample": 200,
                "state": 400,
                "rank": 32,
                "alpha": 32,
                "lr": 0.0001,
                "has_network_args": False,
            },
            "train-quality.toml": {
                "steps": 6000,
                "save": 300,
                "sample": 300,
                "state": 600,
                "rank": 64,
                "alpha": 64,
                "lr": 0.00007,
                "has_network_args": False,
            },
        }

        for filename, values in expected.items():
            with self.subTest(filename=filename):
                config = self.load_template(filename)
                self.assertEqual(values["steps"], config["max_train_steps"])
                self.assertEqual(values["save"], config["save_every_n_steps"])
                self.assertEqual(values["sample"], config["sample_every_n_steps"])
                self.assertEqual(
                    values["state"], config["save_last_n_steps_state"]
                )
                self.assertEqual(values["rank"], config["network_dim"])
                self.assertEqual(values["alpha"], config["network_alpha"])
                self.assertEqual(values["lr"], config["learning_rate"])
                self.assertNotIn("save_last_n_steps", config)
                self.assertEqual(
                    values["has_network_args"], "network_args" in config
                )
                self.assertEqual(
                    20, config["max_train_steps"] // config["save_every_n_steps"]
                )
                self.assertEqual(
                    0, config["max_train_steps"] % config["save_every_n_steps"]
                )

    def test_10gb_retention_is_unchanged(self):
        config = self.load_template("train-10gb.toml")
        self.assertEqual(800, config["max_train_steps"])
        self.assertEqual(100, config["save_every_n_steps"])
        self.assertEqual(900, config["save_last_n_steps"])
        self.assertNotIn("save_last_n_steps_state", config)

    def test_selectors_expose_only_the_new_preset_names(self):
        for filename in (
            "prepare-krea2-character.sh",
            "train-krea2-character.sh",
        ):
            script = (SCRIPTS_DIR / "krea2" / filename).read_text(encoding="utf-8")
            self.assertIn("default|baseline|quality|10gb", script)
            self.assertNotIn("default|quality|attention|10gb", script)

        self.assertFalse((TEMPLATES_DIR / "train-attention.toml").exists())
        self.assertTrue((TEMPLATES_DIR / "train-baseline.toml").is_file())


@unittest.skipUnless(
    importlib.util.find_spec("musubi_tuner"),
    "Musubi Tuner is required for parser integration tests",
)
class MusubiParserIntegrationTests(unittest.TestCase):
    def test_current_musubi_parser_loads_every_bundled_training_template(self):
        from musubi_tuner.krea2_train_network import krea2_setup_parser
        from musubi_tuner.training.parser_common import (
            read_config_from_file,
            setup_parser_common,
        )

        for filename in (
            "train.toml",
            "train-baseline.toml",
            "train-quality.toml",
            "train-10gb.toml",
        ):
            with self.subTest(filename=filename):
                parser = krea2_setup_parser(setup_parser_common())
                argv = [
                    "krea2_train_network.py",
                    "--config_file",
                    str(TEMPLATES_DIR / filename),
                ]
                with (
                    mock.patch.object(sys, "argv", argv),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    args = parser.parse_args()
                    args = read_config_from_file(args, parser)
                self.assertIsNotNone(args.max_train_steps)
                self.assertIsNotNone(args.save_every_n_steps)
                self.assertIsNotNone(args.gradient_accumulation_steps)
                self.assertIsNotNone(args.network_dim)
                self.assertIsNotNone(args.network_alpha)
                self.assertIsNotNone(args.learning_rate)

    def test_real_musubi_image_discovery_drives_structured_inspection(self):
        trigger_words = load_trigger_words_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            images.mkdir()
            for index in range(3):
                stem = f"{index + 1:03d}"
                (images / f"{stem}.png").touch()
                (images / f"{stem}.txt").write_text(
                    "A person. k2v9\n", encoding="utf-8"
                )
            config = root / "dataset.toml"
            config.write_text(
                "\n".join(
                    [
                        "[general]",
                        'caption_extension = ".txt"',
                        "batch_size = 1",
                        "",
                        "[[datasets]]",
                        f"image_directory = {json.dumps(str(images))}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            inspection = trigger_words.inspect_dataset(config)

        self.assertEqual("standard-directory", inspection["layout"])
        self.assertEqual(3, inspection["primary_image_count"])
        self.assertTrue(inspection["estimate_authoritative"])


class DatasetInspectionTests(unittest.TestCase):
    def setUp(self):
        self.trigger_words = load_trigger_words_module()
        toml_module = FakeTomlModule("toml")
        media_utils = types.ModuleType("musubi_tuner.dataset.media_utils")
        media_utils.glob_images = fake_glob_images
        dataset_module = types.ModuleType("musubi_tuner.dataset")
        dataset_module.media_utils = media_utils
        musubi_module = types.ModuleType("musubi_tuner")
        musubi_module.dataset = dataset_module
        self.modules = {
            "toml": toml_module,
            "musubi_tuner": musubi_module,
            "musubi_tuner.dataset": dataset_module,
            "musubi_tuner.dataset.media_utils": media_utils,
        }

    def write_config(self, directory: Path, contents: str) -> Path:
        path = directory / "dataset.toml"
        path.write_text(contents, encoding="utf-8")
        return path

    def add_images(
        self, directory: Path, count: int, *, captions: bool = True
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            stem = f"{index + 1:03d}"
            (directory / f"{stem}.png").touch()
            if captions:
                (directory / f"{stem}.txt").write_text(
                    "A person. k2v9\n", encoding="utf-8"
                )

    def inspect(self, config_path: Path) -> dict[str, object]:
        with mock.patch.dict(sys.modules, self.modules):
            return self.trigger_words.inspect_dataset(config_path)

    def test_standard_directory_is_authoritative(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            self.add_images(images, 30)
            config = self.write_config(
                root,
                f"""
[general]
caption_extension = ".txt"
batch_size = 1

[[datasets]]
image_directory = {json.dumps(str(images))}
num_repeats = 1
""",
            )

            inspection = self.inspect(config)

        self.assertEqual("standard-directory", inspection["layout"])
        self.assertEqual(30, inspection["primary_image_count"])
        self.assertEqual(0, inspection["additional_dataset_count"])
        self.assertEqual(1, inspection["per_device_batch_size"])
        self.assertTrue(inspection["caption_pairs_complete"])
        self.assertTrue(inspection["estimate_authoritative"])
        self.assertIsNone(inspection["estimate_unavailable_reason"])

    def test_inspect_subcommand_emits_structured_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            self.add_images(images, 2)
            config = self.write_config(
                root,
                f"""
[general]
caption_extension = ".txt"
batch_size = 1

[[datasets]]
image_directory = {json.dumps(str(images))}
""",
            )
            output = io.StringIO()
            with (
                mock.patch.dict(sys.modules, self.modules),
                contextlib.redirect_stdout(output),
            ):
                result = self.trigger_words.main(
                    ["inspect", "--dataset-config", str(config)]
                )

        self.assertEqual(0, result)
        inspection = json.loads(output.getvalue())
        self.assertEqual("standard-directory", inspection["layout"])
        self.assertEqual(2, inspection["primary_image_count"])
        self.assertTrue(inspection["estimate_authoritative"])

    def test_standard_directory_rejects_missing_caption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "images"
            self.add_images(images, 1, captions=False)
            config = self.write_config(
                root,
                f"""
[general]
caption_extension = ".txt"
batch_size = 1

[[datasets]]
image_directory = {json.dumps(str(images))}
""",
            )

            with self.assertRaisesRegex(
                self.trigger_words.TriggerWordsError, "Missing caption"
            ):
                self.inspect(config)

    def test_multi_dataset_is_unavailable_without_caption_enforcement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            primary = root / "primary"
            regularization = root / "regularization"
            self.add_images(primary, 30, captions=False)
            self.add_images(regularization, 5, captions=False)
            config = self.write_config(
                root,
                f"""
[general]
caption_extension = ".txt"
batch_size = 1

[[datasets]]
image_directory = {json.dumps(str(primary))}

[[datasets]]
image_directory = {json.dumps(str(regularization))}
""",
            )

            inspection = self.inspect(config)

        self.assertEqual("multi-dataset", inspection["layout"])
        self.assertEqual(30, inspection["primary_image_count"])
        self.assertEqual(1, inspection["additional_dataset_count"])
        self.assertFalse(inspection["estimate_authoritative"])
        self.assertEqual(
            "multi-dataset configuration",
            inspection["estimate_unavailable_reason"],
        )

    def test_jsonl_primary_is_unavailable_without_caption_enforcement(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.write_config(
                root,
                f"""
[general]
batch_size = 1

[[datasets]]
image_jsonl_file = {json.dumps(str(root / "images.jsonl"))}
""",
            )

            inspection = self.inspect(config)

        self.assertEqual("jsonl", inspection["layout"])
        self.assertIsNone(inspection["primary_image_count"])
        self.assertEqual(1, inspection["per_device_batch_size"])
        self.assertFalse(inspection["estimate_authoritative"])
        self.assertEqual(
            "primary dataset uses image_jsonl_file",
            inspection["estimate_unavailable_reason"],
        )


class LauncherIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        git_bash = Path(r"C:\Program Files\Git\bin\bash.exe")
        cls.bash = str(git_bash) if git_bash.is_file() else shutil.which("bash")

    def write_fake_runtime(self, root: Path) -> tuple[Path, Path]:
        source_root = root / "src"
        package = source_root / "musubi_tuner"
        training_package = package / "training"
        dataset_package = package / "dataset"
        fake_bin = root / "bin"
        for directory in (training_package, dataset_package, fake_bin):
            directory.mkdir(parents=True, exist_ok=True)

        for path in (
            package / "__init__.py",
            training_package / "__init__.py",
            dataset_package / "__init__.py",
        ):
            path.write_text("", encoding="utf-8")

        (source_root / "toml.py").write_text(
            textwrap.dedent(
                """
                import tomllib

                def load(path):
                    with open(path, "rb") as handle:
                        return tomllib.load(handle)
                """
            ).lstrip(),
            encoding="utf-8",
        )
        (dataset_package / "media_utils.py").write_text(
            textwrap.dedent(
                """
                from pathlib import Path

                def glob_images(directory):
                    supported = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
                    return sorted(
                        str(path)
                        for path in Path(directory).iterdir()
                        if path.is_file() and path.suffix.lower() in supported
                    )
                """
            ).lstrip(),
            encoding="utf-8",
        )
        (training_package / "parser_common.py").write_text(
            textwrap.dedent(
                """
                import argparse
                import tomllib

                def setup_parser_common():
                    parser = argparse.ArgumentParser()
                    parser.add_argument("--config_file")
                    for name in (
                        "dataset_config",
                        "dit",
                        "vae",
                        "turbo_dit",
                        "text_encoder",
                        "sample_prompts",
                        "output_dir",
                        "logging_dir",
                        "output_name",
                    ):
                        parser.add_argument(f"--{name}")
                    for name in (
                        "max_train_steps",
                        "save_every_n_steps",
                        "sample_every_n_steps",
                        "save_last_n_steps_state",
                        "gradient_accumulation_steps",
                        "network_dim",
                    ):
                        parser.add_argument(f"--{name}", type=int)
                    parser.add_argument("--network_alpha", type=float)
                    parser.add_argument("--learning_rate", type=float)
                    parser.add_argument("--network_args", nargs="+")
                    parser.add_argument("--resume")
                    return parser

                def read_config_from_file(args, parser):
                    config_path = (
                        args.config_file
                        if args.config_file.endswith(".toml")
                        else args.config_file + ".toml"
                    )
                    with open(config_path, "rb") as handle:
                        config = tomllib.load(handle)
                    for key, value in config.items():
                        if not hasattr(args, key) or getattr(args, key) is None:
                            setattr(args, key, value)
                    return args
                """
            ).lstrip(),
            encoding="utf-8",
        )
        (package / "krea2_train_network.py").write_text(
            "def krea2_setup_parser(parser):\n    return parser\n",
            encoding="utf-8",
        )
        for filename in (
            "krea2_cache_latents.py",
            "krea2_cache_text_encoder_outputs.py",
        ):
            (package / filename).write_text("", encoding="utf-8")

        accelerate = fake_bin / "accelerate"
        accelerate.write_text(
            "#!/bin/sh\nprintf 'FAKE_ACCELERATE %s\\n' \"$*\"\n",
            encoding="utf-8",
            newline="\n",
        )
        accelerate.chmod(0o755)
        return source_root, fake_bin

    def add_images(
        self, directory: Path, count: int, *, captions: bool = True
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            stem = f"{index + 1:03d}"
            (directory / f"{stem}.png").touch()
            if captions:
                (directory / f"{stem}.txt").write_text(
                    "A person on a plain background. k2v9\n", encoding="utf-8"
                )

    def run_bash(
        self,
        script: Path,
        args: list[str],
        *,
        environment: dict[str, str],
        fake_bin: Path,
    ) -> subprocess.CompletedProcess[str]:
        fake_bin_path = fake_bin.resolve().as_posix()
        if os.name == "nt" and len(fake_bin_path) >= 3 and fake_bin_path[1:3] == ":/":
            fake_bin_path = f"/{fake_bin_path[0].lower()}{fake_bin_path[2:]}"
        command = " ".join(
            [
                f"export PATH={shlex.quote(fake_bin_path)}:$PATH;",
                "exec",
                shlex.quote(script.as_posix()),
                *(shlex.quote(argument) for argument in args),
            ]
        )
        return subprocess.run(
            [self.bash, "-lc", command],
            cwd=REPOSITORY_ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def write_directory_config(
        self, path: Path, image_directories: list[Path]
    ) -> None:
        sections = [
            "[general]",
            'caption_extension = ".txt"',
            "batch_size = 1",
            "",
        ]
        for directory in image_directories:
            sections.extend(
                [
                    "[[datasets]]",
                    f"image_directory = {json.dumps(directory.as_posix())}",
                    "num_repeats = 1",
                    "",
                ]
            )
        path.write_text("\n".join(sections), encoding="utf-8")

    @unittest.skipIf(
        os.name == "nt",
        "Launcher integration uses POSIX PYTHONPATH semantics",
    )
    @unittest.skipUnless(
        Path(r"C:\Program Files\Git\bin\bash.exe").is_file()
        or shutil.which("bash"),
        "Bash is required for launcher integration tests",
    )
    def test_initialization_selectors_and_launcher_planning(self):
        with tempfile.TemporaryDirectory() as temporary:
            runtime = Path(temporary)
            source_root, fake_bin = self.write_fake_runtime(runtime)
            environment = os.environ.copy()
            environment.update(
                {
                    "MUSUBI_HOME": runtime.as_posix(),
                    "MUSUBI_SCRIPTS_DIR": SCRIPTS_DIR.as_posix(),
                    "PYTHONPATH": source_root.as_posix(),
                }
            )

            init_script = SCRIPTS_DIR / "krea2" / "init-krea2-character.sh"
            prepare_script = SCRIPTS_DIR / "krea2" / "prepare-krea2-character.sh"
            train_script = SCRIPTS_DIR / "krea2" / "train-krea2-character.sh"

            initialized = self.run_bash(
                init_script,
                ["--trigger", "k2v9"],
                environment=environment,
                fake_bin=fake_bin,
            )
            self.assertEqual(
                0, initialized.returncode, initialized.stdout + initialized.stderr
            )
            workflow = runtime / "dataset" / "krea2"
            self.assertTrue((workflow / "train.toml").is_file())
            self.assertTrue((workflow / "train-baseline.toml").is_file())
            self.assertTrue((workflow / "train-quality.toml").is_file())
            self.assertFalse((workflow / "train-attention.toml").exists())

            marker = "\n# preserved-edit\n"
            with (workflow / "train.toml").open("a", encoding="utf-8") as handle:
                handle.write(marker)
            repeated = self.run_bash(
                init_script,
                ["--trigger", "k2v9"],
                environment=environment,
                fake_bin=fake_bin,
            )
            self.assertEqual(0, repeated.returncode)
            self.assertIn(
                marker.strip(),
                (workflow / "train.toml").read_text(encoding="utf-8"),
            )

            self.add_images(workflow / "images", 30)
            for model in (
                runtime / "models" / "krea2" / "raw.safetensors",
                runtime / "models" / "krea2" / "turbo.safetensors",
                runtime / "models" / "vae" / "qwen_image_vae.safetensors",
                runtime
                / "models"
                / "text_encoders"
                / "qwen3vl_4b_bf16.safetensors",
            ):
                model.parent.mkdir(parents=True, exist_ok=True)
                model.touch()

            for preset in ("default", "baseline", "quality", "10gb"):
                with self.subTest(selector="prepare", preset=preset):
                    prepared = self.run_bash(
                        prepare_script,
                        ["--preset", preset],
                        environment=environment,
                        fake_bin=fake_bin,
                    )
                    self.assertEqual(
                        0, prepared.returncode, prepared.stdout + prepared.stderr
                    )

            rejected_prepare = self.run_bash(
                prepare_script,
                ["--preset", "attention"],
                environment=environment,
                fake_bin=fake_bin,
            )
            self.assertEqual(64, rejected_prepare.returncode)
            self.assertIn("Unknown Krea2 preset: attention", rejected_prepare.stderr)

            expected_summaries = {
                "default": (
                    "attention-only",
                    "8000",
                    "266.7",
                    "400",
                ),
                "baseline": ("all-linear", "4000", "133.3", "200"),
                "quality": ("all-linear", "6000", "200.0", "300"),
            }
            for preset, (
                targets,
                steps,
                passes,
                interval,
            ) in expected_summaries.items():
                with self.subTest(selector="train", preset=preset):
                    trained = self.run_bash(
                        train_script,
                        ["--preset", preset],
                        environment=environment,
                        fake_bin=fake_bin,
                    )
                    output = trained.stdout + trained.stderr
                    self.assertEqual(0, trained.returncode, output)
                    self.assertRegex(
                        output, rf"Target modules:\s+{re.escape(targets)}"
                    )
                    self.assertRegex(
                        output, rf"Maximum optimizer steps:\s+{steps}"
                    )
                    self.assertRegex(
                        output, rf"Estimated maximum passes:\s+{passes}"
                    )
                    self.assertRegex(
                        output, rf"Checkpoint interval:\s+{interval}"
                    )
                    self.assertRegex(
                        output, r"Periodic checkpoint candidates:\s+20"
                    )
                    self.assertRegex(
                        output, rf"Final checkpoint:\s+duplicates step {steps}"
                    )
                    self.assertRegex(
                        output, r"Unique candidate states:\s+20"
                    )
                    self.assertIn("FAKE_ACCELERATE launch", output)

            rejected_train = self.run_bash(
                train_script,
                ["--preset", "attention"],
                environment=environment,
                fake_bin=fake_bin,
            )
            self.assertEqual(64, rejected_train.returncode)
            self.assertIn("Unknown Krea2 preset: attention", rejected_train.stderr)

            overridden = self.run_bash(
                train_script,
                [
                    "--max_train_steps",
                    "7900",
                    "--network_args",
                    "exclude_patterns=['foo']",
                ],
                environment=environment,
                fake_bin=fake_bin,
            )
            overridden_output = overridden.stdout + overridden.stderr
            self.assertEqual(0, overridden.returncode, overridden_output)
            self.assertRegex(overridden_output, r"Target modules:\s+custom")
            self.assertRegex(
                overridden_output, r"Periodic checkpoint candidates:\s+19"
            )
            self.assertRegex(
                overridden_output,
                r"Final checkpoint:\s+additional state at step 7900",
            )
            self.assertRegex(
                overridden_output, r"Unique candidate states:\s+20"
            )

            reordered_attention = self.run_bash(
                train_script,
                [
                    "--network_args",
                    (
                        "exclude_patterns=['first','.*\\.mlp\\..*',"
                        "'last\\.linear','tmlp\\..*','txtmlp\\..*',"
                        "'tproj\\.1','txtfusion\\..*']"
                    ),
                ],
                environment=environment,
                fake_bin=fake_bin,
            )
            reordered_attention_output = (
                reordered_attention.stdout + reordered_attention.stderr
            )
            self.assertEqual(
                0, reordered_attention.returncode, reordered_attention_output
            )
            self.assertRegex(
                reordered_attention_output, r"Target modules:\s+attention-only"
            )

            missing_dataset = self.run_bash(
                train_script,
                ["--dataset_config", (runtime / "missing-dataset.toml").as_posix()],
                environment=environment,
                fake_bin=fake_bin,
            )
            self.assertEqual(2, missing_dataset.returncode)
            self.assertIn(
                "Missing required file for --dataset_config:",
                missing_dataset.stderr,
            )
            self.assertIn(
                "Run init-krea2-character.sh or provide an existing dataset config.",
                missing_dataset.stderr,
            )
            self.assertNotIn("Traceback", missing_dataset.stderr)

            no_dataset_config = runtime / "train-without-dataset.toml"
            no_dataset_config.write_text(
                "max_train_steps = 100\n",
                encoding="utf-8",
            )
            missing_dataset_setting = self.run_bash(
                train_script,
                ["--config_file", no_dataset_config.as_posix()],
                environment=environment,
                fake_bin=fake_bin,
            )
            self.assertEqual(2, missing_dataset_setting.returncode)
            self.assertIn(
                "No --dataset_config path is configured",
                missing_dataset_setting.stderr,
            )
            self.assertNotIn("Traceback", missing_dataset_setting.stderr)

            nineteen_images = runtime / "nineteen"
            self.add_images(nineteen_images, 19)
            nineteen_config = runtime / "dataset-19.toml"
            self.write_directory_config(nineteen_config, [nineteen_images])
            warned = self.run_bash(
                train_script,
                ["--dataset_config", nineteen_config.as_posix()],
                environment=environment,
                fake_bin=fake_bin,
            )
            self.assertEqual(0, warned.returncode, warned.stdout + warned.stderr)
            self.assertIn("intended for roughly 20-40", warned.stderr)

            primary = runtime / "multi-primary"
            regularization = runtime / "multi-regularization"
            self.add_images(primary, 30, captions=False)
            self.add_images(regularization, 5, captions=False)
            multi_config = runtime / "dataset-multi.toml"
            self.write_directory_config(multi_config, [primary, regularization])
            multi = self.run_bash(
                train_script,
                ["--dataset_config", multi_config.as_posix()],
                environment=environment,
                fake_bin=fake_bin,
            )
            multi_output = multi.stdout + multi.stderr
            self.assertEqual(0, multi.returncode, multi_output)
            self.assertRegex(multi_output, r"Additional datasets:\s+1")
            self.assertIn(
                "unavailable (multi-dataset configuration)", multi_output
            )

            jsonl_config = runtime / "dataset-jsonl.toml"
            jsonl_file = runtime / "images.jsonl"
            jsonl_file.touch()
            jsonl_config.write_text(
                "\n".join(
                    [
                        "[general]",
                        "batch_size = 1",
                        "",
                        "[[datasets]]",
                        f"image_jsonl_file = {json.dumps(jsonl_file.as_posix())}",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            jsonl = self.run_bash(
                train_script,
                ["--dataset_config", jsonl_config.as_posix()],
                environment=environment,
                fake_bin=fake_bin,
            )
            jsonl_output = jsonl.stdout + jsonl.stderr
            self.assertEqual(0, jsonl.returncode, jsonl_output)
            self.assertIn(
                "unavailable (primary dataset uses image_jsonl_file)",
                jsonl_output,
            )

            missing_caption = runtime / "missing-caption"
            self.add_images(missing_caption, 1, captions=False)
            missing_config = runtime / "dataset-missing.toml"
            self.write_directory_config(missing_config, [missing_caption])
            missing = self.run_bash(
                train_script,
                ["--dataset_config", missing_config.as_posix()],
                environment=environment,
                fake_bin=fake_bin,
            )
            self.assertEqual(2, missing.returncode)
            self.assertIn("Missing caption", missing.stderr)


if __name__ == "__main__":
    unittest.main()
