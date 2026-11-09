"""Verify synchronization against local upstream and fork repositories."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).with_name("sync-upstream.sh")


class SyncTests(unittest.TestCase):
    def git(self, directory, *args):
        result = subprocess.run(
            ["git", "-C", str(directory), *args],
            text=True,
            capture_output=True,
            check=True,
        )
        return result.stdout.strip()

    def commit(self, directory, name, value):
        (directory / name).write_text(value)
        self.git(directory, "add", name)
        self.git(
            directory,
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.test",
            "commit",
            "-m",
            value,
        )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.upstream = self.root / "upstream"
        self.upstream.mkdir()
        self.git(self.upstream, "init", "--initial-branch=main")
        self.commit(self.upstream, "shared.txt", "base")
        self.origin = self.root / "origin.git"
        self.git(self.root, "clone", "--bare", str(self.upstream), str(self.origin))
        self.fork = self.root / "fork"
        self.git(self.root, "clone", str(self.origin), str(self.fork))
        self.git(self.fork, "config", "user.name", "Fixture")
        self.git(self.fork, "config", "user.email", "fixture@example.test")
        self.commit(self.fork, "fork.txt", "fork feature")
        self.git(self.fork, "push", "origin", "main")
        self.main = self.git(self.fork, "rev-parse", "HEAD")
        binary = self.root / "bin"
        binary.mkdir()
        self.log = self.root / "gh.jsonl"
        gh = binary / "gh"
        gh.write_text(
            f"#!{sys.executable}\nimport json,os,pathlib,sys\n"
            "args=sys.argv[1:]\n"
            "with open(os.environ['GH_LOG'],'a') as f: f.write(json.dumps(args)+'\\n')\n"
            "p=pathlib.Path(os.environ['GH_LOG']+'.pr')\n"
            "if args[:2]==['pr','list'] and p.exists(): print('1')\n"
            "if args[:2]==['pr','create']: p.write_text('1')\n"
        )
        gh.chmod(0o755)
        self.env = dict(
            os.environ,
            PATH=str(binary) + os.pathsep + os.environ["PATH"],
            GITHUB_REPOSITORY="fixture/fork",
            GH_LOG=str(self.log),
            GITHUB_STEP_SUMMARY=str(self.root / "summary"),
        )

    def sync(self):
        return subprocess.run(
            ["bash", str(SCRIPT.resolve()), str(self.upstream), "checks.yml"],
            cwd=self.fork,
            env=self.env,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_no_change_does_not_push_or_create_pr(self):
        result = self.sync()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(self.log.exists())
        self.assertEqual(self.git(self.origin, "rev-parse", "main"), self.main)

    def test_updates_preserve_fork_and_reuse_pr(self):
        self.commit(self.upstream, "upstream.txt", "upstream feature")
        result = self.sync()
        self.assertEqual(result.returncode, 0, result.stderr)
        branch = "automation/upstream-sync"
        first = self.git(self.origin, "rev-parse", branch)
        self.assertEqual(
            self.git(self.origin, "show", f"{branch}:fork.txt"), "fork feature"
        )
        self.assertEqual(
            self.git(self.origin, "show", f"{branch}:upstream.txt"), "upstream feature"
        )
        self.assertEqual(self.git(self.origin, "rev-parse", "main"), self.main)
        result = self.sync()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.git(self.origin, "rev-parse", branch), first)
        self.commit(self.upstream, "next.txt", "next upstream feature")
        result = self.sync()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.git(self.origin, "merge-base", "--is-ancestor", first, branch)
        calls = [json.loads(line) for line in self.log.read_text().splitlines()]
        self.assertEqual(sum(call[:2] == ["pr", "create"] for call in calls), 1)
        self.assertEqual(sum(call[:2] == ["workflow", "run"] for call in calls), 2)

    def test_conflict_keeps_main_and_remote_branch_unchanged(self):
        self.commit(self.fork, "shared.txt", "fork change")
        self.git(self.fork, "push", "origin", "main")
        before = self.git(self.origin, "rev-parse", "main")
        self.commit(self.upstream, "shared.txt", "upstream change")
        result = self.sync()
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.git(self.origin, "rev-parse", "main"), before)
        self.assertIn("shared.txt", (self.root / "summary").read_text())
        self.assertFalse(self.log.exists())


if __name__ == "__main__":
    unittest.main()
