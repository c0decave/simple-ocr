"""dist.sh builds a copyable package; package/start.sh starts it elsewhere. podman is faked."""
import hashlib
import os
import shutil
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_FILES = ["dist.sh", "package", "app", "VERSION", "Containerfile", ".env.example", ".gitignore", ".containerignore",
              "README.md"]
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".webp", ".gif")

FAKE_PODMAN = r"""#!/bin/sh
echo "$*" >> "$FAKE_LOG"
case "$1" in
    build)
        [ -n "${FAKE_BUILD_FAIL:-}" ] && exit 3
        for arg; do context=$arg; done
        (cd "$context" && find . -type f | sort) > "$FAKE_LOG.context" ;;
    save)
        while [ $# -gt 0 ]; do [ "$1" = "-o" ] && echo IMAGE > "$2"; shift; done ;;
    image)
        [ -n "${FAKE_HAS_IMAGE:-}" ] && exit 0
        exit 1 ;;
esac
exit 0
"""


def git(repo, *args):
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *args], cwd=repo, check=True,
                   capture_output=True)


class DistTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.repo = self.tmp / "repo"
        self.repo.mkdir()
        for name in REPO_FILES:
            source = ROOT / name
            if source.is_dir():
                shutil.copytree(source, self.repo / name, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(source, self.repo / name)
        git(self.repo, "init", "-q")
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "init")
        self.bin = self.tmp / "bin"
        self.bin.mkdir()
        fake = self.bin / "podman"
        fake.write_text(FAKE_PODMAN)
        fake.chmod(0o755)
        self.log = self.tmp / "podman.log"
        self.version = (self.repo / "VERSION").read_text().strip()

    def env(self, **extra):
        return dict(os.environ, PATH="%s:%s" % (self.bin, os.environ["PATH"]), FAKE_LOG=str(self.log), **extra)

    def dist(self, **extra):
        return subprocess.run(["sh", str(self.repo / "dist.sh")], env=self.env(**extra), capture_output=True, text=True)

    def package_path(self):
        return self.repo / "dist" / ("simple-ocr-%s.tar.gz" % self.version)

    def members(self):
        with tarfile.open(self.package_path()) as archive:
            return {m.name: m for m in archive.getmembers()}

    def unpack(self):
        target = self.tmp / "elsewhere"
        with tarfile.open(self.package_path()) as archive:
            archive.extractall(target, filter="data")
        return target / ("simple-ocr-" + self.version)


class DistBuildTests(DistTestCase):
    def test_builds_one_package_with_everything_needed_to_start_it(self):
        result = self.dist()
        self.assertEqual(result.returncode, 0, result.stderr)
        prefix = "simple-ocr-%s/" % self.version
        names = set(self.members())
        for name in ("image.tar", "start.sh", "compose.yaml", ".env.example", "README.txt", "VERSION", "COMMIT"):
            self.assertIn(prefix + name, names)
        self.assertTrue(self.members()[prefix + "start.sh"].mode & 0o111, "start.sh must be executable")
        unpacked = self.unpack()
        image = "localhost/simple-ocr:" + self.version
        self.assertIn("image: " + image, (unpacked / "compose.yaml").read_text())
        self.assertIn(image, (unpacked / "start.sh").read_text())
        for name in ("start.sh", "compose.yaml", "README.txt"):
            self.assertNotIn("__VERSION__", (unpacked / name).read_text(), name)
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=self.repo, capture_output=True,
                              text=True).stdout.strip()
        self.assertEqual((unpacked / "COMMIT").read_text().strip(), head)
        self.assertIn("build -q -t %s" % image, self.log.read_text())

    def test_checksum_file_matches_the_package(self):
        self.dist()
        line = self.package_path().with_name(self.package_path().name + ".sha256").read_text()
        digest = hashlib.sha256(self.package_path().read_bytes()).hexdigest()
        self.assertEqual(line.split()[0], digest)

    def test_no_env_no_saved_runs_and_no_images_reach_the_build_or_the_package(self):
        (self.repo / ".env").write_text("LMSTUDIO_API_KEY=top-secret-key\n")
        (self.repo / "data" / "runs").mkdir(parents=True)
        (self.repo / "data" / "runs" / "0123456789ab.json").write_text('{"text": "private OCR text"}')
        (self.repo / "app" / "upload.png").write_bytes(b"\x89PNG secret pixels")
        (self.repo / ".git" / "info" / "exclude").write_text("data/\napp/upload.png\n")
        result = self.dist()
        self.assertEqual(result.returncode, 0, result.stderr)
        context = (self.tmp / "podman.log.context").read_text().split()
        self.assertTrue(context, "the build context was not recorded")
        for listing in (context, list(self.members())):
            for name in listing:
                self.assertNotIn(".env", name.replace(".env.example", ""), name)
                self.assertNotIn("data/runs", name)
                self.assertFalse(name.lower().endswith(IMAGE_SUFFIXES), name)
        with tarfile.open(self.package_path()) as archive:
            for member in archive.getmembers():
                if member.isfile():
                    content = archive.extractfile(member).read()
                    self.assertNotIn(b"top-secret-key", content, member.name)
                    self.assertNotIn(b"private OCR text", content, member.name)

    def test_the_source_tarball_is_the_committed_tree_without_git_or_private_files(self):
        (self.repo / ".private-words").write_text("my-secret-host\n")
        (self.repo / ".git" / "info" / "exclude").write_text(".private-words\n")
        result = self.dist()
        self.assertEqual(result.returncode, 0, result.stderr)
        source = self.repo / "dist" / ("simple-ocr-%s-source.tar.gz" % self.version)
        with tarfile.open(source) as archive:
            names = archive.getnames()
        prefix = "simple-ocr-%s-source/" % self.version
        self.assertIn(prefix + "app/server.py", names)
        self.assertIn(prefix + "dist.sh", names)
        self.assertFalse([n for n in names if "/.git/" in n or n.endswith(".private-words") or n.endswith("/.env")], names)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        self.assertEqual(source.with_name(source.name + ".sha256").read_text().split()[0], digest)

    def commit(self, path, text):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text)
        git(self.repo, "add", "-A")
        git(self.repo, "commit", "-q", "-m", "change")

    def assert_refused(self, needle):
        result = self.dist()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Private data", result.stderr)
        self.assertIn(needle, result.stderr)
        self.assertFalse(self.package_path().exists())
        self.assertFalse(self.log.exists(), "nothing may be built")

    def test_private_addresses_stop_the_release(self):
        for address in ("192.168" + ".88.131", "10.20" + ".30.40", "172.16" + ".0.1"):
            with self.subTest(address=address):
                self.commit("app/notes.txt", "LM Studio runs on %s\n" % address)
                self.assert_refused("app/notes.txt")

    def test_mail_addresses_private_keys_and_home_paths_stop_the_release(self):
        begin = "-----" + "BEGIN "
        for text in ("contact: someone" + "@example.org", begin + "PRIVATE KEY-----", begin + "RSA PRIVATE KEY-----",
                     "see /" + "home/someone/project", "data in /" + "mnt/ext/stuff"):
            with self.subTest(text=text):
                self.commit("app/notes.txt", text + "\n")
                self.assert_refused("app/notes.txt")

    def test_words_from_the_local_private_list_stop_the_release(self):
        (self.repo / ".private-words").write_text("# my names\n\nMyServerName\n")
        (self.repo / ".git" / "info" / "exclude").write_text(".private-words\n")
        self.commit("README.md", "Runs on myservername behind a proxy\n")
        self.assert_refused("README.md")

    def test_a_scan_that_fails_stops_the_release_instead_of_passing(self):
        broken = self.bin / "grep"
        broken.write_text("#!/bin/sh\necho 'grep: read error' >&2\nexit 2\n")
        broken.chmod(0o755)
        result = self.dist()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("scan failed", result.stderr)
        self.assertFalse(self.package_path().exists())
        self.assertFalse(self.log.exists(), "nothing may be built")

    def test_an_unreadable_private_word_list_stops_the_release(self):
        (self.repo / ".private-words").write_text("MyServerName\n")
        (self.repo / ".git" / "info" / "exclude").write_text(".private-words\n")
        broken = self.bin / "grep"   # only reading the word list fails, the address scan works
        broken.write_text('#!/bin/sh\ncase "$*" in *.private-words*) exit 2 ;; esac\nexec %s "$@"\n' % shutil.which("grep"))
        broken.chmod(0o755)
        result = self.dist()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(".private-words", result.stderr)
        self.assertFalse(self.package_path().exists())

    def test_a_word_list_with_windows_line_ends_and_trailing_spaces_still_works(self):
        (self.repo / ".private-words").write_bytes(b"# names\r\n \tMyServerName  \r\n")
        (self.repo / ".git" / "info" / "exclude").write_text(".private-words\n")
        self.commit("README.md", "Runs on myservername behind a proxy\n")
        self.assert_refused("README.md")

    def test_a_word_list_with_only_comments_passes(self):
        (self.repo / ".private-words").write_text("# nothing yet\n\n")
        (self.repo / ".git" / "info" / "exclude").write_text(".private-words\n")
        result = self.dist()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_file_names_with_spaces_and_wildcards_are_reported_as_they_are(self):
        self.commit("docs/my notes *.txt", "server at 192.168" + ".1.1\n")
        result = self.dist()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("  docs/my notes *.txt\n", result.stderr)

    def test_documentation_addresses_and_localhost_pass(self):
        self.commit("app/notes.txt", "examples: 192.0.2.10 198.51.100.7 203.0.113.9 127.0.0.1 localhost lm.example\n")
        result = self.dist()
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_refuses_uncommitted_changes(self):
        (self.repo / "app" / "server.py").write_text("changed\n")
        result = self.dist()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Uncommitted changes", result.stderr)
        self.assertFalse(self.package_path().exists())
        self.assertFalse(self.log.exists(), "nothing may be built")

    def test_failed_build_leaves_no_package(self):
        result = self.dist(FAKE_BUILD_FAIL="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.package_path().exists())
        self.assertFalse((self.repo / "dist" / ("simple-ocr-" + self.version)).exists())

    def test_a_second_run_replaces_the_package_without_leftovers(self):
        self.dist()
        first = set(self.members())
        result = self.dist()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(set(self.members()), first)
        source = "simple-ocr-%s-source.tar.gz" % self.version
        self.assertEqual(sorted(p.name for p in (self.repo / "dist").iterdir()),
                         sorted([self.package_path().name, self.package_path().name + ".sha256", source, source + ".sha256"]))


class StartScriptTests(DistTestCase):
    def setUp(self):
        super().setUp()
        self.assertEqual(self.dist().returncode, 0)
        self.log.unlink()
        self.here = self.unpack()

    def start(self, path=None, **extra):
        env = self.env(**extra)
        if path is not None:
            env["PATH"] = path
        return subprocess.run(["sh", str(self.here / "start.sh")], env=env, capture_output=True, text=True)

    def test_loads_the_image_once_and_runs_it_with_env_and_volume(self):
        (self.here / ".env").write_text("BACKEND=ollama\nAPP_PORT=9090\n")
        result = self.start()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.log.read_text().splitlines()
        image = "localhost/simple-ocr:" + self.version
        self.assertIn("load -i image.tar", calls)
        run = [c for c in calls if c.startswith("run ")]
        self.assertEqual(len(run), 1)
        for part in ("--env-file .env", "-v simple-ocr-data:/data", "-p 127.0.0.1:9090:8080", image):
            self.assertIn(part, run[0])

    def test_skips_the_load_when_the_image_is_already_there(self):
        (self.here / ".env").write_text("BACKEND=lmstudio\n")
        result = self.start(FAKE_HAS_IMAGE="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.log.read_text()
        self.assertNotIn("load", calls)
        self.assertIn("-p 127.0.0.1:8080:8080", calls)

    def test_refuses_to_start_without_env(self):
        result = self.start()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No .env found", result.stderr)
        self.assertFalse(self.log.exists())

    def test_app_bind_publishes_on_another_address(self):
        (self.here / ".env").write_text("APP_BIND=0.0.0.0\nAPP_PORT=8088\n")
        result = self.start(FAKE_HAS_IMAGE="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-p 0.0.0.0:8088:8080", self.log.read_text())

    def test_an_ipv6_bind_address_is_put_in_brackets(self):
        (self.here / ".env").write_text("APP_BIND=::1\n")
        result = self.start(FAKE_HAS_IMAGE="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("-p [::1]:8080:8080", self.log.read_text())
        (self.here / ".env").write_text("APP_BIND=[fd00::5]\n")
        self.start(FAKE_HAS_IMAGE="1")
        self.assertIn("-p [fd00::5]:8080:8080", self.log.read_text())

    def test_reports_a_missing_container_engine(self):
        (self.here / ".env").write_text("BACKEND=lmstudio\n")
        tools = self.tmp / "tools"
        tools.mkdir()
        for tool in ("sh", "sed", "tail", "tr", "dirname", "cat"):
            os.symlink(shutil.which(tool), tools / tool)
        result = self.start(path=str(tools))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Neither podman nor docker", result.stderr)


if __name__ == "__main__":
    unittest.main()
