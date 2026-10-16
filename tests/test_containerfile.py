"""The image build stays quiet: no debconf frontend hunt and no pip root warning in the build log."""

import re
import unittest
from pathlib import Path

CONTAINERFILE = Path(__file__).resolve().parent.parent / "Containerfile"

def run_commands(text):
    """Each RUN instruction as one line, with continuation lines joined."""
    joined = re.sub(r"\\\n\s*", " ", text)
    return [line[len("RUN "):] for line in joined.splitlines() if line.startswith("RUN ")]

def commands_of(text, tool):
    """The single commands of all RUN lines that call tool (split at && and ;)."""
    return [part.strip() for run in run_commands(text) for part in re.split(r"&&|;", run)
            if re.search(r"(^|\s)%s(\s|$)" % re.escape(tool), part)]

class ContainerfileTests(unittest.TestCase):
    def setUp(self):
        self.text = CONTAINERFILE.read_text()

    def test_apt_get_install_runs_noninteractive(self):
        installs = [c for c in commands_of(self.text, "apt-get") if " install " in " %s " % c]
        self.assertTrue(installs, "the image installs the DejaVu fonts with apt-get")
        for command in installs:
            self.assertRegex(command, r"(^|\s)DEBIAN_FRONTEND=noninteractive\s", command)

    def test_pip_install_does_not_warn_about_root(self):
        installs = [c for c in commands_of(self.text, "pip") if " install " in " %s " % c]
        self.assertTrue(installs, "the image installs requirements.txt with pip")
        for command in installs:
            self.assertIn("--root-user-action=ignore", command.split(), command)

    def test_frontend_is_not_left_in_the_image_environment(self):
        # DEBIAN_FRONTEND belongs to the build step; an ENV would leak it into every running container.
        self.assertNotRegex(self.text, r"(?m)^ENV\b[^\n]*DEBIAN_FRONTEND")

    def test_helpers_join_continuation_lines(self):
        text = "FROM x\nRUN apt-get update && \\\n    apt-get install -y a; pip install b\n"
        self.assertEqual(commands_of(text, "apt-get"), ["apt-get update", "apt-get install -y a"])
        self.assertEqual(commands_of(text, "pip"), ["pip install b"])
        self.assertEqual(commands_of("FROM x\nCMD pip install c\n", "pip"), [])

if __name__ == "__main__":
    unittest.main()
