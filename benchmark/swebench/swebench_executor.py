"""SWE-bench executor - based on official swebench harness implementation."""
import asyncio
import json
import re
import shutil
import subprocess
import tempfile
import uuid
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

from base.engine.logs import logger
from benchmark.swebench.data_loader import SWEBenchInstance

# ============================================================================
# Constants from official swebench (swebench/harness/constants.py)
# ============================================================================

START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"

NON_TEST_EXTS = [
    ".json", ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".ico",
    ".txt", ".md", ".rst", ".csv", ".tsv", ".xml", ".yaml", ".yml",
    ".toml", ".cfg", ".ini", ".conf", ".lock", ".log",
]

# Repository-specific test commands (simplified from MAP_REPO_VERSION_TO_SPECS)
REPO_TEST_CMDS = {
    "astropy/astropy": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
    "django/django": "./tests/runtests.py --verbosity 2 {tests}",
    "matplotlib/matplotlib": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
    "pallets/flask": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
    "psf/requests": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
    "pylint-dev/pylint": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
    "pytest-dev/pytest": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
    "scikit-learn/scikit-learn": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
    "sphinx-doc/sphinx": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
    "sympy/sympy": "bin/test -C --verbose {tests}",
    "pydata/xarray": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
    "mwaskom/seaborn": "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider",
}

DEFAULT_TEST_CMD = "python -m pytest {tests} --no-header -rA --tb=no -p no:cacheprovider"


def make_official_test_spec(instance: SWEBenchInstance):
    """Build a TestSpec without regenerating the already-cached image environment."""
    from swebench.harness.constants import MAP_REPO_TO_EXT, MAP_REPO_VERSION_TO_SPECS
    from swebench.harness.test_spec.create_scripts import make_eval_script_list
    from swebench.harness.test_spec.test_spec import TestSpec

    instance_data = {
        "instance_id": instance.instance_id,
        "repo": instance.repo,
        "base_commit": instance.base_commit,
        "problem_statement": instance.problem_statement,
        "hints_text": instance.hints_text,
        "created_at": instance.created_at,
        "patch": instance.patch,
        "test_patch": instance.test_patch,
        "version": instance.version,
        "environment_setup_commit": instance.environment_setup_commit,
        "FAIL_TO_PASS": instance.FAIL_TO_PASS or [],
        "PASS_TO_PASS": instance.PASS_TO_PASS or [],
    }
    specs = MAP_REPO_VERSION_TO_SPECS[instance.repo][instance.version]
    env_name = "testbed"
    repo_directory = "/testbed"
    eval_script_list = make_eval_script_list(
        instance_data,
        specs,
        env_name,
        repo_directory,
        instance.base_commit,
        instance.test_patch,
    )
    return TestSpec(
        instance_id=instance.instance_id,
        repo=instance.repo,
        version=instance.version,
        repo_script_list=[],
        eval_script_list=eval_script_list,
        env_script_list=[],
        arch="x86_64",
        FAIL_TO_PASS=instance.FAIL_TO_PASS or [],
        PASS_TO_PASS=instance.PASS_TO_PASS or [],
        language=MAP_REPO_TO_EXT[instance.repo],
        docker_specs=specs.get("docker_specs", {}),
        namespace=None,
    )


def official_report_to_results(instance_id: str, report_map: Dict[str, Any]):
    """Convert an official harness report to the runner's existing result schema."""
    report = dict(report_map.get(instance_id) or {})
    tests_status = report.get("tests_status") or {}
    fail_to_pass = tests_status.get("FAIL_TO_PASS") or {}
    pass_to_pass = tests_status.get("PASS_TO_PASS") or {}
    f2p_passed = list(fail_to_pass.get("success") or [])
    f2p_failed = list(fail_to_pass.get("failure") or [])
    p2p_passed = list(pass_to_pass.get("success") or [])
    p2p_failed = list(pass_to_pass.get("failure") or [])
    resolved = bool(report.get("resolved", False))

    results = {
        "fail_to_pass": {"passed": f2p_passed, "failed": f2p_failed},
        "pass_to_pass": {"passed": p2p_passed, "failed": p2p_failed},
        "reward": 1.0 if resolved else 0.0,
        "resolved": resolved,
        "summary": {
            "fail_to_pass": f"{len(f2p_passed)}/{len(f2p_passed) + len(f2p_failed)}",
            "pass_to_pass": f"{len(p2p_passed)}/{len(p2p_passed) + len(p2p_failed)}",
        },
        "official_report": report,
        "evaluator": f"swebench.harness/{package_version('swebench')}",
    }
    return results


def make_offline_official_eval_script(test_spec) -> str:
    """Render the official eval script while avoiding redundant online rebuilds.

    The cached SWE-bench instance images already contain an installed checkout.
    Pure-Python tasks import from ``/testbed`` during evaluation, so a second
    editable install is redundant. Official commands that already opt out of
    build isolation remain unchanged.
    """
    commands = []
    for command in test_spec.eval_script_list:
        normalized = command.strip().lower()
        is_editable_install = (
            "pip install" in normalized
            and (" -e ." in normalized or " --editable ." in normalized)
        )
        if is_editable_install and "--no-build-isolation" not in normalized:
            commands.append(
                "# [AORCHESTRA OFFLINE] skipped redundant install: " + command
            )
        else:
            commands.append(command)
    return "\n".join(["#!/bin/bash", "set -uxo pipefail", *commands]) + "\n"


# ============================================================================
# Utility functions from official swebench
# ============================================================================

def get_modified_files(patch: str) -> List[str]:
    """Extract list of modified files from a patch (from swebench/harness/utils.py)."""
    diff_pat = r"diff --git a/.* b/(.*)"
    return re.findall(diff_pat, patch)


def get_test_directives(repo: str, test_patch: str) -> List[str]:
    """
    Get test directives from the test_patch of a task instance.
    Based on swebench/harness/test_spec/python.py:get_test_directives
    """
    diff_pat = r"diff --git a/.* b/(.*)"
    directives = re.findall(diff_pat, test_patch)
    directives = [
        d for d in directives if not any(d.endswith(ext) for ext in NON_TEST_EXTS)
    ]
    
    # For Django tests, remove extension + "tests/" prefix and convert slashes to dots
    if repo == "django/django":
        directives_transformed = []
        for d in directives:
            d = d[: -len(".py")] if d.endswith(".py") else d
            d = d[len("tests/") :] if d.startswith("tests/") else d
            d = d.replace("/", ".")
            directives_transformed.append(d)
        directives = directives_transformed
    
    return directives


def make_eval_script(
    repo: str,
    base_commit: str,
    test_patch: str,
    repo_directory: str = "/testbed",
    env_name: str = "testbed",
) -> str:
    """
    Generate evaluation script based on official swebench implementation.
    Based on swebench/harness/test_spec/python.py:make_eval_script_list_py
    """
    HEREDOC_DELIMITER = "EOF_114329324912"
    
    # Get test files and directives
    test_files = get_modified_files(test_patch) if test_patch else []
    test_files_str = " ".join(test_files) if test_files else ""
    
    test_directives = get_test_directives(repo, test_patch) if test_patch else []
    directives_str = " ".join(test_directives) if test_directives else ""
    
    # Get test command for repo
    test_cmd_template = REPO_TEST_CMDS.get(repo, DEFAULT_TEST_CMD)
    test_cmd = test_cmd_template.format(tests=directives_str)
    
    # Reset test files command
    reset_tests_command = f"git checkout {base_commit} -- {test_files_str}" if test_files_str else ":"
    
    # Apply test patch command
    apply_test_patch_command = (
        f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
        if test_patch else ":"
    )
    
    # Build eval script following official pattern
    eval_commands = [
        "#!/bin/bash",
        "set -uxo pipefail",  # Don't use -e to allow tests to fail
        "",
        "# Activate conda environment",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        f"cd {repo_directory}",
        "",
        f"git config --global --add safe.directory {repo_directory}",
        f"cd {repo_directory}",
        "",
        "# Informational output",
        "git status",
        "git show --stat",
        f"git -c core.fileMode=false diff {base_commit}",
        "",
        "# Re-activate environment",
        "source /opt/miniconda3/bin/activate",
        f"conda activate {env_name}",
        "",
        "# Reset test files to base commit state",
        reset_tests_command,
        "",
        "# Apply test patch",
        apply_test_patch_command,
        "",
        "# Run tests",
        f": '{START_TEST_OUTPUT}'",
        test_cmd,
        f": '{END_TEST_OUTPUT}'",
        "",
        "# Revert test files",
        reset_tests_command,
    ]
    
    return "\n".join(eval_commands) + "\n"


TEST_STATUSES = ("PASSED", "FAILED", "SKIPPED", "ERROR", "XFAIL", "XPASS")


def parse_log_pytest(log: str) -> Dict[str, str]:
    """Parse pytest ``-rA`` summaries and older verbose progress output.

    Current pytest summaries put the status first (``FAILED path::test``),
    while some older/verbose versions put it last (``path::test PASSED``).
    SWE-bench instance test IDs must be matched individually; an aggregate
    ``N passed`` line is never evidence that an unparsed target test passed.
    """
    test_status: Dict[str, str] = {}
    escapes = "".join(chr(char) for char in range(1, 32))
    translator = str.maketrans("", "", escapes)

    for raw_line in log.split("\n"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", raw_line).translate(translator).strip()
        if not line:
            continue

        prefix = next((status for status in TEST_STATUSES if line.startswith(status + " ")), None)
        if prefix:
            remainder = line[len(prefix):].strip()
            if prefix == "FAILED":
                remainder = remainder.split(" - ", 1)[0]
            parts = remainder.split()
            if not parts:
                continue
            # Ignore pytest's aggregate ``SKIPPED [N] path:line: reason`` line.
            if prefix == "SKIPPED" and re.fullmatch(r"\[\d+\]", parts[0]):
                continue
            test_status[remainder] = prefix
            continue

        suffix = next((status for status in TEST_STATUSES if line.endswith(" " + status)), None)
        if suffix:
            test_name = line[: -len(suffix)].strip()
            if test_name:
                test_status[test_name] = suffix

    return test_status


def parse_log_django(log: str) -> Dict[str, str]:
    """
    Parse Django test output log.
    Based on swebench/harness/log_parsers/django_log_parser.py
    """
    test_status: Dict[str, str] = {}
    previous_test: Optional[str] = None

    for raw_line in log.split("\n"):
        line = raw_line.strip()
        if " ... " in line:
            previous_test = line.split(" ... ", 1)[0]
        for suffix in (" ... ok", " ... OK", " ...  OK"):
            if line.endswith(suffix):
                test_status[line.rsplit(suffix, 1)[0]] = "PASSED"
                break
        if " ... skipped" in line:
            test_status[line.split(" ... skipped", 1)[0]] = "SKIPPED"
        if line.endswith(" ... FAIL"):
            test_status[line.rsplit(" ... FAIL", 1)[0]] = "FAILED"
        if line.startswith("FAIL:"):
            parts = line.split()
            if len(parts) > 1:
                test_status[parts[1]] = "FAILED"
        if line.endswith(" ... ERROR"):
            test_status[line.rsplit(" ... ERROR", 1)[0]] = "ERROR"
        if line.startswith("ERROR:"):
            parts = line.split()
            if len(parts) > 1:
                test_status[parts[1]] = "ERROR"
        if line.startswith("ok") and previous_test is not None:
            test_status[previous_test] = "PASSED"

    return test_status


def parse_log_sympy(log: str) -> Dict[str, str]:
    """Parse SymPy's custom ``bin/test -C --verbose`` output."""
    test_status: Dict[str, str] = {}
    for match in re.findall(r"(_*) (.*)\.py:(.*) (_*)", log):
        test_status[f"{match[1]}.py:{match[2]}"] = "FAILED"
    for raw_line in log.split("\n"):
        line = raw_line.strip()
        if not line.startswith("test_"):
            continue
        test_name = line.split()[0]
        if line.endswith(" E"):
            test_status[test_name] = "ERROR"
        elif line.endswith(" F"):
            test_status[test_name] = "FAILED"
        elif line.endswith(" ok"):
            test_status[test_name] = "PASSED"
    return test_status


def get_eval_tests_report(
    test_output: str,
    repo: str,
    fail_to_pass: List[str],
    pass_to_pass: List[str],
) -> Dict[str, Any]:
    """
    Parse test output and generate evaluation report.
    Based on swebench/harness/grading.py
    """
    # Extract test output between markers
    start_idx = test_output.find(START_TEST_OUTPUT)
    end_idx = test_output.find(END_TEST_OUTPUT)
    
    if start_idx != -1 and end_idx != -1:
        test_log = test_output[start_idx:end_idx]
    else:
        test_log = test_output
    
    # Parse based on repo type.
    if repo == "django/django":
        test_status = parse_log_django(test_log)
    elif repo == "sympy/sympy":
        test_status = parse_log_sympy(test_log)
    else:
        if repo == "matplotlib/matplotlib":
            test_log = test_log.replace("MouseButton.LEFT", "1").replace("MouseButton.RIGHT", "3")
        test_status = parse_log_pytest(test_log)
    
    # Classify results
    results = {
        "FAIL_TO_PASS": {"success": [], "failure": []},
        "PASS_TO_PASS": {"success": [], "failure": []},
    }
    
    def resolve_test_name(test_name: str) -> Optional[str]:
        """Resolve exact IDs plus SWE-bench's truncated parametrized IDs."""
        if test_name in test_status:
            return test_name
        if test_name.count("[") > test_name.count("]"):
            matches = [key for key in test_status if key.startswith(test_name)]
            passing = {"PASSED", "XFAIL"}
            if matches and len({test_status[key] in passing for key in matches}) == 1:
                return matches[0]
        return None

    def test_passed(test_name: str) -> bool:
        resolved = resolve_test_name(test_name)
        return resolved is not None and test_status[resolved] in {"PASSED", "XFAIL"}

    def test_maintained(test_name: str) -> bool:
        resolved = resolve_test_name(test_name)
        return resolved is not None and test_status[resolved] in {"PASSED", "XFAIL", "SKIPPED"}
    
    # Classify FAIL_TO_PASS tests
    for test in fail_to_pass or []:
        if test_passed(test):
            results["FAIL_TO_PASS"]["success"].append(test)
        else:
            results["FAIL_TO_PASS"]["failure"].append(test)
    
    # Classify PASS_TO_PASS tests
    for test in pass_to_pass or []:
        if test_maintained(test):
            results["PASS_TO_PASS"]["success"].append(test)
        else:
            results["PASS_TO_PASS"]["failure"].append(test)
    
    return results



class SWEBenchExecutor:
    """Executes SWE-bench tasks using Docker containers."""

    def __init__(
        self,
        instance: SWEBenchInstance,
        logs_dir: Path,
        timeout: int = 1800,
        env_init: Optional[Dict[str, str]] = None,
    ):
        self.instance = instance
        self.logs_dir = logs_dir
        self.timeout = timeout
        self.env_init = env_init or {}
        
        self.container_id: Optional[str] = None
        self._temp_dir: Optional[Path] = None
        self._repo_path: Optional[str] = None  # Linux path in container, use str not Path
        self._runtime_workspace_dir: Optional[Path] = None
        
        # Create logs directory
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def set_runtime_workspace_directory(self, directory: Path) -> None:
        """Configure a persistent /workspace mount before starting the container."""
        if self.container_id:
            raise RuntimeError("Configure the runtime mount before env.reset()")
        self._runtime_workspace_dir = directory.resolve()

    async def start_container(self):
        """Start Docker container for the SWE-bench instance."""
        # Create temporary directory for workspace
        if self._runtime_workspace_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix=f"swebench_{self.instance.instance_id}_"))
        else:
            self._temp_dir = self._runtime_workspace_dir
            self._temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine image name based on instance_id
        # SWE-bench official images use format: swebench/sweb.eval.x86_64.{owner}_1776_{owner}-{issue}
        # Example: astropy__astropy-12907 -> swebench/sweb.eval.x86_64.astropy_1776_astropy-12907
        # Parse instance_id: "astropy__astropy-12907" -> owner="astropy", issue="12907"
        parts = self.instance.instance_id.split("__")
        owner = parts[0]  # "astropy"
        repo_issue = parts[1] if len(parts) > 1 else self.instance.instance_id  # "astropy-12907"
        image_name = f"swebench/sweb.eval.x86_64.{owner}_1776_{repo_issue}"
        
        logger.info(f"Starting container for {self.instance.instance_id}")
        logger.info(f"Image: {image_name}")
        
        try:
            # Check if image exists locally
            check_result = await asyncio.to_thread(
                subprocess.run,
                ["docker", "images", "-q", image_name],
                capture_output=True,
                text=True,
                timeout=10,
            )
            
            if not check_result.stdout.strip():
                # Pull image
                logger.info(f"Pulling image: {image_name}")
                result = await asyncio.to_thread(
                    subprocess.run,
                    ["docker", "pull", image_name],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if result.returncode != 0:
                    raise RuntimeError(f"Failed to pull image: {result.stderr}")
            
            # A deterministic instance-only name lets concurrent benchmark runs
            # delete each other's live containers. Use a per-container nonce;
            # cleanup already operates on the returned container ID.
            container_name = f"swebench_{self.instance.instance_id}_{uuid.uuid4().hex[:8]}"
            logger.info(f"Container name: {container_name}")
            
            # Run container
            env_args = []
            for key, value in self.env_init.items():
                env_args.extend(["-e", f"{key}={value}"])
            
            cmd = [
                "docker", "run", "-d",
                "--name", container_name,
                "-v", f"{self._temp_dir}:/workspace",
                *env_args,
                image_name,
                "tail", "-f", "/dev/null",  # Keep container running
            ]
            
            result = await asyncio.to_thread(
                subprocess.run,
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Failed to start container: {result.stderr}")
            
            self.container_id = result.stdout.strip()
            logger.info(f"Container started: {self.container_id[:12]}")
            
            # Setup repository in container
            await self._setup_repo()
            
        except Exception as e:
            await self.cleanup()
            raise RuntimeError(f"Failed to start container: {e}") from e

    async def _setup_repo(self):
        """Setup repository at base commit in container."""
        if not self.container_id:
            raise RuntimeError("Container not started")
        
        # SWE-bench official images always place the repository at /testbed
        self._repo_path = "/testbed"
        
        # Checkout base commit
        logger.info(f"Checking out base commit: {self.instance.base_commit}")
        output, exit_code = await self.execute_command(
            f"cd {self._repo_path} && git checkout -f {self.instance.base_commit}"
        )
        if exit_code != 0:
            logger.warning(f"Failed to checkout base commit: {output}")
        
        # Reset any local changes
        await self.execute_command(f"cd {self._repo_path} && git reset --hard HEAD")
        await self.execute_command(f"cd {self._repo_path} && git clean -fd")

    async def execute_command(
        self, 
        command: str, 
        timeout: Optional[int] = None,
        workdir: Optional[str] = None,
    ) -> Tuple[str, int]:
        """Execute command in container.
        
        Uses stdin to pass command to avoid Windows command line length limit (~8191 chars).
        This allows executing commands with large content (e.g., base64 encoded files).
        """
        if not self.container_id:
            raise RuntimeError("Container not started")

        exec_timeout = timeout if timeout is not None else self.timeout
        proc = None
        try:
            # Use -i (interactive) to read command from stdin
            # This bypasses Windows command line length limits
            cmd = ["docker", "exec", "-i"]
            if workdir:
                cmd.extend(["-w", workdir])
            cmd.extend([self.container_id, "bash"])
            
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )

            # Pass command through stdin
            stdout, _ = await asyncio.wait_for(
                proc.communicate(input=command.encode('utf-8')),
                timeout=exec_timeout
            )

            output = stdout.decode("utf-8", errors="replace")
            exit_code = proc.returncode or 0

            return output, exit_code

        except asyncio.TimeoutError:
            return "Command timed out", -1
        except Exception as e:
            return f"Error executing command: {e}", -1
        finally:
            if proc is not None and proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass
                await proc.wait()

    async def apply_patch(self, patch_content: str) -> Tuple[bool, str]:
        """Apply a patch to the repository."""
        if not self.container_id:
            raise RuntimeError("Container not started")
        
        # Write patch to temp file in container
        patch_path = "/tmp/agent_patch.diff"
        
        # Escape patch content for shell
        escaped_patch = patch_content.replace("'", "'\\''")
        output, exit_code = await self.execute_command(
            f"echo '{escaped_patch}' > {patch_path}"
        )
        
        if exit_code != 0:
            return False, f"Failed to write patch: {output}"
        
        # Apply patch
        output, exit_code = await self.execute_command(
            f"cd {self._repo_path} && git apply --check {patch_path}"
        )
        
        if exit_code != 0:
            return False, f"Patch check failed: {output}"
        
        output, exit_code = await self.execute_command(
            f"cd {self._repo_path} && git apply {patch_path}"
        )
        
        if exit_code != 0:
            return False, f"Failed to apply patch: {output}"
        
        return True, "Patch applied successfully"

    async def run_tests(self) -> Tuple[float, Dict[str, Any]]:
        """Run the official version-specific SWE-bench TestSpec and grader."""
        if not self.container_id:
            raise RuntimeError("Container not started")

        from swebench.harness.grading import get_eval_report

        test_spec = make_official_test_spec(self.instance)
        eval_script = make_offline_official_eval_script(test_spec)

        # Save eval script to log for debugging
        eval_script_log = self.logs_dir / "eval.sh"
        with eval_script_log.open("w", encoding="utf-8") as f:
            f.write(eval_script)

        # Record the exact patch being evaluated. ``git add -N`` makes untracked
        # files visible to diff without staging their content as a real commit.
        model_patch, patch_exit_code = await self.execute_command(
            "git add -N . >/dev/null 2>&1 || true; git diff --binary HEAD",
            timeout=120,
            workdir=self._repo_path,
        )
        if patch_exit_code != 0:
            raise RuntimeError(
                f"Failed to extract model patch for official evaluation: {model_patch}"
            )
        patch_log = self.logs_dir / "model_patch.diff"
        patch_log.write_text(model_patch, encoding="utf-8")

        # Write eval script to container and execute
        await self.execute_command(
            f"cat > /eval.sh << 'EOF_EVAL_SCRIPT'\n{eval_script}\nEOF_EVAL_SCRIPT"
        )
        await self.execute_command("chmod +x /eval.sh")

        # Run eval script with extended timeout for test execution
        test_output, exit_code = await self.execute_command(
            "/bin/bash /eval.sh",
            timeout=self.timeout,
        )

        # Save test output to log
        test_output_log = self.logs_dir / "test_output.txt"
        test_output_log.write_text(test_output, encoding="utf-8")

        prediction = {
            "instance_id": self.instance.instance_id,
            "model_name_or_path": "aorchestra-local",
            "model_patch": model_patch if model_patch.strip() else None,
        }
        report_map = get_eval_report(
            test_spec=test_spec,
            prediction=prediction,
            test_log_path=str(test_output_log),
            include_tests_status=True,
        )
        results = official_report_to_results(self.instance.instance_id, report_map)
        results["exit_code"] = exit_code
        reward = float(results["reward"])

        # Save test results to log
        test_log = self.logs_dir / "test_results.log"
        test_log.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return reward, results

    async def get_file_content(self, file_path: str) -> Tuple[str, int]:
        """Read file content from container."""
        return await self.execute_command(f"cat {file_path}")

    async def write_file(self, file_path: str, content: str) -> Tuple[bool, str]:
        """Write content to file in container."""
        # Escape content for shell
        escaped_content = content.replace("'", "'\\''")
        output, exit_code = await self.execute_command(
            f"cat > {file_path} << 'EOFMARKER'\n{content}\nEOFMARKER"
        )
        if exit_code != 0:
            return False, f"Failed to write file: {output}"
        return True, "File written successfully"

    async def list_files(self, directory: str = ".") -> Tuple[str, int]:
        """List files in directory."""
        return await self.execute_command(f"find {directory} -type f -name '*.py' | head -100")

    def get_container_id(self) -> Optional[str]:
        """Get the container ID."""
        return self.container_id

    async def cleanup(self):
        """Clean up container and temporary files."""
        if self.container_id:
            try:
                # Stop and remove container
                await asyncio.to_thread(
                    subprocess.run,
                    ["docker", "rm", "-f", self.container_id],
                    capture_output=True,
                    timeout=30,
                )
                logger.info(f"Container removed: {self.container_id[:12]}")
            except Exception as e:
                logger.warning(f"Failed to remove container: {e}")
            finally:
                self.container_id = None
        
        if self._temp_dir and self._temp_dir.exists() and self._runtime_workspace_dir is None:
            try:
                shutil.rmtree(self._temp_dir)
            except Exception as e:
                logger.warning(f"Failed to remove temp dir: {e}")
            finally:
                self._temp_dir = None
