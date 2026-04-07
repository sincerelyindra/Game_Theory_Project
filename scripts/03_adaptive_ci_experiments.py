from pathlib import Path
import runpy


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(repo_root / 'part_c_adaptive_ci_experiment.py'), run_name='__main__')


if __name__ == '__main__':
    main()
