from pathlib import Path
import runpy


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    runpy.run_path(str(repo_root / 'tuna_colab_backtest.py'), run_name='__main__')


if __name__ == '__main__':
    main()
