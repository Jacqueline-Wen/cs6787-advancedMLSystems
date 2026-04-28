### CS 6787 Final Project: Communication-Efficient SGD

- install all dependencies via `pip install -r requirements.txt`
- baseline: run_baseline.py
- simple multi-process prototype: run_sync.py

Command for running all 4 tests:
`mkdir -p results && \
python run_baseline.py --dataset mnist --batch-size 64 --lr 0.01 --epochs 10 --seed 42 --save-metrics results/baseline.json && \
python run_sync.py --dataset mnist --batch-size 64 --lr 0.01 --epochs 10 --seed 42 --workers 4 --save-metrics results/sync.json && \
python run_local_sgd.py --dataset mnist --batch-size 64 --lr 0.01 --epochs 10 --seed 42 --workers 4 --sync-every-h 10 --save-metrics results/local.json && \
python run_async.py --dataset mnist --batch-size 64 --lr 0.01 --epochs 10 --seed 42 --workers 4 --save-metrics results/async.json
`