#!/usr/bin/env python3
'''Aggregate Langfuse url-benchmark runs into a method x type score matrix.

For every dataset run on dataset `urls`, walks its run items, fetches each
trace + the source dataset item, then groups scores by (method, type) where:
  - method comes from trace.output.method (or run name as fallback)
  - type comes from datasetItem.metadata (string)

Outputs both a markdown matrix and a per-trace CSV for drill-down.

Requires LANGFUSE_PUBLIC_KEY/SECRET_KEY/BASE_URL in env (loaded from .env if present).
'''
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / '.env'

SCORE_NAMES = ('title_similarity', 'summary_similarity', 'image_similarity')


def load_env() -> None:
    if not ENV_FILE.exists():
        return
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def lf(*args: str) -> dict[str, Any] | None:
    cmd = ['npx', '-y', 'langfuse-cli@latest', 'api', *args, '--json']
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return None
    try:
        return json.loads(p.stdout).get('body')
    except json.JSONDecodeError:
        return None


def fetch_runs(dataset: str, limit: int) -> list[dict[str, Any]]:
    body = lf('datasets', 'get-get-runs', dataset, '--limit', str(limit))
    return (body or {}).get('data', []) if body else []


def fetch_run_items(dataset: str, run_name: str) -> list[dict[str, Any]]:
    body = lf('datasets', 'get-get-run', dataset, run_name)
    return (body or {}).get('datasetRunItems', []) if body else []


def fetch_trace(trace_id: str) -> dict[str, Any] | None:
    return lf('traces', 'get', trace_id)


def fetch_dataset_item(item_id: str) -> dict[str, Any] | None:
    return lf('dataset-items', 'get', item_id)


def derive_method(trace: dict[str, Any], run_name: str) -> str:
    out = trace.get('output')
    if isinstance(out, dict) and out.get('method'):
        return str(out['method'])
    # fallback: run name prefix (computer-use, cf-browser, url-resolver, url-context)
    lower = run_name.lower()
    for m in ('cf-browser', 'computer-use', 'url-resolver', 'url-context'):
        if m in lower:
            return m
    return 'unknown'


def derive_type(item: dict[str, Any]) -> str:
    meta = item.get('metadata')
    if isinstance(meta, str):
        return meta
    if isinstance(meta, dict):
        for k in ('type', 'category', 'kind'):
            v = meta.get(k)
            if isinstance(v, str):
                return v
    return 'unknown'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='urls')
    ap.add_argument('--limit', type=int, default=100)
    ap.add_argument('--run-prefix', default='full-',
                    help='Only include runs whose name starts with this prefix (default: full-)')
    ap.add_argument('--latest-per-method', action='store_true', default=True,
                    help='Keep only the most recent run per method (on by default)')
    ap.add_argument('--no-latest-per-method', dest='latest_per_method', action='store_false',
                    help='Include all matching runs instead of latest per method')
    ap.add_argument('--out-csv', type=Path, default=ROOT / 'scripts' / 'benchmark_report.csv')
    ap.add_argument('--cache', type=Path, default=ROOT / 'scripts' / '.benchmark_cache.json')
    ap.add_argument('--no-cache', action='store_true')
    args = ap.parse_args()

    load_env()
    if not os.environ.get('LANGFUSE_PUBLIC_KEY'):
        print('LANGFUSE_PUBLIC_KEY not set', file=sys.stderr)
        return 1

    cache: dict[str, Any] = {}
    if not args.no_cache and args.cache.exists():
        try:
            cache = json.loads(args.cache.read_text())
        except json.JSONDecodeError:
            cache = {}

    rows: list[dict[str, Any]] = []
    item_type_cache: dict[str, str] = cache.get('item_types', {})

    runs = fetch_runs(args.dataset, args.limit)
    if args.run_prefix:
        runs = [r for r in runs if r.get('name', '').startswith(args.run_prefix)]

    # Run names look like "full-{tag}-{timestamp}"; second segment ({tag}) identifies the method.
    if args.latest_per_method:
        latest: dict[str, dict[str, Any]] = {}
        for r in sorted(runs, key=lambda x: x.get('createdAt', '')):
            parts = r.get('name', '').split('-', 2)
            key = parts[1] if len(parts) >= 2 else r.get('name', '')
            latest[key] = r
        runs = list(latest.values())

    print(f'fetched {len(runs)} runs (prefix={args.run_prefix!r}, latest_per_method={args.latest_per_method})', file=sys.stderr)

    for ri, run in enumerate(runs, 1):
        run_name = run['name']
        run_items = fetch_run_items(args.dataset, run_name)
        for item in run_items:
            trace_id = item.get('traceId')
            ds_item_id = item.get('datasetItemId')
            if not trace_id or not ds_item_id:
                continue

            cache_key = f'trace:{trace_id}'
            trace = cache.get(cache_key) if not args.no_cache else None
            if trace is None:
                trace = fetch_trace(trace_id)
                if trace is not None and not args.no_cache:
                    cache[cache_key] = trace

            if not trace:
                continue

            type_str = item_type_cache.get(ds_item_id)
            if type_str is None:
                ds_item = fetch_dataset_item(ds_item_id)
                type_str = derive_type(ds_item or {})
                item_type_cache[ds_item_id] = type_str

            scores = {s.get('name'): s.get('value') for s in trace.get('scores') or []}
            method = derive_method(trace, run_name)

            url = trace.get('input')
            if isinstance(url, dict):
                url = url.get('url') or url.get('input') or json.dumps(url)
            elif not isinstance(url, str):
                url = json.dumps(url) if url else ''

            rows.append({
                'run': run_name,
                'method': method,
                'type': type_str,
                'url': url,
                'trace_id': trace_id,
                'title': scores.get('title_similarity'),
                'summary': scores.get('summary_similarity'),
                'image': scores.get('image_similarity'),
            })
            print(f'  [{ri}/{len(runs)}] {method:12s} {type_str[:30]:30s} t={scores.get("title_similarity")}', file=sys.stderr)

    if not args.no_cache:
        cache['item_types'] = item_type_cache
        args.cache.write_text(json.dumps(cache))

    # Write per-trace CSV
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['run', 'method', 'type', 'url', 'trace_id', 'title', 'summary', 'image'])
        w.writeheader()
        w.writerows(rows)
    print(f'\nwrote {len(rows)} rows to {args.out_csv}', file=sys.stderr)

    # Build method x type matrix (mean of each score)
    by_cell: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for r in rows:
        by_cell[(r['method'], r['type'])].append(r)

    methods = sorted({r['method'] for r in rows})
    types = sorted({r['type'] for r in rows})

    print()
    print('# URL benchmark report')
    print()
    print(f'Source: dataset={args.dataset}, runs={len(runs)}, traces={len(rows)}')
    print()

    for score_key, score_label in [('title', 'Title similarity'),
                                    ('summary', 'Summary similarity'),
                                    ('image', 'Image similarity')]:
        print(f'## {score_label} (mean | n)')
        print()
        header = '| type \\ method |' + ' | '.join(methods) + ' |'
        sep = '|' + '|'.join(['---'] * (len(methods) + 1)) + '|'
        print(header)
        print(sep)
        for t in types:
            cells = [t]
            for m in methods:
                vals = [r[score_key] for r in by_cell.get((m, t), []) if r[score_key] is not None]
                if vals:
                    cells.append(f'{statistics.mean(vals):.3f} (n={len(vals)})')
                else:
                    cells.append('—')
            print('| ' + ' | '.join(cells) + ' |')
        print()

    print('## Trace count by (method, type)')
    print()
    print('| type \\ method |' + ' | '.join(methods) + ' | total |')
    print('|' + '|'.join(['---'] * (len(methods) + 2)) + '|')
    for t in types:
        cells = [t]
        total = 0
        for m in methods:
            n = len(by_cell.get((m, t), []))
            total += n
            cells.append(str(n) if n else '—')
        cells.append(str(total))
        print('| ' + ' | '.join(cells) + ' |')

    return 0


if __name__ == '__main__':
    sys.exit(main())
