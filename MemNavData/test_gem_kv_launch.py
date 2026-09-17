"""Verify the selected public storage reaches the original service command."""
import json
from contextlib import contextmanager
import sys

import pytest

from MemNavData import run_gem_memory_navigation as launch


@pytest.mark.parametrize('storage', ['reader_precision', 'int8_storage', 'lossless_bf16'])
def test_explicit_kv_argument_and_receipt(tmp_path, monkeypatch, storage):
    (tmp_path / 'logs').mkdir()
    monkeypatch.setattr(launch, 'open_port', lambda port: False)
    seen = []

    def stop_before_launch(command, **kwargs):
        seen.append(command)
        raise RuntimeError('captured before launching a process')

    monkeypatch.setattr(launch.subprocess, 'Popen', stop_before_launch)
    with pytest.raises(RuntimeError, match='captured before launching'):
        with launch.servers(tmp_path, 'native_interval7', 23340, 23341,
                            dense_window=64, geometry_storage='detector_support',
                            kv_storage=storage):
            raise AssertionError('A test must not start a GPU service')
    command = seen[0]
    assert command[command.index('--memory_kv_storage') + 1] == storage
    assert command[command.index('--memory_mechanism') + 1] == 'native_interval7'
    assert command[command.index('--memory_geometry_storage') + 1] == 'detector_support'
    receipt = json.loads((tmp_path / 'kv_storage_receipt.json').read_text())
    assert receipt['kv_storage'] == storage and receipt['dense_window'] == 64


@pytest.mark.parametrize('mode,window,storage', [
    ('legacy', 64, 'int8_storage'),
    ('native_interval7', 16, 'int8_storage'),
    ('legacy', 64, 'lossless_bf16'),
    ('connected_reciprocal', 64, 'lossless_bf16'),
    ('native_interval7', 16, 'lossless_bf16'),
    ('native_interval7', 64, 'unspecified_auto_backend'),
])
def test_unsupported_storage_contract_stops_before_launch(tmp_path, monkeypatch, mode, window, storage):
    monkeypatch.setattr(launch, 'open_port', lambda port: False)
    with pytest.raises(ValueError, match='native interval7 W64'):
        with launch.servers(tmp_path, mode, 23340, 23341,
                            dense_window=window, kv_storage=storage):
            raise AssertionError('An invalid selection must not launch')


@pytest.mark.parametrize('explicit', [False, True])
def test_cli_options_reach_run_manifest_and_service_selection(tmp_path, monkeypatch, explicit):
    plan = tmp_path / 'plan.json'
    plan.write_text(json.dumps({'cells': [{'index': 0, 'scene': 'launch_test'}]}))
    out = tmp_path / 'run'
    args = ['run_gem_memory_navigation.py', 'run', '--plan', str(plan), '--out', str(out),
            '--mode', 'native_interval7']
    if explicit:
        args += ['--dense-window', '64', '--geometry-storage', 'detector_support',
                 '--kv-storage', 'lossless_bf16']
    expected = dict(dense_window=64, geometry_storage='detector_support', kv_storage='lossless_bf16')
    if not explicit:
        expected = dict.fromkeys(expected)
    seen = []

    @contextmanager
    def stop_before_servers(out, mode, mem_port, nav_port, **options):
        seen.append((mode, options))
        raise RuntimeError('captured selection without starting services')
        yield

    monkeypatch.setattr(sys, 'argv', args)
    monkeypatch.setattr(launch, 'servers', stop_before_servers)
    monkeypatch.setattr(launch.existing, 'history', lambda cell: None)
    monkeypatch.setattr(launch, 'execution_environment', lambda: {})
    with pytest.raises(RuntimeError, match='captured selection'):
        launch.main()
    assert seen == [('native_interval7', expected)]
    assert json.loads((out / 'manifest.json').read_text())['memory_options'] == expected
