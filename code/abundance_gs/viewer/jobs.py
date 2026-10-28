"""Training and evaluation jobs of the viewer: `abgs <stage>` subprocesses (the viewer stays responsive), their log,
progress and outputs.

Progress comes from what the stages already write: tqdm counters (anchor), `[step N]` / `Step:` lines (source),
OUTPUT/live/trace.jsonl (train with --live-every), one line per TEST view (eval). A job runs in its own process
group on the viewer's GPU (CUDA_VISIBLE_DEVICES is inherited) and is stopped with SIGTERM to that group only.
"""
import json
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

from ..pipeline import ENTRY

STAGES = {
    'anchor': 'Stage 1: RGB anchor (FastGS)',
    'source': 'Stage 1: RGB feature refinement',
    'prepare': 'appearance clusters',
    'train': 'Stage 2: spectral decoder',
    'run': 'all stages (abgs run)',
    'eval': 'evaluation',
}
_TQDM = re.compile(r'(\d+)/(\d+) \[')
_STEP = re.compile(r'(?:\[step |Step:\s+)(\d+)')
_VIEW = re.compile(r"^\S+ \{'clean'")


def command(stage, scene, output, steps=None, materials=None, live_every=0, model=None, extra='', fast=False, inputs=None):
    """argv of `abgs <stage>` for a scene (unset options are filled from the scene configuration by the stage)."""
    inputs = inputs or {}
    argv = [stage, '--scene', scene]
    if stage == 'run':
        argv += ['--workdir', output]
        train_args = f'--live-every {live_every}' if live_every else ''
        if steps:
            train_args += f' --steps {steps}'
        if train_args:
            argv += ['--train-args', train_args.strip()]
        if materials:
            argv += ['--materials', materials]
    else:
        argv += ['--output', output]
    if stage == 'anchor' and steps:
        argv += ['--iterations', steps]
    if stage in ('source', 'train') and steps:
        argv += ['--steps', steps]
    if stage == 'source' and inputs.get('anchor_ply'):
        argv += ['--anchor-ply', inputs['anchor_ply']]
    if stage == 'train':
        if inputs.get('source'):
            argv += ['--source', inputs['source']]
        if inputs.get('clusters'):
            argv += ['--clusters', inputs['clusters']]
        if live_every:
            argv += ['--live-every', live_every]
    if stage in ('prepare', 'train') and materials:
        argv += ['--materials', materials]
    if stage == 'eval':
        argv += ['--model', model] + (['--fast'] if fast else [])
    return [str(a) for a in argv] + shlex.split(extra or '')


class Job:
    """One `abgs` subprocess; `lines` keeps the log tail, `progress` (done, total) the last parsed counter."""

    def __init__(self, stage, argv, output, log_dir, total=None):
        self.stage, self.argv, self.output = stage, argv, Path(output)
        self.total = total
        self.lines = deque(maxlen=400)
        self.progress = (0, total or 0)
        self.views = 0
        self.started, self.finished, self.returncode = time.time(), None, None
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log = log_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{stage}.log"
        env = dict(os.environ, PYTHONUNBUFFERED='1')
        self.process = subprocess.Popen([sys.executable, str(ENTRY), *argv], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                        start_new_session=True, env=env, cwd=os.getcwd())
        self.thread = threading.Thread(target=self._read, daemon=True)
        self.thread.start()

    @property
    def command(self):
        return 'abgs ' + ' '.join(shlex.quote(a) for a in self.argv)

    def _read(self):
        with open(self.log, 'w') as log:
            log.write(self.command + '\n')
            buffer = b''
            while True:
                chunk = self.process.stdout.read1(4096) if hasattr(self.process.stdout, 'read1') else self.process.stdout.read(4096)
                if not chunk:
                    break
                buffer += chunk
                parts = re.split(rb'[\r\n]', buffer)
                buffer = parts.pop()
                for raw in parts:
                    line = raw.decode('utf-8', 'replace').rstrip()
                    if line:
                        self._line(line)
                        log.write(line + '\n')
                log.flush()
            if buffer:
                self._line(buffer.decode('utf-8', 'replace'))
                log.write(buffer.decode('utf-8', 'replace') + '\n')
        self.returncode = self.process.wait()
        self.finished = time.time()

    def _line(self, line):
        if self.lines and _TQDM.search(line) and _TQDM.search(self.lines[-1]):
            self.lines[-1] = line                  # one line per progress bar
        else:
            self.lines.append(line)
        m = _TQDM.search(line)
        if m and self.stage in ('anchor', 'run'):
            self.progress = (int(m.group(1)), int(m.group(2)))
        m = _STEP.search(line)
        if m and self.stage in ('source', 'run'):
            self.progress = (int(m.group(1)), self.total or self.progress[1])
        if _VIEW.search(line) and self.stage == 'eval':
            self.views += 1
            self.progress = (self.views, self.total or self.progress[1])

    @property
    def running(self):
        return self.returncode is None

    @property
    def status(self):
        if self.running:
            return 'running'
        return 'finished' if self.returncode == 0 else f'failed ({self.returncode})'

    def stop(self):
        if self.running:
            os.killpg(self.process.pid, signal.SIGTERM)

    # ------------------------------------------------------------ outputs
    def decoder_dir(self):
        return self.output / 'decoder' if self.stage == 'run' else self.output

    def live_trace(self):
        """Loss rows of a training job (OUTPUT/live/trace.jsonl, written by train --live-every)."""
        path = self.decoder_dir() / 'live' / 'trace.jsonl'
        if not path.exists():
            return []
        rows = []
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        return rows

    def latest_model(self):
        """The newest checkpoint of a training job: the final model_ema.pt, else the live preview."""
        for path in (self.decoder_dir() / 'model_ema.pt', self.decoder_dir() / 'live' / 'model_ema.pt'):
            if path.exists():
                return path
        return None

    def trainer_log_progress(self):
        """Stage anchor writes the FastGS trainer's console (tqdm) to OUTPUT/train.log, not to stdout."""
        log = self.output / ('anchor/train.log' if self.stage == 'run' else 'train.log')
        if self.stage not in ('anchor', 'run') or not log.exists():
            return
        with open(log, 'rb') as f:
            f.seek(max(0, log.stat().st_size - 4096))
            found = _TQDM.findall(f.read().decode('utf-8', 'replace'))
        if found:
            self.progress = (int(found[-1][0]), int(found[-1][1]))

    def fraction(self):
        self.trainer_log_progress()
        rows = self.live_trace() if self.stage in ('train', 'run') else []
        if rows:
            return min(1., rows[-1]['step'] / max(rows[-1].get('steps', 1), 1))
        done, total = self.progress
        return min(1., done / total) if total else 0.

    def summary(self):
        elapsed = (self.finished or time.time()) - self.started
        self.trainer_log_progress()
        done, total = self.progress
        rows = self.live_trace() if self.stage in ('train', 'run') else []
        text = f'**{self.stage}** {self.status}, {elapsed:.0f} s  \n`{self.command}`  \nlog `{self.log}`'
        if rows:
            r = rows[-1]
            text += f"  \nupdate {r['step']} / {r.get('steps', '?')}: loss {r['loss']:.4f}, L1 {r['l1']:.4f}, " \
                    f"SAM {r['sam_rad'] * 180 / 3.141592653589793:.2f} deg" + \
                    (f", L_p {r['reg_nll']:.3f} (weight {r['reg_weight']:.3f})" if r.get('reg_nll') is not None else '')
        elif total:
            text += f'  \nprogress {done} / {total}'
        return text


# ---------------------------------------------------------------- evaluation results

def metrics_table(eval_dir, label_space=None):
    """Markdown table of an evaluation directory: spectral metrics (TEST, clean and noisy references), material
    scores (mIoU, accuracy, per-class IoU) and endmember angles."""
    d = Path(eval_dir)
    lines = []
    spectral = d / 'spectral_metrics.json'
    if spectral.exists():
        s = json.loads(spectral.read_text())['summary']
        lines += ['| TEST (macro mean) | PSNR (dB) | SAM (deg) | SAM p95 (deg) |', '|---|---|---|---|']
        for ref in ('clean', 'noisy'):
            if ref in s:
                u = s[ref]['unified_macro_mean']
                lines.append(f"| vs {ref} | {u['hsi_psnr']:.2f} | {u['hsi_sam_deg']:.2f} | {u['hsi_sam_p95_deg']:.2f} |")
        if 'train' in s and 'clean' in s['train']:
            u = s['train']['clean']['unified_macro_mean']
            lines.append(f"| TRAIN vs clean | {u['hsi_psnr']:.2f} | {u['hsi_sam_deg']:.2f} | {u['hsi_sam_p95_deg']:.2f} |")
        lines.append('')
    elif (d / 'status.json').exists():                 # a training directory: TEST metrics at the pack resolution
        m = json.loads((d / 'status.json').read_text()).get('metrics_ema', {})
        lines += ['| metric (EMA, TEST, pack resolution) | value |', '|---|---|'] + [f'| {k} | {v:.4g} |' for k, v in m.items()
                                                                                     if isinstance(v, (int, float))] + ['']
    scores = d / 'benchmark_material_evaluation.json'
    if scores.exists():
        e = json.loads(scores.read_text())
        spaces = e['evaluation']['label_spaces']
        space = label_space if label_space in spaces else next(iter(spaces))
        m = spaces[space]['mapped_dominant_slot_material_agreement']
        lines += [f"**materials** ({space}, TRAIN-calibrated many-to-one map, TEST views): mIoU **{m['foreground_miou']:.3f}**, "
                  f"pixel accuracy {m['foreground_pixel_accuracy']:.3f}", '', '| class | IoU |', '|---|---|']
        lines += [f'| {n} | {v:.3f} |' for n, v in zip(m['class_names'], m['iou_per_class'])]
        try:
            diag = e['endmember_diagnostics']['label_spaces'][space]
            rec = diag['reference_recall_training_radiance_weighted']['weighted_mean_SAM_degrees']
            pre = diag['learned_channel_precision_training_mass_weighted']['weighted_mean_SAM_degrees']
            lines += ['', f'**endmember angles** (deg, to the nearest direct / sky reference): reference recall {rec:.2f}, '
                          f'learned-endmember precision {pre:.2f}']
        except KeyError:
            pass
    return '\n'.join(lines) if lines else f'no evaluation results in `{d}`'
