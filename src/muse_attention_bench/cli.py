"""Local orchestration. A host command never imports candidate Python code."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from .protocol import ROOT, DEFAULT_PROFILE, load_profile, canonical_sha256, verify_vendor


def emit(value: object) -> None:
    print(json.dumps(value, indent=2, default=str))


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description='Muse attention benchmark: documented competition protocol, not the hosted judge')
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('doctor',help='Host prerequisites, no GPU job, image pull or model call')
    q=sub.add_parser('container-doctor',help='Validate the built image and H200 runtime without importing a candidate')
    q.add_argument('--gpu',default='0');q.add_argument('--attest-h200-sxm',action='store_true');q.add_argument('--dry-run',action='store_true')
    sub.add_parser('internal-doctor',help=argparse.SUPPRESS)
    sub.add_parser('verify',help='Verify pinned third-party source hashes')
    q=sub.add_parser('providers',help='Inspect provider configuration without network or inference calls')
    q.add_argument('--config',type=Path,default=ROOT/'configs/providers.example.json')
    q.add_argument('--catalog-plans',action='store_true')
    q.add_argument('--local-context',type=Path)
    q=sub.add_parser('profile',help='Print the immutable profile and its identity');q.add_argument('--profile',type=Path,default=DEFAULT_PROFILE)
    q=sub.add_parser('build-image',help='Build from the pinned public competition image');q.add_argument('--dry-run',action='store_true')
    q=sub.add_parser('evaluate',help='Evaluate one sealed source in a restricted GPU container')
    q.add_argument('candidate',type=Path);q.add_argument('--output',type=Path,required=True)
    q.add_argument('--profile',type=Path,default=DEFAULT_PROFILE);q.add_argument('--gpu',default='0')
    q.add_argument('--correctness-only',action='store_true');q.add_argument('--attest-h200-sxm',action='store_true')
    q.add_argument('--dry-run',action='store_true')
    q=sub.add_parser('internal-evaluate',help=argparse.SUPPRESS)
    q.add_argument('candidate',type=Path);q.add_argument('--output',type=Path,required=True)
    q.add_argument('--profile',type=Path,default=DEFAULT_PROFILE);q.add_argument('--correctness-only',action='store_true')
    q=sub.add_parser('prepare',help='Create a fresh, identical actor workspace for a model')
    q.add_argument('--model',required=True);q.add_argument('--trial-id',required=True);q.add_argument('--root',type=Path,default=ROOT,help='Benchmark checkout root; creates root/trials/TRIAL_ID')
    q.add_argument('--profile',type=Path,default=DEFAULT_PROFILE);q.add_argument('--provider',default='unspecified')
    q.add_argument('--effort',default='unspecified');q.add_argument('--seconds',type=int,default=1800)
    q.add_argument('--diagnostics',type=int,default=3);q.add_argument('--context-budget',type=int)
    q.add_argument('--cohort',default='default');q.add_argument('--hide-baseline',action='store_true')
    for name in ['start','seal']:
        q=sub.add_parser(name,help=f'{name.title()} a prepared trial');q.add_argument('trial',type=Path)
        if name=='seal':q.add_argument('--candidate',type=Path)
    q=sub.add_parser('attach',help='Bind evaluated output and source review to a sealed trial')
    q.add_argument('trial',type=Path);q.add_argument('result',type=Path);q.add_argument('--review',type=Path)
    q=sub.add_parser('report',help='Compare only trials with matching conditions and runtime identities')
    q.add_argument('trials',type=Path,nargs='+')
    q=sub.add_parser('official-check',help='Print the actual hosted OKC check command; --run consumes its check quota')
    q.add_argument('candidate',type=Path);q.add_argument('--run',action='store_true')
    return p


def main(argv: list[str] | None=None) -> int:
    args=parser().parse_args(argv)
    try:
        if args.command=='doctor':
            from .container import host_doctor
            report=host_doctor();report['vendor']=verify_vendor();emit(report)
            return 0 if report['vendor']['ok'] else 1
        if args.command=='verify':
            r=verify_vendor();emit(r);return 0 if r['ok'] else 1
        if args.command=='providers':
            from .providers import main as provider_main
            provider_args=['--config',str(args.config)]
            if args.catalog_plans:provider_args.append('--catalog-plans')
            if args.local_context:provider_args.extend(['--local-context',str(args.local_context)])
            return provider_main(provider_args)
        if args.command=='container-doctor':
            from .container import doctor_command,run_container
            cmd=doctor_command(gpu=args.gpu,attest_sxm=args.attest_h200_sxm)
            if args.dry_run:print(shlex.join(cmd));return 0
            return run_container(cmd,timeout=120)
        if args.command=='internal-doctor':
            if os.environ.get('MUSE_ATTENTION_BENCH_CONTAINER')!='1':raise ValueError('Container runtime preflight is container-only')
            from .gpu import collect_environment
            r=collect_environment(load_profile());emit(r);return 0 if r.get('status')=='ready' else 1
        if args.command=='profile':
            p=load_profile(args.profile);emit({'protocol_sha256':canonical_sha256(p),'profile':p});return 0
        if args.command in {'build-image','evaluate','internal-evaluate'}:
            verified=verify_vendor()
            if not verified['ok']:raise ValueError(f'Vendored source drift: {verified["mismatches"]}')
        if args.command=='build-image':
            from .container import build_command
            cmd=build_command()
            if args.dry_run:print(shlex.join(cmd));return 0
            return subprocess.run(cmd,check=False).returncode
        if args.command=='evaluate':
            from .container import run_command,run_container
            cmd=run_command(args.candidate,args.output,args.profile,gpu=args.gpu,correctness_only=args.correctness_only,attest_sxm=args.attest_h200_sxm)
            if args.dry_run:print(shlex.join(cmd));return 0
            return run_container(cmd,timeout=load_profile(args.profile)['timing']['run_timeout_seconds'])
        if args.command=='internal-evaluate':
            if os.environ.get('MUSE_ATTENTION_BENCH_CONTAINER')!='1':
                raise ValueError('Use evaluate to run inside the GPU container; host candidate imports are disabled')
            from .gpu import evaluate
            r=evaluate(args.candidate,args.output,load_profile(args.profile),correctness_only=args.correctness_only)
            emit(r)
            return 0 if r.get('status') in {'passed','correctness_passed'} else 1
        if args.command=='prepare':
            from .trials import prepare_trial
            if not verify_vendor()['ok']:raise ValueError('Vendored source hashes changed')
            path=prepare_trial(args.root,args.model,args.trial_id,args.profile,provider=args.provider,effort=args.effort,
                wall_clock_seconds=args.seconds,diagnostic_budget=args.diagnostics,context_budget=args.context_budget,
                expose_baseline=not args.hide_baseline,cohort=args.cohort)
            emit({'trial_dir':str(path)});return 0
        if args.command=='start':
            from .trials import mark_trial_started
            emit(mark_trial_started(args.trial));return 0
        if args.command=='seal':
            from .trials import seal_trial
            emit(seal_trial(args.trial,args.candidate));return 0
        if args.command=='attach':
            from .trials import record_result
            emit(record_result(args.trial,args.result,review_path=args.review));return 0
        if args.command=='report':
            from .trials import summarize_trials
            emit(summarize_trials(args.trials));return 0
        if args.command=='official-check':
            if not args.candidate.is_file():raise ValueError('Candidate must exist')
            cmd=['okc','check',str(args.candidate.resolve()),'--json']
            if not args.run:
                print(shlex.join(cmd));print('Requires an already authenticated official CLI and available hosted checks.');return 0
            # No login/terms acceptance is automated by this repository.
            return subprocess.run(cmd,check=False,timeout=1000).returncode
    except (ValueError,FileNotFoundError,FileExistsError,TimeoutError,PermissionError) as e:
        print(f'error: {e}',file=sys.stderr);return 2
    return 1


if __name__=='__main__':raise SystemExit(main())
