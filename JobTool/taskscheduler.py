
import os
import stat
import sys
import shutil
import subprocess
import platform
from rich import print
from rich.table import Table
from rich import box
from gconst import JOB_SCHEDULE_DIR
from utils import random_string


class TaskScheduler:
    def __init__(self):
        self._run_local = False
        self._dry_run = False
        self._reset_jobs_table()

    def check_prerequisites(self):
        if not self._run_local and not shutil.which('sbatch'):
            print('SLURM batch job submission tool not found. Are we running on the HPC?', file=sys.stderr)
            return False
        return True

    def _reset_jobs_table(self):
        self._jobs_table = Table(box=box.MINIMAL)
        self._jobs_table.add_column('Job Name')
        self._jobs_table.add_column('Status')

    @property
    def run_local(self) -> bool:
        return self._run_local

    @run_local.setter
    def run_local(self, v: bool):
        self._run_local = v

    @property
    def dry_run(self) -> bool:
        return self._dry_run

    @dry_run.setter
    def dry_run(self, v: bool):
        self._dry_run = v

    def _schedule_job_slurm(self, batch_fname, name):
        ''' Submit a new SLURM job '''

        os.makedirs(JOB_SCHEDULE_DIR, exist_ok=True)
        # actually submit the job
        print('SUBMIT: {}'.format(name))
        subprocess.check_call(['sbatch', batch_fname], cwd=JOB_SCHEDULE_DIR)
        self._jobs_table.add_row(name, '[[purple]⏲ Submitted')

    def _execute_job_immediately(self, batch_fname, name):
        from rich.live import Live
        from rich.table import Table

        table = Table(box=None, pad_edge=False, show_header=False)
        table.add_row('• {}'.format(name), '[purple]⏲ Running')
        with Live(table, refresh_per_second=4) as live:
            live.update(table)

            tmpdir = '/tmp/scratch/{}'.format(random_string(4))

            envp = os.environ.copy()
            envp['JOB_NO_SLURM'] = 'true'
            envp['SLURM_JOB_NAME'] = name
            envp['SLURM_SUBMIT_DIR'] = JOB_SCHEDULE_DIR
            envp['SLURMD_NODENAME'] = platform.node()
            envp['SLURM_JOB_ID'] = 'None'
            envp['SLURM_JOB_NUM_NODES'] = '1'
            envp['SLURM_NTASKS'] = '1'
            envp['TMPDIR'] = tmpdir

            os.makedirs(tmpdir, exist_ok=True)
            try:
                proc = subprocess.run([batch_fname], cwd=JOB_SCHEDULE_DIR, env=envp, check=False)
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)

            table = Table(box=None, pad_edge=False, show_header=False)
            if proc.returncode == 0:
                table.add_row('• {}'.format(name), '[green]✓ Done')
                self._jobs_table.add_row(name, '[green]✓ Success')
            else:
                table.add_row('• {}'.format(name), '[red]✗ Failed')
                self._jobs_table.add_row(name, '[red]✗ Failed')
            live.update(table)


    def schedule_job(self, batch_fname, name):
        ''' Plan running a new job. If we don't have SLURM, the job may be executed immediately. '''

        # make sure job is executable
        st = os.stat(batch_fname)
        os.chmod(batch_fname, st.st_mode | stat.S_IEXEC)

        if self._dry_run:
            # print what we would do
            print('WOULD SUBMIT: [italic]{} ({})[/italic]'.format(name, batch_fname))
            self._jobs_table.add_row(name, '[magenta]Prepared')
        elif self._run_local:
            self._execute_job_immediately(batch_fname, name)
        else:
            self._schedule_job_slurm(batch_fname, name)

    def mark_skipped_job(self, name, reason=None):
        if reason:
            self._jobs_table.add_row(name, '[yellow]⏏ Skip: {}'.format(reason))
        else:
            self._jobs_table.add_row(name, '[yellow]⏏ Skipped')

    def print_jobs_summary(self):
        ''' Print the SLURM job queue or jobs table for the current user '''
        print(self._jobs_table)
        self._reset_jobs_table = Table(box=box.MINIMAL)

        if not self._run_local:
            print()
            subprocess.call(['squeue', '-l'])
