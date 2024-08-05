import os
from pathlib import Path


class Globals:
    # SDS root directory
    SDS_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..')
    )  # '/mnt/sds-hd/sd20h003/<user>/'


PIPELINES_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'Pipelines')
    )

# SLURM job template directory
SLURM_TEMPLATE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), 'job-templates')
)

# Directory from which jobs are scheduled
JOB_SCHEDULE_DIR = os.path.join(str(Path.home()), 'job-schedule')

# Temporary directory for automatic job creation
JOB_TEMP_DIR = os.path.join(JOB_SCHEDULE_DIR, '_tmp')

USE_SLURM = True
