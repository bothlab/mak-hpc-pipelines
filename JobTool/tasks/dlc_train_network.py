# -*- coding: utf-8 -*-

import os
import sys
from gconst import SDS_ROOT
from utils import JobTemplateLoader
from argparse import ArgumentParser


DLC_CONFIG_ROOT = os.path.join(SDS_ROOT, 'DLCProjects')


def run(scheduler, data_location, options):
    ''' Create job for DLC network training '''

    if not data_location:
        print('You need to specify a DLC project name!')
        sys.exit(1)

    project_config_fname = os.path.join(DLC_CONFIG_ROOT, data_location, 'config.yaml')
    project_name = data_location.strip('/').replace('/', '_')

    if not os.path.isfile(project_config_fname):
        print('DLC configuration file "{}" does not exist!'.format(project_config_fname))
        sys.exit(4)

    tmpl_loader = JobTemplateLoader()
    job_fname = tmpl_loader.create_job_file('dlc-train-network.tmpl',
                                            'DLCTraining_{}'.format(project_name),
                                            CONFIG_FNAME=project_config_fname)

    # submit the new job
    scheduler.schedule_job(job_fname,
                           name=project_name)
