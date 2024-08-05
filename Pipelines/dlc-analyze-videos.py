#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess
from glob import glob
from pathlib import Path

import torch
import deeplabcut as dlc
from utils.messages import (
    print_info,
    print_task,
    print_warn,
    print_error,
    print_header,
    print_section,
)


def run(options):
    if not options.config:
        print_error('No DLC config file selected!')
        sys.exit(1)
    if not options.videos:
        print_error('No video files set.')
        sys.exit(1)
    config_fname = options.config
    video_fnames = options.videos
    dest_dir = options.dest_dir if options.dest_dir else None

    print_header('Analyzing videos (via {})'.format(config_fname.replace('/mnt/sds-hd/', '', 1)))
    for vf in video_fnames:
        print_info('Video: {}'.format(vf.replace('/mnt/sds-hd/', '', 1)))
    print()

    print_info(f'PyTorch version: {torch.__version__}')
    if torch.cuda.is_available():
        print_info(f'CUDA version: {torch.version.cuda}')
    if torch.backends.mps.is_available():
        print_info(f'ROCm version: {torch.version.hip}')
    print_info(f'DeepLabCut version: {dlc.__version__}')
    sys.stdout.flush()

    print_section('Analyzing Videos')
    dlc.analyze_videos(config_fname, video_fnames, destfolder=dest_dir)

    print_section('Creating Labeled Videos')
    dlc.create_labeled_video(config_fname, video_fnames, destfolder=dest_dir)

    video_dir = Path(video_fnames[0]).parents[0]
    if dest_dir is None:
        dest_dir = video_dir

    result_videos = glob(os.path.join(dest_dir, '*labeled.mp4'))
    if len(result_videos) != 1:
        print_error('Unexpected number of result videos found - skipping reencoding step.')
    else:
        print_section('Re-encoding overview video for smaller size')
        result_src_video = os.path.join(dest_dir, 'dlc-temp.mp4')
        if os.path.lexists(result_src_video):
            os.remove(result_src_video)
        os.rename(result_videos[0], result_src_video)
        result_dest_video = '{}.mkv'.format(result_videos[0].replace('.mkv', '').replace('.mp4', ''))

        subprocess.run(
            [
                'ffmpeg',
                '-nostats',
                '-nostdin',
                '-hide_banner',
                '-y',
                '-i',
                result_src_video,
                '-c:v',
                'libsvtav1',
                '-b:v',
                '0',
                '-g',
                '300',
                '-qp',
                '50',
                '-preset',
                '6',
                '-c:a',
                'copy',
                result_dest_video,
            ],
            check=True,
        )
        os.remove(result_src_video)

    print_info('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze videos with a DLC model.')
    parser.add_argument('-c', '--config', action='store', dest='config', help='Path to the DLC config file.')
    parser.add_argument(
        '-d', '--destdir', action='store', dest='dest_dir', help='Destination to store the analyzed data in.'
    )
    parser.add_argument('videos', action='store', nargs='+', help='The video files to analyze.')

    args = parser.parse_args(sys.argv[1:])
    run(args)
