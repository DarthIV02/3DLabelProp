#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
from auxiliary.laserscanvis import LaserScanVis
from auxiliary.dataset import SemKITTI_sk
from torch.utils.data import DataLoader

import csv
import datetime

def collate_fn_BEV(data):
    points = data[0][0]
    remissions = data[0][1]
    labels = data[0][2]
    viridis_colors = data[0][3]
    sem_label_colors = data[0][4]
    return points, remissions, labels, viridis_colors, sem_label_colors

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./visualize.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset to visualize. No Default',
  )
  parser.add_argument(
      '--config', '-c',
      type=str,
      required=False,
      default="config/semantic-kitti.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )
  parser.add_argument(
      '--sequence', '-s',
      type=str,
      default="00",
      required=False,
      help='Sequence to visualize. Defaults to %(default)s',
  )
  parser.add_argument(
      '--ignore_semantics', '-i',
      dest='ignore_semantics',
      default=False,
      action='store_true',
      help='Ignore semantics. Visualizes uncolored pointclouds. '
      'Defaults to %(default)s',
  )
  parser.add_argument(
    '--log_path',
    dest='log_path',
    default=None,
    required=False,
    help='Path to log visualization time to .csv file. Requires log_data argument. '
    'Defaults to %(default)s',
  )
  parser.add_argument(
    '--log_data',
    dest='log_data',
    default=False,
    action='store_true',
    help='Records and stores runtime for scanning and plotting visualizaions. Defaults to False'
  )
  parser.add_argument(
    '--print_data',
    dest='print_data',
    default=False,
    action='store_true',
    help='Enables printing of runtime data to terminal. Defaults to False.'
  )
  parser.add_argument(
    '--enable_auto',
    dest='enable_auto',
    default=False,
    required=False,
    action='store_true',
    help='Enables instantaneous visualization of files for collecting'
    ' large amounts of visualization time measurements',
  )
  parser.add_argument(
    '--shuffle',
    dest='shuffle',
    default=False,
    required=False,
    action='store_true',
    help='Shuffles scans before visualization. Defaults to False'
  )
  FLAGS, unparsed = parser.parse_known_args()

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Dataset", FLAGS.dataset)
  print("Config", FLAGS.config)
  print("Sequence", FLAGS.sequence)
  print("ignore_semantics", FLAGS.ignore_semantics)
  print("log_path", FLAGS.log_path)
  print("log_data", FLAGS.log_data)
  print("print_data", FLAGS.print_data)
  print("enable_auto", FLAGS.enable_auto)
  print("shuffle", FLAGS.shuffle)
  print("*" * 80)

  # prevent updating log_path if log_data not used
  if FLAGS.log_path and not FLAGS.log_data:
    print("Must pass log_data argument to specify log_path")
    quit()

  # Require log_path argument if log_data is passed
  if FLAGS.log_data and not FLAGS.log_path:
    print("Must specify a log_path")
    quit()

  # open config file
  try:
    print("Opening config file %s" % FLAGS.config)
    CFG = yaml.safe_load(open(FLAGS.config, 'r'))
  except Exception as e:
    print(e)
    print("Error opening yaml file.")
    quit()

  # fix sequence name
  FLAGS.sequence = '{0:02d}'.format(int(FLAGS.sequence))

  # does sequence folder exist?
  scan_paths = os.path.join(FLAGS.dataset, "sequences",
                            FLAGS.sequence, "velodyne")
  if os.path.isdir(scan_paths):
    print(f"Sequence folder {scan_paths} exists! Using sequence from {scan_paths}")
  else:
    print(f"Sequence folder {scan_paths} doesn't exist! Exiting...")
    quit()

  color_dict = CFG["color_map"]
  mydataset = SemKITTI_sk(data_path = "dataset/sequences", 
                          imageset="train", 
                          label_mapping="config/semantic-kitti-00.yaml",
                          sem_color_dict=color_dict,
                          percentLabels=1)
  
  dataloader = DataLoader(mydataset, batch_size=1, shuffle=FLAGS.shuffle, collate_fn=collate_fn_BEV, num_workers=0)

  # create a visualizer
  # TODO update class variables
  vis = LaserScanVis(dataloader=dataloader,
                    dataset=mydataset,
                    enable_auto=FLAGS.enable_auto,
                    semantics=(not FLAGS.ignore_semantics),
                    verbose_runtime=FLAGS.print_data)

  # print instructions
  print("To navigate:")
  print("\tn: next (next scan)")
  print("\tq: quit (exit program)")
  
  # if log_data flag is false, do not specify filewriter
  if not FLAGS.log_data:
    # run visualizer
    vis.run()
    quit()

  # if log_data flag is true, open csv file for writing
  now = datetime.datetime.now()
  with open(FLAGS.log_path + '/{0}.csv'.format('{0}'.format(now).replace(" ", "_").replace(":", "-")[:19]),
            'w', newline='') as csvfile:
    print("Saving runtime data to ", csvfile.name)
    fieldNames = ['Points','LoadData', 'PlotRaw','PlotSem']
    writer = csv.DictWriter(csvfile, fieldnames=fieldNames)
    writer.writeheader()
    vis.csvwriter = writer

    # run the visualizer
    vis.run()
