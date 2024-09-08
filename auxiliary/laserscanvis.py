#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import vispy
from vispy.scene import visuals, SceneCanvas
import numpy as np
from matplotlib import pyplot as plt
import time

class LaserScanVis:
  """Class that creates and handles a visualizer for a pointcloud"""

  def __init__(self, dataloader, dataset, enable_auto = False, semantics=True, verbose_runtime=False):
    self.dataloader=enumerate(dataloader)
    self.dataset=dataset
    self.semantics = semantics
    self.csvwriter = None
    self.verbose_runtime = verbose_runtime

    self.DEBUG_AUTO = enable_auto
    # number of scans to visualize during auto visualization
    self.DEBUG_AUTO_VALUE = 100

    # time (s) between each individual scan for timer-based visualization
    self.TIME_INTERVAL = 0.5

    # sanity check
    if not self.semantics and self.instances:
      print("Instances are only allowed in when semantics=True")
      raise ValueError

    self.reset()
    self.update_scan()

  # wrapper method for clock event callback
  def next_scan(self, event):
    self.update_scan()

  def reset(self):
    """ Reset. """
    # last key press (it should have a mutex, but visualization is not
    # safety critical, so let's do things wrong)
    self.action = "no"  # no, next, back, quit are the possibilities

    # new canvas prepared for visualizing data
    self.canvas = SceneCanvas(keys='interactive', show=True, size=(1600, 600))
    # interface (n next, b back, q quit, very simple)
    self.canvas.events.key_press.connect(self.key_press)
    self.canvas.events.draw.connect(self.draw)
    
    #Change camera speed
    self.clock = vispy.app.timer.Timer(interval=self.TIME_INTERVAL, connect=self.next_scan)

    # grid
    self.grid = self.canvas.central_widget.add_grid()
    zoom = 10.0

    # laserscan part
    self.scan_view = vispy.scene.widgets.ViewBox(
        border_color='white', parent=self.canvas.scene)
    self.grid.add_widget(self.scan_view, 0, 0)
    self.scan_vis = visuals.Markers()
    self.scan_view.camera = vispy.scene.cameras.turntable.TurntableCamera(scale_factor = zoom)
    self.scan_view.add(self.scan_vis)
    visuals.XYZAxis(parent=self.scan_view.scene)
    # add semantics
    if self.semantics:
      print("Using semantics in visualizer")
      self.sem_view = vispy.scene.widgets.ViewBox(
          border_color='white', parent=self.canvas.scene)
      self.grid.add_widget(self.sem_view, 0, 1)
      self.sem_vis = visuals.Markers()
      self.sem_view.camera = vispy.scene.cameras.turntable.TurntableCamera(scale_factor = zoom)
      self.sem_view.add(self.sem_vis)
      visuals.XYZAxis(parent=self.sem_view.scene)
      # synchronize raw and semantic cameras
      self.sem_view.camera.link(self.scan_view.camera)    

  def get_mpl_colormap(self, cmap_name):
    cmap = plt.get_cmap(cmap_name)

    # Initialize the matplotlib color map
    sm = plt.cm.ScalarMappable(cmap=cmap)

    # Obtain linear color range
    color_range = sm.to_rgba(np.linspace(0, 1, 256), bytes=True)[:, 2::-1]

    return color_range.reshape(256, 3).astype(np.float32) / 255.0

  #@getTime
  def update_scan(self):
    # record start time
    start = time.time()

    # iterate loader to obtain datapoints and color maps
    _, (points, _, _, viridis_colors, sem_label_color) = next(self.dataloader)

    # record end loading time
    load = time.time()

    self.scan_vis.set_data(points,
                           face_color=viridis_colors[..., ::-1],
                           edge_color=viridis_colors[..., ::-1],
                           size=1)
    scan_data = time.time()
    # plot semantics
    if self.semantics:
      self.sem_vis.set_data(points,
                            face_color=sem_label_color[..., ::-1],
                            edge_color=sem_label_color[..., ::-1],
                            size=1)
    sem_data = time.time()

    if self.verbose_runtime:
      print("Visualizing {0} points".format(len(points)))
      print("""Time Elapsed: 
            Loading the data:\t{0}
            Plotting Raw:\t\t{1}
            Plotting Semantic:\t{2}
            Total time:\t\t{3}"""
            .format(load-start, scan_data-load, sem_data-scan_data, sem_data-start))
    
    if not(self.csvwriter == None):
      self.csvwriter.writerow({
        'Points':len(points),
        'LoadData':load-start,
        'PlotRaw':scan_data-load,
        'PlotSem':sem_data-scan_data})

  # interface
  def key_press(self, event):
    #if self.DEBUG_AUTO:
    #  return
    self.canvas.events.key_press.block()
    if event.key == 'N' and not self.clock.running:
      self.update_scan()
    elif event.key == 'Q' or event.key == 'Escape':
      self.destroy()
    elif event.key == ' ':
      if not self.clock.running:
        self.clock.start()
      else:
        self.clock.stop()
    elif event.key == 'D' and self.DEBUG_AUTO:
      for i in range(self.DEBUG_AUTO_VALUE):
        self.update_scan()

  def draw(self, event):
    if self.canvas.events.key_press.blocked():
      self.canvas.events.key_press.unblock()

  def destroy(self):
    # destroy the visualization
    self.canvas.close()
    vispy.app.quit()

  def run(self):
    if self.DEBUG_AUTO and not self.csvwriter:
      print("Error: CSV writer required for auto visualization. log_data=True and log_path must be specified")
      raise ValueError
    vispy.app.run()
