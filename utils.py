
import numpy as np
import argparse
from copy import *
from typing import List
import time
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal
from torch.distributions import MultivariateNormal
import math
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import carla


argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')

argparser.add_argument(
    '--host',
    metavar='H',
    default='127.0.0.1',

    help='IP of the host server (default: 127.0.0.1)')
argparser.add_argument(
    '-p', '--port',
    metavar='P',
    default=2000,
    type=int,
    help='TCP port to listen to (default: 2000)')
argparser.add_argument(
    '--res',
    metavar='WIDTHxHEIGHT',
    default='1280x720',
    help='Window resolution (default: 1280x720)')
argparser.add_argument(
    '--sync',
    action='store_true',
    help='Synchronous mode execution')
argparser.add_argument(
    '--filter',
    metavar='PATTERN',
    # TODO: change for other vehicle models
    default='vehicle.tesla.model3',
    help='Actor filter (default: "vehicle.*")')
argparser.add_argument(
    '-s', '--seed',
    help='Set seed for repeating executions (default: None)',
    default=1234,
    type=int)
argparser.add_argument(
    '-m', '--map',
    help='Set Different Map for testing: shanghai_intl_circuit, t1_triple, t2_triple, t3, t4',
    default="shanghai_intl_circuit")

args = argparser.parse_args()

args.width, args.height = [int(x) for x in args.res.split('x')]


def get_device():
  if torch.cuda.is_available():
    print("Has GPU")
    return torch.device('cuda')
  else:
    print("Only have CPU")
    return torch.device('cpu')


device = get_device()

def plot(x, y, title = "", fn="", shown=False):
  x = np.array(x)
  y = np.array(y)
  plt.title(title)
  plt.xlabel("Episode")
  plt.ylabel("Average Cumulative Reward")
  plt.plot(x, y, color ="red")
  filename = fn
  plt.savefig(filename)
  if shown:
    plt.show()


def extract_road_boundary(boundary):
    """extract the left and right road boundary"""

    # extract left road boundary
    left = []

    for p in boundary[0]:
        left.append([p.transform.location.x, p.transform.location.y])

    left = np.asarray(left)

    # extract right road boundary
    right = []

    for p in boundary[1]:
        right.append([p.transform.location.x, p.transform.location.y])

    right = np.asarray(right)

    return left, right


def convert_action_type(action):
  # Given an action tensor on gpu, convert it to the proper action accepted by the env
  a = action.cpu().numpy()
  a = a.reshape(a.shape[1])
  return a


def compute_distance(pt1, pt2, target_pt):
    pt1, pt2, target_pt = np.array(pt1), np.array(pt2), np.array(target_pt)
    n = np.abs((pt2[0] - pt1[0]) * (pt1[1] - target_pt[1]) - (pt1[0] - target_pt[0])*(pt2[1] - pt1[1]))
    d = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
    dist = n / d
    return dist.item()

def convert_state_to_tensor(state):
  if state is None:
    return torch.zeros(1, 206)
  
  # Need to consider the case where the state is none
  obs = state[0]
  waypoints = np.array(state[1]).reshape(1, -1)
  
  vel_x, vel_y = state[2].x, state[2].y
  s_x, s_y = state[3].location.x, state[3].location.y
  orientation = np.deg2rad(state[3].rotation.yaw)
  left, right = extract_road_boundary(state[4])
  left = left.reshape(1, -1)
  right = right.reshape(1, -1)
  bd = np.hstack((left, right))
  bd = torch.from_numpy(bd)
  dist = state[5]
  temp_s = torch.tensor([[vel_x, vel_y, s_x, s_y, orientation, dist]])
  result_state = torch.hstack((temp_s, bd)).to(device)
  return result_state


def get_control_from_action(action):
  action = convert_action_type(action)
  control = carla.VehicleControl()
  steer = action.item(0)
  #print("Steer type is: ", type(steer))
  control.steer = steer

  acc = action.item(1)
  if acc > 0:
    control.throttle = acc
    control.brake = 0
  else:
    control.throttle = 0
    control.brake = -acc
  
  return control
