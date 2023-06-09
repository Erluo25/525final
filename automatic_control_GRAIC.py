#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Yan's Note: GRAIC 2023 -- NO ROS VERSION"""

from __future__ import print_function

#import argparse
import collections
import datetime
import glob
import logging
import math
import os
from turtle import right
import numpy.random as random
import re
import sys
import weakref

import pickle
import time
import subprocess
import multiprocessing
from utils import *

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

#import carla
from carla import ColorConverter as cc

# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, args):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, args):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'ego_vehicle')
        if blueprint.has_attribute('color'):
            # color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', "0,255,0")
        
        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            # TODO: Hardcoded starting point, to be changed to 1st wp in waypoint pickle file
            if args.map == 'shanghai_intl_circuit':
                lower = - 1
                upper = 1
                x = random.uniform(lower, upper)
                y = random.uniform(lower, upper)
                spawn_point = carla.Transform(carla.Location(90.0 + x, 93.0 + y, 1.0), carla.Rotation(0, 0, 0))
            elif args.map == 't1_triple':
                spawn_point = carla.Transform(carla.Location(153.0, -12.0, 1.0), carla.Rotation(0, 180, 0))
            elif args.map == 't2_triple':
                spawn_point = carla.Transform(carla.Location(85.0, -105.0, 1.0), carla.Rotation(0, 135, 0))
            elif args.map == 't3':
                spawn_point = carla.Transform(carla.Location(70.0, -115.0, 1.0), carla.Rotation(0, 16.75, 0))
            elif args.map == 't4':
                spawn_point = carla.Transform(carla.Location(155.0, -96.0, 1.0), carla.Rotation(0, 50, 0))
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)

        if self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud, args)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Welcome to GRAIC Competition", seconds=2.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self.has_collision = None
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        collision_info = world.collision_sensor.get_last_collision_frame()
        max_col = max(1.0, max(collision), collision_info[1])
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
       
        #print("max col is: ", max_col)
        if self.frame == collision_info[0]:
            self.has_collision = (collision_info[0], (collision_info[1] / max_col)) 
        #print("self frame is: ", self.frame)
        #print("LAST CH IS: ", ch)
        #print("Last collision is: ", collision[-1])
        #print("Last collision history is: ", colhist[-1])
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud, args):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

        self.prev_collision_actor_id = []
        self.args = args

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history
    
    def get_last_collision_frame(self):
        if len(self.history) != 0:
            return self.history[-1]
        else:
            return (-1, -1)
    

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)
        
        if event.other_actor.id not in self.prev_collision_actor_id:
            self.prev_collision_actor_id.append(event.other_actor.id)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        # self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(
                carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

#===============================================================================
# ------ environment initialization --------------------------------------------
#===============================================================================
class RACE_ENV():
    def __init__(self, args, collision_weight, distance_weight, center_line_weight, render=False, round_precision=2, stuck_counter_limit=20):
        self.args = args
        self.render=render
        self.collision_weight = collision_weight
        self.distance_weight = distance_weight
        self.center_line_weight = center_line_weight
        self.round_precision = round_precision
        self.stuck_counter_limit = stuck_counter_limit
        if args.seed:
            random.seed(args.seed)
        
        pygame.init()
        pygame.font.init()
        
        client = carla.Client(args.host, args.port)
        client.set_timeout(4.0)
        self.client = client

        #self.traffic_manager = self.client.get_trafficmanager()
        self.sim_world = self.client.load_world(args.map)
        
        self.hud = HUD(args.width, args.height)
        self.world = World(client.get_world(), self.hud, args)

        if args.sync:
            print("SYNC MODE")
            settings = self.sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            self.sim_world.apply_settings(settings)

            #self.traffic_manager.set_synchronous_mode(True)
            #self.traffic_manager.set_random_device_seed(args.seed) # define TM seed for determinism
        
        # TODO: Change track name to a parameter
        self.original_wps = pickle.load(open("./waypoints/{}".format(args.map), 'rb'))

        if self.render is True:
            self.display = pygame.display.set_mode(
                (args.width, args.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
        else:
            self.display = None
    
    def close(self):
        if self.world is not None:
            #settings = self.world.world.get_settings()
            #settings.synchronous_mode = False
            #settings.fixed_delta_seconds = None
            #self.world.world.apply_settings(settings)
            #self.traffic_manager.set_synchronous_mode(True)
            self.world.destroy()
            print("The world has been destoryed")
        pygame.quit()

    # Return state, info
    def reset(self):
        if self.world is not None:
            self.world.destroy()
        
        try:
            # Set the basic parameters of the environment
            self.world.restart(self.args)
            self.waypoints = self.original_wps[1:]
            self.idx = 0 # The waypoint index
            self.clock = pygame.time.Clock()
            self.total_score = 0
            self.start = True
            self.err = None

            # Pre-compute the waypoint distances
            total_dist_to_go = 0
            cumulative_dist = [0]
            for i in range(0, len(self.waypoints) - 1):
                x1, y1, _ = self.waypoints[i]
                x2, y2, _ = self.waypoints[i + 1]
                temp_dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                total_dist_to_go += temp_dist
                cumulative_dist.append(total_dist_to_go)
            assert len(cumulative_dist) == len(self.waypoints), "Pre-compute distance error"
            cumulative_dist = np.array(cumulative_dist)
            cumulative_dist = total_dist_to_go - cumulative_dist
            self.cumulative_dist = cumulative_dist
            self.prev_remain_dist = None
            
            self.prev_acc = 0
            self.prev_steer = 0
            
            state, _, terminated, truncated = self.get_current_state()
            if terminated or truncated is True:
                self.err = "Can not terminate at start"
                print(self.err)
                return None, self.err
            self.prev_dist = round(state[5], self.round_precision)
            self.stuck_counter = 0
            return state, None

        except Exception as err:
            print("Unexpected during environment initialization", err)
            self.err = err
            return None, self.err
    
    # Trying to return state, reward, terminated, distance
    def get_current_state(self):
        self.clock.tick()
        
        self.world.world.tick()
        #if self.args.sync:
        #    self.world.world.tick()
        #else:
        #    self.world.world.wait_for_tick()
        
        #if self.controller.parse_events():
        #    print("Parse event exist")
        #    return None, None, True, False
            
        self.world.tick(self.clock)
        if self.render is True:
            self.world.render(self.display)
            pygame.display.flip()

        if self.start:
            self.start_time = self.hud.simulation_time
            self.start = False
        
        #if self.hud.has_collision is not None:
        #    print(self.hud.has_collision)
        #print("hud frame is: ", self.hud.frame)
        #print("collision info is: ", self.hud.has_collision)

        vehicle = self.world.player
        cur_pos = vehicle.get_location()

        cur_x, cur_y, cur_z = cur_pos.x, cur_pos.y, cur_pos.z
        target_x, target_y, target_z = self.waypoints[self.idx]
        distance = math.sqrt((cur_x - target_x)**2 + (cur_y-target_y)**2)
        target_dist = distance
        #print(distance, self.idx)
        
        # Both Scoring Function + Waypoint Update
        if distance < 5:
            self.idx += 1
            if self.idx == len(self.waypoints):
                cur_time = self.hud.simulation_time
                self.total_score += (cur_time - self.start_time)
                self.start_time =  cur_time
                self.hud.notification("Score: " + str(round(self.total_score, 1)), 3)
                print("Lap Done")
                print("Final Score is ", self.total_score)
                self.idx = 0
                return None, 1000, True, False
            
            # Consider the case of crossing the boundary. 
            crossing_diff = self.cumulative_dist[self.idx - 1] - self.cumulative_dist[self.idx]
            #print("Difference of crossing is: ", crossing_diff)
            target_dist =  target_dist + crossing_diff

            # Draw the waypoints as Gate for Fancy Visualization
            x, y, z = self.waypoints[self.idx]
            location = carla.Location(x, y, z)
            rotation = self.world.map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=carla.LaneType.Driving).transform.rotation
            box = carla.BoundingBox(location, carla.Vector3D(0, 6, 4))

            # Goal Gate is RED
            if self.idx == len(self.waypoints) - 1:
                self.world.world.debug.draw_box(
                    box,
                    rotation,
                    thickness=0.5,
                    color=carla.Color(255, 0, 0, 255),
                    life_time=0)
            
            # Since wps is very dense in Shanghai Track, we only choose 1 in every 4 waypoints
            elif self.idx % 4 == 0:
                cur_time = self.hud.simulation_time
                self.total_score += (cur_time - self.start_time)
                self.start_time =  cur_time

                self.hud.notification("Score: " + str(round(self.total_score, 1)), 3)
                self.world.world.debug.draw_box(
                    box,
                    rotation,
                    thickness=0.5,
                    color=carla.Color(0, 0, 100, 255),
                    life_time=2)

        ###### Get Obstacles ######
        all_actors = self.world.world.get_actors()
        sensoring_radius = 100
        filtered_obstacles = []
        for actor in all_actors:
            # get actor's location
            cur_loc = actor.get_location()
            # determine whether actor is within the radius
            if cur_pos.distance(cur_loc) <= sensoring_radius:
                # we need to exclude actors such as camera
                # types we need: vehicle, walkers, Traffic signs and traffic lights
                # reference: https://github.com/carla-simulator/carla/blob/master/PythonAPI/carla/scene_layout.py
                if 'vehicle' in actor.type_id and actor.id != vehicle.id:
                    filtered_obstacles.append(actor)
                elif 'pedestrian' in actor.type_id:
                    filtered_obstacles.append(actor)
                elif 'static.prop' in actor.type_id:
                    filtered_obstacles.append(actor)

        ############## Get Lane Info ###############
        cur_waypoint = self.world.map.get_waypoint(cur_pos)
        cur_left = cur_waypoint
        cur_right = cur_waypoint
        left_boundary, right_boundary = [], []

        # Get Left Boundary
        while cur_left.get_left_lane():
            cur_left = cur_left.get_left_lane()
        for i in range(50):
            left_boundary.extend(cur_left.next(i+0.5))

        # Get Right Boundary
        while cur_right.get_right_lane():
            cur_right = cur_right.get_right_lane()
        for i in range(50):
            right_boundary.extend(cur_right.next(i+0.5))

        boundary = []
        boundary.append(left_boundary)
        boundary.append(right_boundary)
        ############################################
        
        # Get the current block
        mb = []
        mlb, mrb = [], []
        mlb.extend(cur_left.previous(0.5))
        mlb.extend(cur_left.next(0.5))
        mrb.extend(cur_right.previous(0.5))
        mrb.extend(cur_right.next(0.5))
        mb.append(mlb)
        mb.append(mrb)
        

        ########## Get Vehicle State Info ##########
        vel = vehicle.get_velocity()
        transform = vehicle.get_transform()
        ############################################

        end_waypoints = min(len(self.waypoints), self.idx + 50)

        # Compute the remain distance

        remain_distance = target_dist + self.cumulative_dist[self.idx]
        #print("Current idx is: ", self.idx)
        if self.prev_remain_dist is None:
            progress = 0
        else:
        #    print("distance is: ", distance)
        #    print("cum distance is: ", self.cumulative_dist[self.idx])
        #    print("Current remain distance: ", remain_distance)
        #    print("Previous remain distance: ", self.prev_remain_dist)
            progress = self.prev_remain_dist - remain_distance # Can be negative

        self.prev_remain_dist = remain_distance # Update the remain distance.

        # Compute the current block center line and the distance to the center line.
        mml, mmr = extract_road_boundary(mb)
        mmc = 0.5 * (mml + mmr)
        dist_to_center_line = compute_distance(mmc[0], mmc[1], [transform.location.x, transform.location.y])

        # Construct the reward
        #print("Progress is: ", progress)
        r1 = self.distance_weight * (progress) # The progress has been made
        #print("r1 is: ", r1)
        #print("Distance to center line is: ", dist_to_center_line)
        r2 = self.center_line_weight * (-1) * dist_to_center_line # distance to the center line
        #print("r2 is: ", r2)
        r3 = 0
        collision_happen_flag = False
        if self.hud.has_collision is not None:
            collision_happen_flag = True
            intensity = self.hud.has_collision[1]
            #print("Collision happens, intensity is: ", intensity)
            r3 = self.collision_weight * (-1) * intensity
            #print("r3 is: ", r3)
        reward = r1 + r2 + r3
        #print("total reward is: ", reward)
        #print()
        # Return the desired objects
        return (filtered_obstacles, self.waypoints[self.idx:end_waypoints], vel, transform,\
                 boundary, distance, self.prev_acc, self.prev_steer), reward, False, collision_happen_flag 


    # Only call this function after a reset
    def step(self, control):
        self.world.player.apply_control(control)
        
        if control.brake == 0:
            self.prev_acc = control.throttle
        else:
            self.prev_acc = (-1) * control.brake
        self.prev_steer = control.steer
        
        state, reward, terminated, collision_happen_flag = self.get_current_state()
        
        if state is None and reward == 100 and terminated is True:
            return state, reward, terminated, False
        
        if collision_happen_flag is True:
            return state, reward, terminated, True
        
        truncation = False
        rounded_dist = round(state[5], self.round_precision)
        if rounded_dist == self.prev_dist:
            self.stuck_counter += 1
            #print("Current stuck counter is: ", self.stuck_counter)
        else:
            self.stuck_counter = 0
        self.prev_dist = rounded_dist
        if self.stuck_counter == self.stuck_counter_limit: # Allows stucking at the same position for 50 times
            #print("Already stuck for", self.stuck_counter_limit, " times")
            truncation = True
            reward -= 100
        return state, reward, terminated, truncation
        
#===============================================================================
# ------------ testing with some agent  ----------------------------------------
#===============================================================================
def test_plain(args, render=False, rounds=1):
    from agent import Agent1
    agent = Agent1()
    try:
        env = RACE_ENV(args, collision_weight=30, distance_weight=20, center_line_weight=1, render=render, round_precision=3, stuck_counter_limit=30)
        for i in range(0, rounds):
            print("Start round: ", i)
            state, info = env.reset()
            if info is not None:
                print("Testing: error environment reset")
                return
            test_end = False
            while test_end is False:
                control = agent.run_step(state[0], state[1], state[2], state[3], state[4], state[5])
                state, reward, terminated, truncation = env.step(control)
                print("Reward is: ", reward)
                if terminated or truncation:
                    #print("Meet termination or truncation")
                    test_end = True
            print("Finish round: ", i)
    finally:
        env.close()
    return


def test_a2c_agent(args, render=False, rounds=1):
    from agent import Agent
    # Initialize the environment
    agent = Agent(episode_num=1, gamma=0.9, a_lr=1e-5, c_lr=3e-5, batch_size=1024, batch_round=1,\
                    update_round=3, step_limit=100000, action_dim=2, \
                    action_bound=torch.tensor([math.pi / 6, 1]).to(device), rb_max=2048, input_dim=208,\
                    collision_weight=30, distance_weight=20, center_line_weight=1,\
                    render=False, round_precision=3, stuck_counter_limit=20)
    loaded_state_dict = torch.load("./actor.pth")
    agent.act_net.load_state_dict(loaded_state_dict)
    try:
        env = RACE_ENV(args, collision_weight=30, distance_weight=5, center_line_weight=5, render=render, round_precision=4, stuck_counter_limit=100)
        for _ in range(0, rounds):
            state, info = env.reset()
            if info is not None:
                print("Testing: error environment reset")
                return
            test_end = False
            while test_end is False:
                control = agent.run_step(state)
                state, reward, terminated, truncation = env.step(control)
                #print("Reward is: ", reward)
                if terminated or truncation:
                    #print("Meet termination or truncation")
                    test_end = True
    finally:
        env.close()
    return

def a2c_train():
    from agent import Agent

    agent = Agent(episode_num=15000, gamma=0.9, a_lr=1e-5, c_lr=5e-5, batch_size=8192, batch_round=1,\
                    update_round=5, step_limit=10000000, action_dim=2, \
                    action_bound=torch.tensor([math.pi / 6, 1]).to(device), rb_max=50000, input_dim=208,\
                    collision_weight=3, distance_weight=5, center_line_weight=0.5,\
                    render=False, round_precision=2, stuck_counter_limit=20)
    loaded_actor_dict = torch.load("./actor.pth")
    agent.act_net.load_state_dict(loaded_actor_dict)
    loaded_critic_dict = torch.load("./critic.pth")
    agent.critic_net.load_state_dict(loaded_critic_dict)
    agent.train()
    torch.save(agent.act_net.state_dict(), "./actor1.pth")
    torch.save(agent.critic_net.state_dict(), "./critic1.pth")
    #print(agent.training_reward_x, agent.training_reward_y)
    #plot(agent.training_reward_x, agent.training_reward_y, "Cumulative reward", fn="./cumulative_reward.png", shown=True)
    x = torch.tensor(agent.training_reward_x)
    y = torch.tensor(agent.training_reward_y)
    torch.save(x, 'tx1.pt')
    torch.save(y, 'ty1.pt')

    return

def test_a2c_star_agent(args, render=True, rounds=1):
    from staragent import StarAgent 
    agent = StarAgent(10, 0.9, a_lr=1e-4, c_lr=5e-4, batch_size =16, batch_round=3,\
                      update_round=5, step_limit=10000000, action_dim=2, \
                      action_bound=torch.tensor([math.pi / 6, 1]).to(device), rb_max=50000, input_dim=208,\
                        collision_weight=3, distance_weight=5, center_line_weight=0.1,\
                        render=True, round_precision=3, stuck_counter_limit=20, maxT=5, patch_length=16)
    loaded_actor_dict = torch.load("./actor_str.pth")
    agent.act_net.load_state_dict(loaded_actor_dict)
    try:
        env = RACE_ENV(args, collision_weight=30, distance_weight=5, center_line_weight=5, render=True, round_precision=4, stuck_counter_limit=100)
        for _ in range(0, rounds):
            state, info = env.reset()
            state = convert_state_to_tensor(state)
            if info is not None:
                print("Testing: error environment reset")
                return
            test_end = False
            visiting_states = None
            visiting_actions = None

            while test_end is False:
                assert (visiting_states is None and visiting_actions is None) or \
                    ((visiting_states.size(0) == visiting_actions.size(0))\
                    and visiting_states.size(0) <=agent.maxT), "Error stacking visiting states"
                
                if visiting_states is None:
                    visiting_states = deepcopy(state)
                else:
                    if visiting_states.size(0) == agent.maxT:
                        visiting_states = visiting_states[1:, ...]
                        visiting_actions = visiting_actions[1:, ...]
                    visiting_states = torch.vstack((visiting_states, state))
            
                # Step the environment based on the selected action
                # Note: this action is on GPU and is a tensor
                # Add a dummpy action padding
                if visiting_actions is None:
                    visiting_actions = torch.zeros(1, agent.action_dim).to(device)
                else:
                    visiting_actions = torch.vstack((visiting_actions, torch.zeros(1, visiting_actions.size(1)).to(device)))
            
                # Convert the form of input
                visiting_states = visiting_states.unsqueeze(0)
                visiting_actions = visiting_actions.unsqueeze(0)
                action = agent.sample_action_from_state_gaussian(visiting_states, visiting_actions)
                
                # Convert the forms back
                visiting_states = visiting_states.squeeze(0)#.view(visiting_states.size(1), -1)
                visiting_actions = visiting_actions.squeeze(0)#.view(visiting_actions.size(1), -1)
                visiting_actions = visiting_actions[:-1, ...] # Throuw away the dummy action
                visiting_actions = torch.vstack((visiting_actions, action))
                assert visiting_states.size(0) == visiting_actions.size(0), "Visting states and actions are not eqaul"
                
                # Convert the action to control object before stepping.
                control = get_control_from_action(action)
                
                state, reward, terminated, truncation = env.step(control)
                state = convert_state_to_tensor(state)
                #print("Reward is: ", reward)
                if terminated or truncation:
                    #print("Meet termination or truncation")
                    test_end = True
    finally:
        env.close()
    return

def a2c_star_train():
    from staragent import StarAgent 
    agent = StarAgent(2000, 0.95, a_lr=1e-4, c_lr=5e-4, batch_size =16, batch_round=1,\
                      update_round=5, step_limit=10000, action_dim=2, \
                      action_bound=torch.tensor([math.pi / 6, 1]).to(device), rb_max=50000, input_dim=208,\
                        collision_weight=3, distance_weight=8, center_line_weight=0.5,\
                        render=False, round_precision=1, stuck_counter_limit=30, maxT=5, patch_length=16)
    loaded_actor_dict = torch.load("./actor_str.pth")
    agent.act_net.load_state_dict(loaded_actor_dict)
    loaded_critic_dict = torch.load("./critic_str.pth")
    agent.critic_net.load_state_dict(loaded_critic_dict)
    agent.train()
    torch.save(agent.act_net.state_dict(), "./actor_str.pth")
    torch.save(agent.critic_net.state_dict(), "./critic_str.pth")
    #print(agent.training_reward_x, agent.training_reward_y)
    #plot(agent.training_reward_x, agent.training_reward_y, "Cumulative reward", fn="./cumulative_reward.png", shown=True)
    x = torch.tensor(agent.training_reward_x)
    y = torch.tensor(agent.training_reward_y)
    torch.save(x, 'tx_str1.pt')
    torch.save(y, 'ty_str1.pt')

    return

def a2c_star_train_only_steer():
    from staragent import StarAgent 
    agent = StarAgent(1000, 0.95, a_lr=1e-4, c_lr=5e-4, batch_size =16, batch_round=1,\
                      update_round=10, step_limit=10000, action_dim=1, \
                      action_bound=torch.tensor([0.5]).to(device), rb_max=50000, input_dim=48,\
                        collision_weight=0, distance_weight=8, center_line_weight=0,\
                        render=False, round_precision=3, stuck_counter_limit=30, maxT=5, patch_length=8)
    #loaded_actor_dict = torch.load("./actor_str2.pth")
    #agent.act_net.load_state_dict(loaded_actor_dict)
    #loaded_critic_dict = torch.load("./critic_str2.pth")
    #agent.critic_net.load_state_dict(loaded_critic_dict)
    agent.train()
    torch.save(agent.act_net.state_dict(), "./actor_str.pth")
    torch.save(agent.critic_net.state_dict(), "./critic_str.pth")
    #print(agent.training_reward_x, agent.training_reward_y)
    #plot(agent.training_reward_x, agent.training_reward_y, "Cumulative reward", fn="./cumulative_reward.png", shown=True)
    x = torch.tensor(agent.training_reward_x)
    y = torch.tensor(agent.training_reward_y)
    torch.save(x, 'tx_str.pt')
    torch.save(y, 'ty_str.pt')

    return

def test_a2c_star_agent_only_steer(args, render=True, rounds=1):
    from staragent import StarAgent 
    agent = StarAgent(2000, 0.95, a_lr=1e-4, c_lr=5e-4, batch_size =16, batch_round=1,\
                      update_round=5, step_limit=10000, action_dim=1, \
                      action_bound=torch.tensor([0.1]).to(device), rb_max=50000, input_dim=208,\
                        collision_weight=3, distance_weight=8, center_line_weight=0.5,\
                        render=False, round_precision=3, stuck_counter_limit=30, maxT=5, patch_length=16)
    loaded_actor_dict = torch.load("./actor_str1.pth")
    agent.act_net.load_state_dict(loaded_actor_dict)
    try:
        env = RACE_ENV(args, collision_weight=30, distance_weight=5, center_line_weight=5, render=True, round_precision=4, stuck_counter_limit=100)
        for _ in range(0, rounds):
            state, info = env.reset()
            state = convert_state_to_tensor(state)
            if info is not None:
                print("Testing: error environment reset")
                return
            test_end = False
            visiting_states = None
            visiting_actions = None

            while test_end is False:
                assert (visiting_states is None and visiting_actions is None) or \
                    ((visiting_states.size(0) == visiting_actions.size(0))\
                    and visiting_states.size(0) <=agent.maxT), "Error stacking visiting states"
                
                if visiting_states is None:
                    visiting_states = deepcopy(state)
                else:
                    if visiting_states.size(0) == agent.maxT:
                        visiting_states = visiting_states[1:, ...]
                        visiting_actions = visiting_actions[1:, ...]
                    visiting_states = torch.vstack((visiting_states, state))
            
                # Step the environment based on the selected action
                # Note: this action is on GPU and is a tensor
                # Add a dummpy action padding
                if visiting_actions is None:
                    visiting_actions = torch.zeros(1, agent.action_dim).to(device)
                else:
                    visiting_actions = torch.vstack((visiting_actions, torch.zeros(1, visiting_actions.size(1)).to(device)))
            
                # Convert the form of input
                visiting_states = visiting_states.unsqueeze(0)
                visiting_actions = visiting_actions.unsqueeze(0)
                visiting_states = visiting_states.to(torch.float32)
                action, _ = agent.forward_state(visiting_states, visiting_actions)
                action = action.detach()
                #action = agent.sample_action_from_state_gaussian(visiting_states, visiting_actions)
                
                # Convert the forms back
                visiting_states = visiting_states.squeeze(0)#.view(visiting_states.size(1), -1)
                visiting_actions = visiting_actions.squeeze(0)#.view(visiting_actions.size(1), -1)
                visiting_actions = visiting_actions[:-1, ...] # Throuw away the dummy action
                visiting_actions = torch.vstack((visiting_actions, action))
                assert visiting_states.size(0) == visiting_actions.size(0), "Visting states and actions are not eqaul"
                
                # Convert the action to control object before stepping.
                control = get_control_from_action(action)
                
                state, reward, terminated, truncation = env.step(control)
                state = convert_state_to_tensor(state)
                #print("Reward is: ", reward)
                if terminated or truncation:
                    #print("Meet termination or truncation")
                    test_end = True
    finally:
        env.close()
    return
#===============================================================================
# ------------ utils  ----------------------------------------
#===============================================================================




# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    try:
        # Triaining =================================================
        #a2c_train()
        #a2c_star_train()
        a2c_star_train_only_steer()

        # Testing ===================================================
        #test_plain(args, render=True, rounds=30)
        #test_a2c_agent(args, True, 10)
        #test_a2c_star_agent(args, True, 10)
        #test_a2c_star_agent_only_steer(args, True, 10)
        print('end of game loop')
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
