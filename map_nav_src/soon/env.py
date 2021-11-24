''' Batched REVERIE navigation environment '''

import json
import os
import numpy as np
import math
import random
import networkx as nx
from collections import defaultdict
import copy
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

import MatterSim

from utils.data import load_nav_graphs, new_simulator
from utils.data import angle_feature, get_all_point_angle_feature
from reverie.env import EnvBatch

from soon.data_utils import normalize_angle

class SoonObjectNavBatch(object):
    def __init__(
        self, view_db, obj_db, instr_data, connectivity_dir,
        batch_size=64, angle_feat_size=4, max_objects=100, 
        seed=0, name=None, sel_data_idxs=None, is_train=False,
        multi_endpoints=False, multi_startpoints=False,
    ):
        self.env = EnvBatch(connectivity_dir, feat_db=view_db, batch_size=batch_size)
        self.obj_db = obj_db
        self.data = instr_data
        self.scans = set([x['scan'] for x in self.data])
        self.multi_endpoints = multi_endpoints
        self.multi_startpoints = multi_startpoints
        self.connectivity_dir = connectivity_dir
        self.batch_size = batch_size
        self.angle_feat_size = angle_feat_size
        self.max_objects = max_objects
        self.name = name
        self.is_train = is_train

        self.gt_trajs = self._get_gt_trajs(self.data) # for evaluation

        # in validation, we would split the data
        if sel_data_idxs is not None:
            t_split, n_splits = sel_data_idxs
            ndata_per_split = len(self.data) // n_splits 
            start_idx = ndata_per_split * t_split
            if t_split == n_splits - 1:
                end_idx = None
            else:
                end_idx = start_idx + ndata_per_split
            self.data = self.data[start_idx: end_idx]

        # use different seeds in different processes to shuffle data
        self.seed = seed
        random.seed(self.seed)
        if self.is_train:
            random.shuffle(self.data)

        self.ix = 0
        self._load_nav_graphs()

        self.sim = new_simulator(self.connectivity_dir)
        self.angle_feature = get_all_point_angle_feature(self.sim, self.angle_feat_size)
        
        self.buffered_state_dict = {}
        print('%s loaded with %d instructions, using splits: %s' % (
            self.__class__.__name__, len(self.data), self.name))

    def _get_gt_trajs(self, data):
        # for evaluation
        gt_trajs = {
            x['path_id']: copy.deepcopy(x) for x in data if 'bboxes' in x 
        }
        # normalize
        for path_id, value in gt_trajs.items():
            new_bboxes = {}
            for vp, bbox in value['bboxes'].items():
                new_bbox = copy.deepcopy(bbox)
                new_bbox['heading'] = new_bbox['target']['center']['heading'] / (2 * math.pi)
                new_bbox['elevation'] = (new_bbox['target']['center']['elevation'] + math.pi) / (2 * math.pi)
                new_bbox['target']['left_top']['heading'] = new_bbox['target']['left_top']['heading'] / (2 * math.pi)
                new_bbox['target']['left_top']['elevation'] = (new_bbox['target']['left_top']['elevation'] + math.pi) / (2 * math.pi)
                new_bbox['target']['right_bottom']['heading'] = new_bbox['target']['right_bottom']['heading'] / (2 * math.pi)
                new_bbox['target']['right_bottom']['elevation'] = (new_bbox['target']['right_bottom']['elevation'] + math.pi) / (2 * math.pi)
                new_bbox['target']['left_bottom']['heading'] = new_bbox['target']['left_bottom']['heading'] / (2 * math.pi)
                new_bbox['target']['left_bottom']['elevation'] = (new_bbox['target']['left_bottom']['elevation'] + math.pi) / (2 * math.pi)
                new_bbox['target']['right_top']['heading'] = new_bbox['target']['right_top']['heading'] / (2 * math.pi)
                new_bbox['target']['right_top']['elevation'] = (new_bbox['target']['right_top']['elevation'] + math.pi) / (2 * math.pi)
                new_bboxes[vp] = new_bbox
            gt_trajs[path_id]['bboxes'] = new_bboxes
        return gt_trajs

    def size(self):
        return len(self.data)

    def _load_nav_graphs(self):
        """
        load graph from self.scan,
        Store the graph {scan_id: graph} in self.graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in self.paths
        Store the distances in self.distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        :return: None
        """
        print('Loading navigation graphs for %d scans' % len(self.scans))
        self.graphs = load_nav_graphs(self.connectivity_dir, self.scans)
        self.shortest_paths = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        self.shortest_distances = {}
        for scan, G in self.graphs.items():  # compute all shortest paths
            self.shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

    def _next_minibatch(self, batch_size=None, **kwargs):
        if batch_size is None:
            batch_size = self.batch_size
        
        batch = self.data[self.ix: self.ix+batch_size]
        if len(batch) < batch_size:
            random.shuffle(self.data)
            self.ix = batch_size - len(batch)
            batch += self.data[:self.ix]
        else:
            self.ix += batch_size
        self.batch = batch

        for item in self.batch:
            if self.is_train:
                item['heading'] = np.random.rand() * np.pi * 2
            else:
                item['heading'] = 1.52
            item['elevation'] = 0
        
        if self.is_train:
            batch = copy.deepcopy(self.batch)
            start_vps = [x['path'][0] for x in self.batch]
            end_vps = [x['path'][-1] for x in self.batch]
            if self.multi_startpoints:
                for i, item in enumerate(batch):
                    cand_vps = []
                    for cvp, cpath in self.shortest_paths[item['scan']][end_vps[i]].items():
                        if len(cpath) >= 6 and len(cpath) <= 15:
                            cand_vps.append(cvp)
                    if len(cand_vps) > 0:
                        start_vps[i] = cand_vps[np.random.randint(len(cand_vps))]
            if self.multi_endpoints:
                for i, item in enumerate(batch):
                    end_vp = item['end_image_ids'][np.random.randint(len(item['end_image_ids']))]
                    end_vps[i] = end_vp
            for i, item in enumerate(batch):
                item['path'] = self.shortest_paths[item['scan']][start_vps[i]][end_vps[i]]
            self.batch = batch
    
    def reset_epoch(self, shuffle=False):
        ''' Reset the data index to beginning of epoch. Primarily for testing.
            You must still call reset() for a new episode. '''
        if shuffle:
            random.shuffle(self.data)
        self.ix = 0

    def make_candidate(self, feature, scanId, viewpointId, viewId):
        def _loc_distance(loc):
            return np.sqrt(loc.rel_heading ** 2 + loc.rel_elevation ** 2)
        base_heading = (viewId % 12) * math.radians(30)
        base_elevation = (viewId // 12 - 1) * math.radians(30)

        adj_dict = {}
        long_id = "%s_%s" % (scanId, viewpointId)
        if long_id not in self.buffered_state_dict:
            for ix in range(36):
                if ix == 0:
                    self.sim.newEpisode([scanId], [viewpointId], [0], [math.radians(-30)])
                elif ix % 12 == 0:
                    self.sim.makeAction([0], [1.0], [1.0])
                else:
                    self.sim.makeAction([0], [1.0], [0])

                state = self.sim.getState()[0]
                assert state.viewIndex == ix

                # Heading and elevation for the viewpoint center
                heading = state.heading - base_heading
                elevation = state.elevation - base_elevation

                visual_feat = feature[ix]

                # get adjacent locations
                for j, loc in enumerate(state.navigableLocations[1:]):
                    # if a loc is visible from multiple view, use the closest
                    # view (in angular distance) as its representation
                    distance = _loc_distance(loc)

                    # Heading and elevation for for the loc
                    loc_heading = heading + loc.rel_heading
                    loc_elevation = elevation + loc.rel_elevation
                    angle_feat = angle_feature(loc_heading, loc_elevation, self.angle_feat_size)
                    if (loc.viewpointId not in adj_dict or
                            distance < adj_dict[loc.viewpointId]['distance']):
                        adj_dict[loc.viewpointId] = {
                            'heading': loc_heading,
                            'elevation': loc_elevation,
                            "normalized_heading": state.heading + loc.rel_heading,
                            "normalized_elevation": state.elevation + loc.rel_elevation,
                            'scanId': scanId,
                            'viewpointId': loc.viewpointId, # Next viewpoint id
                            'pointId': ix,
                            'distance': distance,
                            'idx': j + 1,
                            'feature': np.concatenate((visual_feat, angle_feat), -1),
                            'position': (loc.x, loc.y, loc.z),
                        }
            candidate = list(adj_dict.values())
            self.buffered_state_dict[long_id] = [
                {key: c[key]
                 for key in
                    ['normalized_heading', 'normalized_elevation', 'scanId', 'viewpointId',
                     'pointId', 'idx', 'position']}
                for c in candidate
            ]
            return candidate
        else:
            candidate = self.buffered_state_dict[long_id]
            candidate_new = []
            for c in candidate:
                c_new = c.copy()
                ix = c_new['pointId']
                visual_feat = feature[ix]
                c_new['heading'] = c_new['normalized_heading'] - base_heading
                c_new['elevation'] = c_new['normalized_elevation'] - base_elevation
                angle_feat = angle_feature(c_new['heading'], c_new['elevation'], self.angle_feat_size)
                c_new['feature'] = np.concatenate((visual_feat, angle_feat), -1)
                c_new.pop('normalized_heading')
                c_new.pop('normalized_elevation')
                candidate_new.append(c_new)
            return candidate_new

    def _get_obs(self, t=None, shortest_teacher=False):
        obs = []
        for i, (feature, state) in enumerate(self.env.getStates()):
            item = self.batch[i]
            base_view_id = state.viewIndex
           
            # Full features
            candidate = self.make_candidate(feature, state.scanId, state.location.viewpointId, state.viewIndex)
            # [visual_feature, angle_feature] for views
            feature = np.concatenate((feature, self.angle_feature[base_view_id]), -1)

            # objects
            obj_img_fts, obj_ang_fts, obj_box_fts, obj_directions, obj_ids = self.obj_db.get_object_feature(
                state.scanId, state.location.viewpointId, 
                state.heading, state.elevation, self.angle_feat_size,
                max_objects=self.max_objects
            )
            
            gt_obj_id = None
            vp = state.location.viewpointId
            if vp in item.get('end_image_ids', []):
                pseudo_label = item['image_id_to_obj_label'][vp]
                if pseudo_label is not None:
                    if self.max_objects is None or pseudo_label['idx'] < self.max_objects:
                        assert pseudo_label['obj_id'] == obj_ids[pseudo_label['idx']]
                        gt_obj_id = pseudo_label['obj_id']

            ob = {
                'instr_id' : item['instr_id'],
                'scan' : state.scanId,
                'viewpoint' : state.location.viewpointId,
                'viewIndex' : state.viewIndex,
                'position': (state.location.x, state.location.y, state.location.z),
                'heading' : state.heading,
                'elevation' : state.elevation,
                'feature' : feature,
                'candidate': candidate,
                'obj_img_fts': obj_img_fts,
                'obj_ang_fts': obj_ang_fts,
                'obj_box_fts': obj_box_fts,
                'obj_directions': obj_directions,
                'obj_ids': obj_ids,
                'navigableLocations' : state.navigableLocations,
                'instruction' : item['instruction'],
                'instr_encoding': item['instr_encoding'],
                'gt_path' : item['path'],
                'gt_end_vps': item.get('end_image_ids', []),
                'gt_obj_id': gt_obj_id,
                'path_id' : item['path_id']
            }

            if ob['path_id'] in self.gt_trajs:
                # A3C reward. There are multiple gt end viewpoints on SOON. 
                min_dist = np.inf
                for vp in self.batch[i]['end_image_ids']:
                    min_dist = min(min_dist, self.shortest_distances[ob['scan']][ob['viewpoint']][vp])
                ob['distance'] = min_dist
            else:
                ob['distance'] = 0

            obs.append(ob)
        return obs

    def reset(self, **kwargs):
        ''' Load a new minibatch / episodes. '''
        self._next_minibatch(**kwargs)
        
        scanIds = [item['scan'] for item in self.batch]
        viewpointIds = [item['path'][0] for item in self.batch]
        headings = [item['heading'] for item in self.batch]
        self.env.newEpisodes(scanIds, viewpointIds, headings)
        return self._get_obs()

    def step(self, actions):
        ''' Take action (same interface as makeActions) '''
        self.env.makeActions(actions)
        return self._get_obs()

    ############### Evaluation ###############
    def _get_nearest(self, shortest_distances, goal_id, path):
        near_id = path[0]
        near_d = shortest_distances[near_id][goal_id]
        for item in path:
            d = shortest_distances[item][goal_id]
            if d < near_d:
                near_id = item
                near_d = d
        return near_id

    def _eval_item(self, pred_path, obj_heading, obj_elevation, gt_item):
        scores = {}

        scan = gt_item['scan']
        shortest_distances = self.shortest_distances[scan]

        gt_path = gt_item['path']
        gt_bboxes = gt_item['bboxes']
        start_vp = gt_path[0]
        goal_vp = gt_path[-1]   

        path = sum(pred_path, [])
        assert gt_path[0] == path[0], 'Result trajectories should include the start position'

        # follow the original evaluation
        nearest_position = self._get_nearest(shortest_distances, goal_vp, path)
        if path[-1] in gt_bboxes:
            goal_vp = path[-1]  # update goal

        if path[-1] in gt_bboxes:
            gt_bbox = gt_bboxes[path[-1]]
            
            scores['heading_error'] = math.fabs(gt_bbox['heading'] - obj_heading)
            scores['elevation_error'] = math.fabs(gt_bbox['elevation'] - obj_elevation)
            scores['point_det_error'] = math.hypot(
                gt_bbox['heading'] - obj_heading, gt_bbox['elevation'] - obj_elevation)
            
            # TODO: there might be a bug due to radians angle as it is a circle
            obj_point = Point(obj_heading, obj_elevation)
            gt_poly = Polygon([(gt_bbox['target']['left_top']['heading'], gt_bbox['target']['left_top']['elevation']),
                               (gt_bbox['target']['right_top']['heading'], gt_bbox['target']['right_top']['elevation']),
                               (gt_bbox['target']['right_bottom']['heading'], gt_bbox['target']['right_bottom']['elevation']),
                               (gt_bbox['target']['left_bottom']['heading'], gt_bbox['target']['left_bottom']['elevation'])])

            if gt_poly.contains(obj_point):
                scores['det_success'] = True
            else:
                scores['det_success'] = False
            
        else:
            scores['det_success'] = False

        scores['action_steps'] = len(pred_path) - 1
        scores['trajectory_steps'] = len(path) - 1
        scores['trajectory_lengths'] = np.sum([shortest_distances[a][b] for a, b in zip(path[:-1], path[1:])])

        # navigation: success is navigation error < 3m 
        scores['nav_error'] = shortest_distances[path[-1]][goal_vp]
        # nearest_position = self._get_nearest(shortest_distances, goal_vp, path)
        scores['oracle_error'] = shortest_distances[nearest_position][goal_vp]
        scores['success'] = scores['nav_error'] < 3.
        scores['oracle_success'] = scores['oracle_error'] < 3.

        scores['goal_progress'] = shortest_distances[start_vp][goal_vp] - \
                                  shortest_distances[path[-1]][goal_vp]

        # gt_lengths = np.sum([shortest_distances[a][b] for a, b in zip(gt_path[:-1], gt_path[1:])])
        gt_lengths = shortest_distances[gt_path[0]][goal_vp]

        scores['spl'] = scores['success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        scores['det_spl'] = scores['det_success'] * gt_lengths / max(scores['trajectory_lengths'], gt_lengths, 0.01)
        return scores

    def eval_metrics(self, preds):
        ''' Evaluate each agent trajectory based on how close it got to the goal location 
        the path contains [view_id, angle, vofv]'''
        print('eval %d predictions' % (len(preds)))

        metrics = defaultdict(list)
        for item in preds:
            instr_id = item['instr_id']
            path_id = instr_id.split('_')[0]
            gt_item = self.gt_trajs[path_id]
            traj = item['trajectory']['path'] #[x[0] for x in item['trajectory']['path']]
            traj_scores = self._eval_item(traj, 
                item['trajectory']['obj_heading'][0], item['trajectory']['obj_elevation'][0], gt_item)
            for k, v in traj_scores.items():
                metrics[k].append(v)
            metrics['instr_id'].append(instr_id)
        
        avg_metrics = {
            'steps': np.mean(metrics['trajectory_steps']),
            'lengths': np.mean(metrics['trajectory_lengths']),
            'nav_error': np.mean(metrics['nav_error']),
            'oracle_error': np.mean(metrics['oracle_error']),
            'goal_progress': np.mean(metrics['goal_progress']),
            # 'heading_error': np.mean(metrics['heading_error']),
            # 'elevation_error': np.mean(metrics['elevation_error']),
            # 'point_det_error': np.mean(metrics['point_det_error']),
            'sr': np.mean(metrics['success']) * 100,
            'oracle_sr': np.mean(metrics['oracle_success']) * 100,
            'spl': np.mean(metrics['spl']) * 100,
            'det_sr': np.mean(metrics['det_success']) * 100,
            'det_spl': np.mean(metrics['det_spl']) * 100,
        }
        return avg_metrics, metrics
