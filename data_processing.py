#import libs
from collections import defaultdict
import json

#import custom functions
from utils import *
import logging

logging.basicConfig(level=logging.INFO)

class Ego4d:
    def __init__(self, path):
        """Class for reading and visualizing annotations.
        Args:
            path (str): location of annotation file
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading annotations.")
        
        self.dataset = self._load_json(path)
        self.version = self.dataset['version']
        self.description = self.dataset['description']
        
        assert (
            type(self.dataset) == dict
        ), "Annotation file format {} not supported.".format(type(self.dataset))
        
        self._create_index()
        
    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)
        
    def _create_index(self):
        self.logger.info("Creating index.")
        self.videos = defaultdict()
        
        for v in self.dataset['videos']:
            self.videos[v['video_uid']] = v
            
        pass


class Ego4d_NQL:
    def __init__(self, path):
        """Class for reading and visualizing annotations.
        Args:
            path (str): location of annotation file
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading annotations.")
        
        self.dataset = self._load_json(path)
        self.version = self.dataset['version']
        self.description = self.dataset['description']
        
        assert (
            type(self.dataset) == dict
        ), "Annotation file format {} not supported.".format(type(self.dataset))
        
        self._create_index()
        
    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)
        
    def _create_index(self):
        self.logger.info("Creating index.")
        self.videos = defaultdict()
        self.clips = defaultdict()
        self.video_clip = defaultdict(list)
        self.clip_video = defaultdict()
        
        for v in self.dataset['videos']:
            self.videos[v['video_uid']] = v
            _temp_clip_id = [] 
            for c in v['clips']:
                self.clips[c['clip_uid']] = c
                self.clip_video[c['clip_uid']] = v['video_uid']
                _temp_clip_id.append(c['clip_uid'])
            self.video_clip[v['video_uid']] = _temp_clip_id
        pass

class Ego4d_VQ:
    def __init__(self, path):
        """Class for reading and visualizing annotations.
        Args:
            path (str): location of annotation file
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading annotations.")
        
        self.dataset = self._load_json(path)
        self.version = self.dataset['version']
        self.description = self.dataset['description']
        
        assert (
            type(self.dataset) == dict
        ), "Annotation file format {} not supported.".format(type(self.dataset))
        
        self._create_index()
        
    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)
        
    def _create_index(self):
        self.logger.info("Creating index.")
        self.videos = defaultdict()
        self.clips = defaultdict()
        self.video_clip = defaultdict(list)
        self.clip_video = defaultdict()
        
        for v in self.dataset['videos']:
            self.videos[v['video_uid']] = v
            _temp_clip_id = [] 
            for c in v['clips']:
                self.clips[c['clip_uid']] = c
                self.clip_video[c['clip_uid']] = v['video_uid']
                _temp_clip_id.append(c['clip_uid'])
            self.video_clip[v['video_uid']] = _temp_clip_id
        pass

class Ego4d_MQ:
    def __init__(self, path):
        """Class for reading and visualizing annotations.
        Args:
            path (str): location of annotation file
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading annotations.")
        
        self.dataset = self._load_json(path)
        self.version = self.dataset['version']
        self.description = self.dataset['description']
        
        assert (
            type(self.dataset) == dict
        ), "Annotation file format {} not supported.".format(type(self.dataset))
        
        self._create_index()
        
    def _load_json(self, path):
        with open(path, "r") as f:
            return json.load(f)
        
    def _create_index(self):
        self.logger.info("Creating index.")
        self.videos = defaultdict()
        self.clips = defaultdict()
        self.video_clip = defaultdict(list)
        self.clip_video = defaultdict()
        
        for v in self.dataset['videos']:
            self.videos[v['video_uid']] = v
            _temp_clip_id = [] 
            for c in v['clips']:
                self.clips[c['clip_uid']] = c
                self.clip_video[c['clip_uid']] = v['video_uid']
                _temp_clip_id.append(c['clip_uid'])
            self.video_clip[v['video_uid']] = _temp_clip_id
        pass