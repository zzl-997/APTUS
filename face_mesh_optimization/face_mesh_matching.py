import copy
import logging
import math
from typing import Dict, List, Tuple, Optional, Any, Union

import cv2
import mediapipe as mp
import numpy as np
from scipy.optimize import linear_sum_assignment


class FaceMeshDetector:
    """MediaPipe face mesh detector for single face."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize face mesh detector.
        
        Args:
            config: Configuration dictionary containing:
                - detection_confidence: Minimum confidence for face detection
                - num_landmarks: Number of face landmarks to detect
        """
        self.detection_confidence = config['detection_confidence']
        self.num_landmarks = config['num_landmarks']
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.detector = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=self.detection_confidence
        )
        
        self.mp_connections = self.mp_face_mesh.FACEMESH_TESSELATION
        self.connections = set(self.mp_connections)

    @staticmethod
    def _calculate_point_distances(point: Tuple[float, float], 
                                  landmarks: List[List[float]], 
                                  num_landmarks: int) -> Tuple[List[int], List[float]]:
        """
        Calculate distances from a point to all face landmarks.
        
        Args:
            point: Query point coordinates (x, y)
            landmarks: List of landmark coordinates
            num_landmarks: Number of landmarks to consider
            
        Returns:
            Tuple of (sorted landmark indices by distance, distance list)
        """
        assert len(point) == 2, f"Point must have 2 coordinates, got {len(point)}"
        
        x, y = point[0], point[1]
        distances = []
        
        for i in range(num_landmarks):
            dx = x - landmarks[i][0]
            dy = y - landmarks[i][1]
            distances.append(math.sqrt(dx ** 2 + dy ** 2))
        
        sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])
        
        return sorted_indices, distances

    def _calculate_average_min_distance(self, landmarks: List[List[float]]) -> float:
        """
        Calculate average minimum distance between each landmark and its nearest neighbor.
        Used for evaluating mapping quality and determining color coding thresholds.
        
        Args:
            landmarks: List of landmark coordinates
            
        Returns:
            Average minimum distance between connected landmarks
        """
        neighbor_list = MultiViewOptimizer._get_all_connected_neighbors(
            self.num_landmarks, self.connections)
        
        min_distances = []
        
        for i, mark in enumerate(landmarks):
            if i >= self.num_landmarks:
                break
                
            x, y = mark[0], mark[1]
            min_dist = max(x, y)
            
            for neighbor_idx in neighbor_list[i]:
                dx = x - landmarks[neighbor_idx][0]
                dy = y - landmarks[neighbor_idx][1]
                min_dist = min(math.sqrt(dx ** 2 + dy ** 2), min_dist)
            
            min_distances.append(min_dist)
        
        return sum(min_distances) / len(min_distances)

    def detect(self, image: np.ndarray) -> Tuple[Any, float]:
        """
        Detect face mesh in image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (face mesh result, average minimum distance between landmarks)
            
        Raises:
            RuntimeError: If face detection fails
        """
        assert isinstance(image, np.ndarray), f"Image must be numpy array, got {type(image)}"
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb_image)
        
        if not results.multi_face_landmarks:
            raise RuntimeError(f"MediaPipe face detection failed with confidence={self.detection_confidence}")
        
        face_mesh_result = results.multi_face_landmarks[0]
        height, width, _ = image.shape
        
        landmarks = FaceMapper._extract_landmark_coordinates(face_mesh_result)
        avg_min_distance = self._calculate_average_min_distance(landmarks) * math.sqrt(width ** 2 + height ** 2)
        
        return face_mesh_result, avg_min_distance


class FaceMapper:
    """Maps points between different face images using face mesh landmarks."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize face mapper.
        
        Args:
            config: Configuration dictionary for face mesh detection
        """
        self.config = config
        self.num_landmarks = config['num_landmarks']
        self.source_detector = FaceMeshDetector(config)
        self.target_detector = FaceMeshDetector(config)

    def _find_base_point_index(self, point: Tuple[float, float], 
                               landmarks: List[List[float]], 
                               neighbor_list: List[List[int]]) -> Tuple[int, List[int]]:
        """
        Find the nearest landmark index and its neighbors for a given point.
        
        Args:
            point: Query point coordinates
            landmarks: List of landmark coordinates
            neighbor_list: List of neighbor indices for each landmark
            
        Returns:
            Tuple of (nearest landmark index, list of its neighbors)
        """
        sorted_indices, _ = FaceMeshDetector._calculate_point_distances(
            point, landmarks, self.num_landmarks)
        
        for idx in sorted_indices:
            if idx < self.num_landmarks:
                return idx, neighbor_list[idx]
        
        return -1, []

    def _compute_transformation_matrix(self, use_indices: List[int],
                                       source_landmarks: List[List[float]],
                                       target_landmarks: List[List[float]]) -> np.ndarray:
        """
        Compute transformation matrix between source and target landmarks using least squares.
        
        Args:
            use_indices: Indices of landmarks to use for computation
            source_landmarks: Source landmark coordinates
            target_landmarks: Target landmark coordinates
            
        Returns:
            3x3 transformation matrix
        """
        assert isinstance(use_indices, list), f"use_indices must be list, got {type(use_indices)}"
        
        if len(use_indices) == 0:
            source_matrix = np.array([
                [point[0], point[1], 1]
                for point in source_landmarks
            ])
            target_matrix = np.array([
                [point[0], point[1], point[2]]
                for point in target_landmarks
            ])
        else:
            source_matrix = np.array([
                [source_landmarks[idx][0], source_landmarks[idx][1], 1]
                for idx in use_indices
            ])
            target_matrix = np.array([
                [target_landmarks[idx][0], target_landmarks[idx][1], target_landmarks[idx][2]]
                for idx in use_indices
            ])
        
        # Solve using pseudo-inverse (least squares solution)
        inv_matrix = np.linalg.pinv(np.matmul(source_matrix.T, source_matrix))
        transformation = np.matmul(np.matmul(inv_matrix, source_matrix.T), target_matrix)
        
        return transformation

    @staticmethod
    def _extract_landmark_coordinates(face_mesh_result: Any) -> List[List[float]]:
        """
        Extract x, y, z coordinates from face mesh result.
        
        Args:
            face_mesh_result: MediaPipe face mesh result
            
        Returns:
            List of [x, y, z] coordinates for each landmark
        """
        coordinates = []
        for landmark in face_mesh_result.landmark:
            coordinates.append([landmark.x, landmark.y, landmark.z])
        return coordinates

    def process(self, points: List[List[float]], 
                source_image: np.ndarray, 
                target_image: np.ndarray,
                neighbor_list: List[List[int]]) -> Tuple[List[np.ndarray], float]:
        """
        Map points from source image to target image using face mesh transformation.
        
        Args:
            points: List of points to map (each point as [x, y])
            source_image: Source image
            target_image: Target image
            neighbor_list: Neighbor indices for each landmark
            
        Returns:
            Tuple of (mapped points, average minimum distance in target image)
        """
        assert isinstance(points, list), f"points must be list, got {type(points)}"
        assert isinstance(source_image, np.ndarray) and isinstance(target_image, np.ndarray)
        
        source_height, source_width = source_image.shape[:2]
        
        source_mesh_result, _ = self.source_detector.detect(source_image)
        target_mesh_result, target_avg_min_dist = self.target_detector.detect(target_image)
        
        # Normalize points to [0, 1] range
        normalized_points = [[p[0] / source_height, p[1] / source_width] for p in points]
        
        source_landmarks = self._extract_landmark_coordinates(source_mesh_result)
        target_landmarks = self._extract_landmark_coordinates(target_mesh_result)
        
        mapped_points = []
        for point in normalized_points:
            _, use_indices = self._find_base_point_index(point, source_landmarks, neighbor_list)
            transformation = self._compute_transformation_matrix(use_indices, source_landmarks, target_landmarks)
            
            point_with_bias = point + [1]  # Add bias term for affine transformation
            mapped_points.append(np.matmul(np.array(point_with_bias), transformation))
        
        return mapped_points, target_avg_min_dist


class PointMatcher:
    """Matches points between two sets using various algorithms."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize point matcher.
        
        Args:
            config: Configuration dictionary containing:
                - match_method: Matching algorithm ('greed' or 'hungarian')
                - class_threshold_strict: Strict confidence thresholds per class
                - class_threshold_loose: Loose confidence thresholds per class
                - float_max: Large value for infinity in cost matrices
        """
        self.match_method = config['match_method']
        self.strict_thresholds = config['class_threshold_strict']
        self.loose_thresholds = config['class_threshold_loose']
        self.float_max = config['float_max']

    def _hungarian_match(self, points1: List[List[float]], points2: List[List[float]],
                         scores1: List[float], scores2: List[float],
                         classes1: List[str], classes2: List[str],
                         threshold: float, filter_mode: str) -> List[List[Union[int, float]]]:
        """
        Match points using Hungarian algorithm.
        
        Args:
            points1: Points from first set
            points2: Points from second set
            scores1: Confidence scores for first set
            scores2: Confidence scores for second set
            classes1: Class labels for first set
            classes2: Class labels for second set
            threshold: Distance threshold for matching
            filter_mode: Filtering mode ('min_or', 'max_and', etc.)
            
        Returns:
            List of matches [index1, index2, distance]
        """
        n1, n2 = len(points1), len(points2)
        
        cost_matrix = np.full((n1, n2), -1, dtype=np.float64)
        valid_indices1 = []
        valid_indices2 = []
        
        # Build cost matrix with filtering
        for i in range(n1):
            p1 = points1[i]
            for j in range(n2):
                p2 = points2[j]
                dist_squared = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
                
                if dist_squared > threshold ** 2:
                    continue
                
                if not self._apply_filter(scores1[i], scores2[j], 
                                         classes1[i], classes2[j], filter_mode):
                    continue
                
                if i not in valid_indices1:
                    valid_indices1.append(i)
                if j not in valid_indices2:
                    valid_indices2.append(j)
                    
                cost_matrix[i][j] = dist_squared
        
        if not valid_indices1 or not valid_indices2:
            return []
        
        # Create reduced cost matrix
        n_valid1, n_valid2 = len(valid_indices1), len(valid_indices2)
        reduced_cost = np.full((n_valid1, n_valid2), self.float_max, dtype=np.float64)
        
        for i in range(n_valid1):
            for j in range(n_valid2):
                if cost_matrix[valid_indices1[i]][valid_indices2[j]] > 0:
                    reduced_cost[i][j] = cost_matrix[valid_indices1[i]][valid_indices2[j]]
        
        # Solve assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix=reduced_cost, maximize=False)
        
        # Filter results
        matches = []
        for i in range(len(row_indices)):
            if reduced_cost[row_indices[i]][col_indices[i]] < (self.float_max - 10):
                matches.append([
                    valid_indices1[row_indices[i]],
                    valid_indices2[col_indices[i]],
                    reduced_cost[row_indices[i]][col_indices[i]]
                ])
        
        return matches

    def _greedy_match(self, points1: List[List[float]], points2: List[List[float]],
                      scores1: List[float], scores2: List[float],
                      classes1: List[str], classes2: List[str],
                      threshold: float, filter_mode: str) -> List[List[Union[int, float]]]:
        """
        Match points using greedy algorithm (nearest neighbor).
        
        Args:
            Same as _hungarian_match
        """
        n1, n2 = len(points1), len(points2)
        
        # Build distance list
        distances = []
        for i in range(n1):
            p1 = points1[i]
            for j in range(n2):
                p2 = points2[j]
                dist_squared = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
                distances.append([dist_squared, i, j])
        
        # Sort by distance
        sorted_indices = sorted(range(len(distances)), key=lambda idx: distances[idx][0])
        
        visited1, visited2 = [], []
        matches = []
        
        for idx in sorted_indices:
            dist_squared, i, j = distances[idx]
            
            if dist_squared > threshold ** 2:
                break
            
            if not self._apply_filter(scores1[i], scores2[j], classes1[i], classes2[j], filter_mode):
                continue
            
            if i not in visited1 and j not in visited2:
                matches.append([i, j, dist_squared])
                visited1.append(i)
                visited2.append(j)
        
        return matches

    def _apply_filter(self, score1: float, score2: float, 
                      class1: str, class2: str, filter_mode: str) -> bool:
        """
        Apply confidence threshold filter to point pair.
        
        Args:
            score1: Confidence score for first point
            score2: Confidence score for second point
            class1: Class label for first point
            class2: Class label for second point
            filter_mode: Filtering mode
            
        Returns:
            True if point pair passes filter, False otherwise
        """
        if filter_mode == 'max_and':
            return not (score1 < self.strict_thresholds[class1] and 
                       score2 < self.strict_thresholds[class2])
        elif filter_mode == 'max_or':
            return not (score1 < self.strict_thresholds[class1] or 
                       score2 < self.strict_thresholds[class2])
        elif filter_mode == 'min_and':
            return not (score1 < self.loose_thresholds[class1] and 
                       score2 < self.loose_thresholds[class2])
        elif filter_mode == 'min_or':
            return not (score1 < self.loose_thresholds[class1] or 
                       score2 < self.loose_thresholds[class2])
        else:
            raise ValueError(f"Unknown filter mode: {filter_mode}")

    def _merge_matches(self, loose_matches: List[List], strict_matches: List[List]) -> List[List]:
        """
        Merge matches from strict and loose thresholds.
        
        Args:
            loose_matches: Matches from loose threshold
            strict_matches: Matches from strict threshold
            
        Returns:
            Merged matches list
        """
        if not strict_matches:
            return loose_matches
        
        result = strict_matches.copy()
        
        for loose_match in loose_matches:
            is_unique = True
            for strict_match in strict_matches:
                if strict_match[0] == loose_match[0] or strict_match[1] == loose_match[1]:
                    is_unique = False
                    break
            if is_unique:
                result.append(loose_match)
        
        return result

    def match(self, points1: List[List[float]], points2: List[List[float]],
              scores1: List[float], scores2: List[float],
              classes1: List[str], classes2: List[str],
              threshold: float) -> List[List[Union[int, float]]]:
        """
        Match points between two sets using two-stage filtering.
        
        Returns:
            List of matches [index1, index2, distance]
        """
        methods = {
            'greedy': self._greedy_match,
            'hungarian': self._hungarian_match
        }
        
        matcher = methods.get(self.match_method)
        if matcher is None:
            raise ValueError(f"Unknown match method: {self.match_method}")
        
        # First pass: loose threshold
        loose_matches = matcher(points1, points2, scores1, scores2, 
                               classes1, classes2, threshold, 'min_or')
        
        # Second pass: strict threshold
        strict_matches = matcher(points1, points2, scores1, scores2,
                                classes1, classes2, threshold, 'max_or')
        
        # Merge results
        return self._merge_matches(loose_matches, strict_matches)


class MultiViewOptimizer:
    """Optimizes detection results across multiple facial views."""
    
    def __init__(self):
        """Initialize multi-view optimizer."""
        self.initialized = False

    def initialize(self, config: Dict[str, Any]) -> None:
        """
        Initialize optimizer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.strict_thresholds = config['class_threshold_strict']
        self.loose_thresholds = config['class_threshold_loose']
        self.distance_scale = config['distance_scale']
        self.match_method = config['match_method']
        self.num_landmarks = config['num_landmarks']
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_connections = self.mp_face_mesh.FACEMESH_TESSELATION
        self.connections = set(self.mp_connections)
        
        self.face_mapper = FaceMapper(config)
        self.point_matcher = PointMatcher(config)
        
        self.initialized = True

    @staticmethod
    def _get_all_connected_neighbors(num_landmarks: int, 
                                     connections: set) -> List[List[int]]:
        """
        Get all directly connected neighbors for each landmark.
        
        Args:
            num_landmarks: Number of landmarks
            connections: Set of connection tuples
            
        Returns:
            List of neighbor indices for each landmark
        """
        neighbors = [[] for _ in range(num_landmarks)]
        
        for connection in connections:
            idx1, idx2 = connection
            if idx2 not in neighbors[idx1]:
                neighbors[idx1].append(idx2)
            if idx1 not in neighbors[idx2]:
                neighbors[idx2].append(idx1)
        
        return neighbors

    def _extract_detection_info(self, detection_result: Dict) -> Dict:
        """
        Extract and organize detection information from raw detection result.
        
        Args:
            detection_result: Raw detection result dictionary
            
        Returns:
            Organized detection info with boxes, centers, classes, scores, etc.
        """
        result = {
            'boxes': [],
            'centers': [],
            'classes': [],
            'scores': [],
            'show': [],
            'match_indices': []
        }
        
        for class_name in detection_result.get('det_boxes', {}):
            num_detections = len(detection_result['det_boxes'][class_name])
            
            for i in range(num_detections):
                if detection_result['scores'][class_name][i] < self.loose_thresholds[class_name]:
                    continue
                
                box = detection_result['det_boxes'][class_name][i]
                result['boxes'].append(box)
                result['classes'].append(class_name)
                result['scores'].append(detection_result['scores'][class_name][i])
                
                # Calculate center point from bounding box [xmin, ymin, xmax, ymax]
                center_x = round((box[0] + box[2]) / 2)
                center_y = round((box[1] + box[3]) / 2)
                result['centers'].append([center_x, center_y])
                
                should_show = detection_result['scores'][class_name][i] >= self.strict_thresholds[class_name]
                result['show'].append(should_show)
                result['match_indices'].append(-1)
        
        return result

    def _optimize_with_matches(self, source_info: Dict, target_info: Dict,
                               matches: List[List]) -> Tuple[Dict, Dict]:
        """
        Optimize detection results using matched points.
        
        Args:
            source_info: Source detection information
            target_info: Target detection information
            matches: List of matches [source_idx, target_idx, distance]
            
        Returns:
            Optimized source and target information
        """
        match_counter = 1
        
        for match in matches:
            src_idx, tgt_idx = match[0:2]
            
            # Skip low-confidence matches
            if (source_info['scores'][src_idx] < self.loose_thresholds[source_info['classes'][src_idx]] or
                target_info['scores'][tgt_idx] < self.loose_thresholds[target_info['classes'][tgt_idx]]):
                continue
            
            # Mark matched points for display
            source_info['show'][src_idx] = True
            target_info['show'][tgt_idx] = True
            
            # Assign match indices
            source_info['match_indices'][src_idx] = match_counter
            target_info['match_indices'][tgt_idx] = match_counter
            match_counter += 1
            
            # If classes differ, optimize based on confidence
            if source_info['classes'][src_idx] == target_info['classes'][tgt_idx]:
                continue
                
            if source_info['scores'][src_idx] >= target_info['scores'][tgt_idx]:
                if target_info['scores'][tgt_idx] < self.loose_thresholds[source_info['classes'][src_idx]]:
                    target_info['show'][tgt_idx] = False
                else:
                    target_info['classes'][tgt_idx] = source_info['classes'][src_idx]
            else:
                if source_info['scores'][src_idx] < self.loose_thresholds[target_info['classes'][tgt_idx]]:
                    source_info['show'][src_idx] = False
                else:
                    source_info['classes'][src_idx] = target_info['classes'][tgt_idx]
        
        return source_info, target_info

    def _process_single_face_pair(self, source_detection: Dict, target_detection: Dict,
                                  source_image: np.ndarray, target_image: np.ndarray) -> Tuple[Dict, Dict]:
        """
        Process a pair of face images (source and target).
        
        Args:
            source_detection: Detection results for source image
            target_detection: Detection results for target image
            source_image: Source image
            target_image: Target image
            
        Returns:
            Optimized detection results for source and target
        """
        # Extract and organize detection information
        source_info = self._extract_detection_info(source_detection)
        target_info = self._extract_detection_info(target_detection)
        
        # Get neighbor connections for landmarks
        neighbor_list = self._get_all_connected_neighbors(self.num_landmarks, self.connections)
        
        # Map points from source to target
        mapped_points, target_avg_min_dist = self.face_mapper.process(
            source_info['centers'], source_image, target_image, neighbor_list)
        
        # Convert mapped points back to pixel coordinates
        target_height, target_width = target_image.shape[:2]
        mapped_pixel_points = [
            [round(p[0] * target_height), round(p[1] * target_width)]
            for p in mapped_points
        ]
        
        # Match points
        matches = self.point_matcher.match(
            mapped_pixel_points, target_info['centers'],
            source_info['scores'], target_info['scores'],
            source_info['classes'], target_info['classes'],
            threshold=target_avg_min_dist * self.distance_scale
        )
        
        # Optimize with matches
        source_optimized, target_optimized = self._optimize_with_matches(
            copy.deepcopy(source_info), copy.deepcopy(target_info), matches)
        
        return source_optimized, target_optimized

    def process(self, detection_results: Dict, image_dict: Dict[str, np.ndarray]) -> Dict:
        """
        Process multi-view facial images and optimize detection results.
        
        Args:
            detection_results: Detection results for left, front, right views
            image_dict: Dictionary mapping view names to images
            
        Returns:
            Optimized detection results for all views
            
        Raises:
            RuntimeError: If optimizer not initialized or input invalid
        """
        if not self.initialized:
            raise RuntimeError("MultiViewOptimizer not initialized")
        
        if not isinstance(image_dict, dict) or not isinstance(detection_results, dict):
            raise RuntimeError("Invalid input parameters")
        
        # Process left-front pair
        left_optimized, front_optimized_left = self._process_single_face_pair(
            detection_results['left'], detection_results['front'],
            image_dict['left'], image_dict['front'])
        
        # Process right-front pair (updating front results)
        right_optimized, front_optimized = self._process_single_face_pair(
            detection_results['right'], front_optimized_left,
            image_dict['right'], image_dict['front'])
        
        return {
            'left': left_optimized,
            'front': front_optimized,
            'right': right_optimized
        }