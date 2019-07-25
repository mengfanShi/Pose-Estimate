# -*- coding:utf-8 -*-
# @TIME     :2019/3/8 16:17
# @Author   :Fan
# @File     :Parser.py
"""
Purpose: Build the skeleton using PAF
"""
import math
import cv2
import numpy as np
import matplotlib.cm
from scipy.ndimage.filters import gaussian_filter, maximum_filter
from scipy.ndimage.morphology import generate_binary_structure

# To build limb by two joints, wo need to find their relationships
# index means the assigned limb of two joints
limb_heatmap = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6],
                [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
                [11, 12], [12, 13], [1, 0], [0, 14], [14, 16],
                [0, 15], [15, 17], [2, 16], [5, 17]]

# To get the index in our predicted PAF
limb_paf = [[12, 13], [20, 21], [14, 15], [16, 17], [22, 23],
            [24, 25], [0, 1], [2, 3], [4, 5], [6, 7], [8, 9],
            [10, 11], [28, 29], [30, 31], [34, 35], [32, 33],
            [36, 37], [18, 19], [26, 27]]

# To plot different joints and limbs
colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
          [170, 255, 0],[85, 255, 0], [0, 255, 0], [0, 255, 85],
          [0, 255, 170], [0, 255, 255],[0, 170, 255], [0, 85, 255],
          [0, 0, 255], [85, 0, 255], [170, 0, 255],[255, 0, 255],
          [255, 0, 170], [255, 0, 85], [255, 0, 0]]

num_joints = 18
num_limbs = len(limb_heatmap)


def find_peaks(param, img):
    # Given a Gray_scale image, find local max
    # whose value is above a threshould (param['thre1']
    peaks_binary = (maximum_filter(img, footprint=
                generate_binary_structure(2, connectivity=1))
                    == img) * (img > param['thre1'])

    # Do the reverse, return [[x y], ...] instead of [[y x],...]
    return np.array(np.nonzero(peaks_binary)[::-1]).T


def compute_resize_coord(coord, resizeFactor):
    # Given the coordinate and the resize factor(make image bigger)
    # compute the resize image coordinates
    return (np.array(coord, dtype=float) + 0.5) * resizeFactor - 0.5


def NMS(param, Heatmap, upsampleFactor=1., refine_center=True,
        gaussian_filt=False):
    # Non Maximum Suppression: find peaks in a set of gray_scale images
    # upsampleFactor: size ratio between the input image and HeatMap
    # refine_center: to fine-tune the location of the peak at the
    #                resolution of the input image
    # Return: (num_joints, 4) array where each row represents a joint and
    #         cols indicate the {x,y} position, the score and an unique id

    joint_id = 0
    window_size = 2         # specifies how many pixels in each direction from peak
    joint_peaks = []

    for joint in range(num_joints):
        map_origin = Heatmap[:, :, joint]
        peak_coord = find_peaks(param, map_origin)
        peaks = np.zeros((len(peak_coord), 4))
        for i, peak in enumerate(peak_coord):
            if refine_center:
                x_min, y_min = np.maximum(0, peak - window_size)
                x_max, y_max = np.minimum(np.array(map_origin.T.shape) - 1,
                                          peak + window_size)

                # Take a small patch around each peak and only upSample that
                patch = map_origin[y_min:y_max+1, x_min:x_max+1]

                map_upsample = cv2.resize(patch, None, fx=upsampleFactor,
                                          fy=upsampleFactor, interpolation=cv2.INTER_CUBIC)

                if gaussian_filt:
                    map_upsample = gaussian_filter(map_upsample, sigma=3)

                location_max = np.unravel_index(map_upsample.argmax(), map_upsample.shape)
                location_center = compute_resize_coord(peak[::-1] - [y_min, x_min], upsampleFactor)

                refined_center = location_max - location_center
                peak_score = map_upsample[location_max]
            else:
                refined_center = [0, 0]
                peak_score = map_origin[tuple(peak[::-1])]

            peaks[i, :] = tuple([int(round(x)) for x in compute_resize_coord(
                peak_coord[i], upsampleFactor) + refined_center[::-1]]) + (peak_score, joint_id)

            joint_id += 1
        joint_peaks.append(peaks)
    return joint_peaks


def find_connected_joints(param, paf, joint_peaks, num_points=10):
    # Evaluate PAFs to determine which pair of joints are indeed body limbs
    # num_points: Int indicating how many intermediate points to take
    #             between joint_src and joint_dst, at which the PAFs will be evaluated
    # Return: List of num_limbs rows. For every limb_type (a row) we store
    #         a list of all limbs of that type found (eg: all the right forearms).

    # For each limb (each item in connected_limbs[limb_type]), we store 5 cells:
    #  {joint_src_id,joint_dst_id}: a unique number associated with each joint
    #  limb_score_penalizing_long_dist: a score of how good a connection
    #           of the joints is, penalized if the limb length is too long
    #  {joint_src_index,joint_dst_index}: the index of the joint within
    #           all the joints of that type found (eg: the 3rd right elbow found)
    connected_limbs = []

    # Auxiliary array to access pafle quickly
    limb_intermed_coords = np.empty((4, num_points), dtype=np.intp)
    for limb_type in range(num_limbs):
        joints_src = joint_peaks[limb_heatmap[limb_type][0]]
        joints_dst = joint_peaks[limb_heatmap[limb_type][1]]

        if len(joints_src) == 0 or len(joints_dst) == 0:
            connected_limbs.append([])
        else:
            connection_candidates = []
            # Specify the paf index that contains the xy-coord of the paf
            limb_intermed_coords[2, :] = limb_paf[limb_type][0]
            limb_intermed_coords[3, :] = limb_paf[limb_type][1]
            for i, joint_src in enumerate(joints_src):
                # Try every possible joints_src[i]-joints_dst[j] pair and see
                # if it's a feasible limb
                for j, joint_dst in enumerate(joints_dst):
                    # Subtract the position of both joints to obtain the
                    # direction of the potential limb
                    limb_dir = joint_dst[:2] - joint_src[:2]

                    # Compute the distance/length of the potential limb (norm of limb_dir)
                    limb_dist = np.sqrt(np.sum(limb_dir ** 2)) + 1e-8
                    limb_dir = limb_dir / limb_dist

                    # Linearly distribute num_points points from the x
                    # coordinate of joint_src to the x coordinate of joint_dst
                    limb_intermed_coords[1, :] = np.round(np.linspace(
                        joint_src[0], joint_dst[0], num=num_points))
                    limb_intermed_coords[0, :] = np.round(np.linspace(
                        joint_src[1], joint_dst[1], num=num_points))

                    # Find the corresponding value in the PAF
                    intermed_paf = paf[limb_intermed_coords[0, :],
                                       limb_intermed_coords[1, :],
                                       limb_intermed_coords[2:4, :]].T

                    score_intermed_pts = intermed_paf.dot(limb_dir)

                    # Criterion 1: At least 80% of the intermediate points have
                    # a score higher than thre2
                    criterion1 = (np.count_nonzero(
                        score_intermed_pts > param['thre2']) > 0.8 * num_points)

                    # Criterion 2: Mean score, penalized for large limb
                    # distances (larger than half the image height), is positive
                    score_penalizing_long_dist = score_intermed_pts.mean(
                    ) + min(0.5 * paf.shape[0] / limb_dist - 1, 0)

                    criterion2 = (score_penalizing_long_dist > 0)

                    if criterion1 and criterion2:
                        # Last value is the combined paf(+limb_dist) +
                        # heatmap scores of both joints
                        connection_candidates.append(
                            [i, j, score_penalizing_long_dist,
                             score_penalizing_long_dist + joint_src[2] + joint_dst[2]])

            # Sort connection candidates based on their score_penalizing_long_dist
            connection_candidates = sorted(
                connection_candidates, key=lambda x: x[2], reverse=True)
            connections = np.empty((0, 5))
            # There can only be as many limbs as the smallest number of source or destination joints
            max_connections = min(len(joints_src), len(joints_dst))

            # Traverse all potential joint connections (sorted by their score)
            for potential_connection in connection_candidates:
                i, j, s = potential_connection[0:3]
                # Make sure joints_src[i] or joints_dst[j] haven't already been
                # connected to other joints_dst or joints_src
                if i not in connections[:, 3] and j not in connections[:, 4]:
                    # [joint_src_id, joint_dst_id, limb_score_penalizing_long_dist,
                    # joint_src_index, joint_dst_index]
                    connections = np.vstack(
                        [connections, [joints_src[i][3], joints_dst[j][3], s, i, j]])
                    # each joint can't be connected to more than one joint
                    if len(connections) >= max_connections:
                        break
            connected_limbs.append(connections)

    return connected_limbs


def group_limbs_of_same_person(connected_limbs, joint_list):
    """
    Associate limbs belonging to the same person together.
    Return: 2D np.array of (num_people, num_joints+2)
    For each person found:
    # 1rt num_joints columns contain the index (in joint_list) of the joints associated
        with that person (or -1 if their i-th joint wasn't found)
    # 2nd-to-last column: Overall score of the joints+limbs that belong to this person
    # 3rd-to-last column: Total count of joints found for this person
    """
    person_to_joint_assoc = []

    for limb_type in range(num_limbs):
        joint_src_type, joint_dst_type = limb_heatmap[limb_type]

        for limb_info in connected_limbs[limb_type]:
            person_assoc_idx = []
            for person, person_limbs in enumerate(person_to_joint_assoc):
                if person_limbs[joint_src_type] == limb_info[0] \
                        or person_limbs[joint_dst_type] == limb_info[1]:
                    person_assoc_idx.append(person)

            # If one of the joints has been associated to a person, and either
            # the other joint is also associated with the same person or not
            # associated to anyone yet:
            if len(person_assoc_idx) == 1:
                person_limbs = person_to_joint_assoc[person_assoc_idx[0]]
                # If the other joint is not associated to anyone yet,
                # Associate it with the current person
                if person_limbs[joint_dst_type] != limb_info[1]:
                    person_limbs[joint_dst_type] = limb_info[1]
                    # Increase the number of limbs associated to this person
                    person_limbs[-1] += 1
                    # And update the total score
                    person_limbs[-2] += joint_list[limb_info[1]
                                                        .astype(int), 2] + limb_info[2]
                elif person_limbs[joint_src_type] != limb_info[0]:                                  #
                    person_limbs[joint_src_type] = limb_info[0]
                    person_limbs[-1] += 1
                    person_limbs[-2] += joint_list[limb_info[0]
                                                        .astype(int), 2] + limb_info[2]

            # if found 2 and disjoint, merge them
            elif len(person_assoc_idx) == 2:
                person1_limbs = person_to_joint_assoc[person_assoc_idx[0]]
                person2_limbs = person_to_joint_assoc[person_assoc_idx[1]]
                membership = ((person1_limbs >= 0) & (person2_limbs >= 0))[:-2]

                # If both people have no same joints connected, merge them into a single person
                # Update which joints are connected
                if not membership.any():
                    person1_limbs[:-2] += (person2_limbs[:-2] + 1)     # plus 1 because init -1
                    # Update the overall score and total count of joints
                    # connected by summing their counters
                    person1_limbs[-2:] += person2_limbs[-2:]
                    # Add the score of the current joint connection to the overall score
                    person1_limbs[-2] += limb_info[2]
                    person_to_joint_assoc.pop(person_assoc_idx[1])
                # Same case as len(person_assoc_idx)==1 above
                else:                                                                               #
                    if person1_limbs[joint_dst_type] != limb_info[1]:
                        person1_limbs[joint_dst_type] = limb_info[1]
                        person1_limbs[-1] += 1
                        person1_limbs[-2] += joint_list[limb_info[1]
                                                        .astype(int), 2] + limb_info[2]
                    elif person1_limbs[joint_src_type] != limb_info[0]:
                        person1_limbs[joint_src_type] = limb_info[0]
                        person1_limbs[-1] += 1
                        person1_limbs[-2] += joint_list[limb_info[0]
                                                        .astype(int), 2] + limb_info[2]

            # No person has claimed any of these joints, create a new person
            # Initialize person info to all -1 (no joint associations)
            else:
                row = -1 * np.ones(num_joints+2)
                # Store the joint info of the new connection
                row[joint_src_type] = limb_info[0]
                row[joint_dst_type] = limb_info[1]
                # Total count of connected joints for this person: 2
                row[-1] = 2
                # Compute overall score
                # {joint_src,joint_dst}
                row[-2] = sum(joint_list[limb_info[:2].astype(int), 2]
                              ) + limb_info[2]
                person_to_joint_assoc.append(row)

    # Delete people who have very few parts connected
    people_to_delete = []
    for person_id, person_info in enumerate(person_to_joint_assoc):
        if person_info[-1] < 3 or person_info[-2] / person_info[-1] < 0.2:
            people_to_delete.append(person_id)

    for index in people_to_delete[::-1]:
        person_to_joint_assoc.pop(index)

    # we treat the set of people as a list (fast to append items)
    # and only convert to np.array at the end
    return np.array(person_to_joint_assoc)


def plot_pose(img, joint_list, person_to_joint_assoc, bool_fast_plot=True,
              plot_ear_to_shoulder=False):
    # Make a copy so we don't modify the original image
    image = img.copy()

    limb_thickness = 4

    if plot_ear_to_shoulder:
        which_limbs_to_plot = num_limbs
    else:
        which_limbs_to_plot = num_limbs - 2

    for limb_type in range(which_limbs_to_plot):
        for person_joint_info in person_to_joint_assoc:
            joint_indices = person_joint_info[limb_heatmap[limb_type]].astype(int)
            if -1 in joint_indices:
                continue
            # joint_coords[:,0] represents Y coord of both joints
            # joint_coords[:,1], X coord
            joint_coords = joint_list[joint_indices, 0:2]

            for joint in joint_coords:
                cv2.circle(image, tuple(joint[0:2].astype(
                    int)), 4, (255, 255, 255), thickness=-1)

            # mean along the axis=0 computes meanmY coord and mean X coord
            coords_center = tuple(
                np.round(np.mean(joint_coords, 0)).astype(int))

            # joint_coords[0,:] is the coord of joint_src;
            # joint_coords[1,:] is the coord of joint_dst
            limb_dir = joint_coords[0, :] - joint_coords[1, :]
            limb_length = np.linalg.norm(limb_dir)
            # Get the angle of limb_dir in degrees using atan2(limb_dir_x, limb_dir_y)
            angle = math.degrees(math.atan2(limb_dir[1], limb_dir[0]))

            cur_image = image if bool_fast_plot else image.copy()
            polygon = cv2.ellipse2Poly(
                coords_center, (int(limb_length / 2), limb_thickness), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_image, polygon, colors[limb_type])

            if not bool_fast_plot:
                image = cv2.addWeighted(image, 0.4, cur_image, 0.6, 0)

    # to_plot is the location of all joints found overlaid on top of image
    if bool_fast_plot:
        to_plot = image.copy()
    else:
        to_plot = cv2.addWeighted(img, 0.3, image, 0.7, 0)

    return to_plot, image


def decode_pose(img, param, heatmaps, pafs):
    # Bottom-up approach:

    # Step 1: find all joints in the image
    joint_peaks = NMS(param, heatmaps,
                      img.shape[0] / float(heatmaps.shape[0]))
    
    joint_list = np.array([tuple(peak) + (joint_type,) for joint_type,
                           joint_peak in enumerate(joint_peaks)
                           for peak in joint_peak])

    # Step 2: find which joints go together to form limbs
    PAF = cv2.resize(pafs, (img.shape[1], img.shape[0]),
                     interpolation=cv2.INTER_CUBIC)
    connected_limbs = find_connected_joints(param,
                                            PAF, joint_peaks)
    
    # print(connected_limbs)
    # Step 3: associate limbs that belong to the same person
    person_to_joint_assoc = group_limbs_of_same_person(
        connected_limbs, joint_list)

    # Step 4: plot results
    to_plot, image = plot_pose(img, joint_list, person_to_joint_assoc)

    return to_plot, image, joint_list, person_to_joint_assoc

def people_to_pose(people_list):
    """
    covert people list to joint_list and person_to_joint_assoc
    :param people_list: generated by cpm.cpm_layer.rtpose_postprocess, a numpy array of [num_people, num_part, 3] => [x, y, score]
    :return: joint_list, person_to_joint_assoc
    """

    joint_list = []
    count = 0
    person_to_joint_assoc = np.empty(
        [len(people_list), people_list.shape[1] + 2], dtype=np.float)
    for i, person in enumerate(people_list):
        person_score = 0.
        person_count = 0
        for j, joint in enumerate(person):
            x, y, score = joint
            if score > 1e-4:
                joint_list.append([x, y, score, count, j])
                person_to_joint_assoc[i, j] = count
                count += 1
                person_score += score
                person_count += 1
            else:
                person_to_joint_assoc[i, j] = -1
        person_to_joint_assoc[i, -2] = person_score
        person_to_joint_assoc[i, -1] = person_count

    joint_list = np.asarray(joint_list, dtype=np.float)

    return joint_list, person_to_joint_assoc