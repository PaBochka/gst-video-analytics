/*******************************************************************************
 * Copyright (C) 2018-2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "trackskeleton.h"
#include <math.h>

using namespace skeletontracker;
Tracker::Tracker(int _frame_width, int _frame_height, std::vector<GVA::Tensor> _poses,
                 std::vector<GVA::Tensor> _unfound_tensors, int _object_id, float _threshold)
    : poses(_poses), unfound_tensors(_unfound_tensors), object_id(_object_id), threshold(_threshold),
      frame_width(_frame_width), frame_height(_frame_height) {
}

ITracker *Tracker::Create(const GstVideoInfo *video_info) {
    return new Tracker(video_info->width, video_info->height);
}

void Tracker::track(GstBuffer *buffer) {
    GVA::VideoFrame frame(buffer);
    if (poses.empty()) {
        for (auto &tensor : frame.tensors()) {
            if (tensor.is_human_pose()) {
                tensor.set_int("object_id", ++object_id);
                tensor.set_int("time_live", 10);
                poses.push_back(gst_structure_copy(tensor.gst_structure()));
            }
        }
    } else {
        std::multimap<float, GVA::Tensor> matches;
        for (auto &tensor : frame.tensors()) {
            if (tensor.is_human_pose())
                for (auto &pose : poses) {
                    matches.insert({Distance(tensor, pose), pose});
                }
            if (matches.begin()->first < threshold) {
                tensor.set_int("object_id", matches.begin()->second.get_int("object_id"));
                tensor.set_int("time_live", 10);

            } else {
                tensor.set_int("object_id", ++object_id);
                tensor.set_int("time_live", 10);
            }
            matches.clear();
        }
        RedefinitionObjectsId(frame.tensors());
        SaveLiveIfObjectsLoss(poses, frame.tensors(), unfound_tensors);
        RewriteOldObjects(poses, frame.tensors(), unfound_tensors);
    }
}

void Tracker::RedefinitionObjectsId(std::vector<GVA::Tensor> tensors) {

    for (auto &tensor_with_unique_id : tensors) {
        int unique_id = 0;
        for (auto &tensor : tensors) {
            if (tensor_with_unique_id.get_int("object_id") == tensor.get_int("object_id"))
                unique_id++;
        }
        if (unique_id > 1) {
            tensor_with_unique_id.set_int("object_id", ++object_id);
        }
    }
}
void Tracker::SaveLiveIfObjectsLoss(std::vector<GVA::Tensor> &_poses, const std::vector<GVA::Tensor> &tensors,
                                    std::vector<GVA::Tensor> &_unfound_tensors) {

    for (auto &pose : _poses) {
        if (pose.is_human_pose()) {
            int current_id = pose.get_int("object_id");
            bool is_found = false;
            for (auto &tensor : tensors) {
                if (tensor.get_int("object_id") == current_id) {
                    is_found = true;
                    break;
                }
            }
            if (!is_found) {
                int time_live = pose.get_int("time_live");
                if (time_live == 0)
                    continue;
                _unfound_tensors.push_back(pose);
                pose.set_int("time_live", --time_live);
            }
        }
    }
}

void Tracker::RewriteOldObjects(std::vector<GVA::Tensor> &_poses, const std::vector<GVA::Tensor> &tensors,
                                std::vector<GVA::Tensor> &_unfound_tensors) {
    _poses.clear();
    for (auto &tensor : tensors) {
        if (tensor.is_human_pose()) {
            _poses.push_back(gst_structure_copy(tensor.gst_structure()));
        }
    }
    for (auto &unfound_tensor : _unfound_tensors) {
        _poses.push_back(unfound_tensor);
    }
}

float Tracker::Distance(const GVA::Tensor &tensor, const GVA::Tensor &pose) {
    float nose = 0;
    float neck = 0;
    float r_shoulder = 0;
    float l_shoulder = 0;
    float r_cubit = 0;
    float r_hand = 0;
    float r_hip = 0;
    float r_knee = 0;
    float r_foot = 0;
    float r_eye = 0;
    float r_ear = 0;
    float l_cubit = 0;
    float l_hand = 0;
    float l_hip = 0;
    float l_knee = 0;
    float l_foot = 0;
    float l_eye = 0;
    float l_ear = 0;
    if (tensor.get_double("nose_x") > 0 && tensor.get_double("nose_y") > 0 && pose.get_double("nose_x") > 0 &&
        pose.get_double("nose_y") > 0)
        nose = sqrt(pow((tensor.get_double("nose_x") - pose.get_double("nose_x")) / frame_width, 2) +
                    pow((tensor.get_double("nose_y") - pose.get_double("nose_y")) / frame_height, 2));

    if (tensor.get_double("neck_x") > 0 && tensor.get_double("neck_y") > 0 && pose.get_double("neck_x") > 0 &&
        pose.get_double("neck_y") > 0)
        neck = sqrt(pow((tensor.get_double("neck_x") - pose.get_double("neck_x")) / frame_width, 2) +
                    pow((tensor.get_double("neck_y") - pose.get_double("neck_y")) / frame_height, 2));

    if (tensor.get_double("r_shoulder_x") > 0 && tensor.get_double("r_shoulder_y") > 0 &&
        pose.get_double("r_shoulder_x") > 0 && pose.get_double("r_shoulder_y") > 0)
        r_shoulder = sqrt(pow((tensor.get_double("r_shoulder_x") - pose.get_double("r_shoulder_x")) / frame_width, 2) +
                          pow((tensor.get_double("r_shoulder_y") - pose.get_double("r_shoulder_y")) / frame_height, 2));

    if (tensor.get_double("r_cubit_x") > 0 && tensor.get_double("r_cubit_y") > 0 && pose.get_double("r_cubit_x") > 0 &&
        pose.get_double("r_cubit_y") > 0)
        r_cubit = sqrt(pow((tensor.get_double("r_cubit_x") - pose.get_double("r_cubit_x")) / frame_width, 2) +
                       pow((tensor.get_double("r_cubit_y") - pose.get_double("r_cubit_y")) / frame_height, 2));

    if (tensor.get_double("r_hand_x") > 0 && tensor.get_double("r_hand_y") > 0 && pose.get_double("r_hand_x") > 0 &&
        pose.get_double("r_hand_y") > 0)
        r_hand = sqrt(pow((tensor.get_double("r_hand_x") - pose.get_double("r_hand_x")) / frame_width, 2) +
                      pow((tensor.get_double("r_hand_y") - pose.get_double("r_hand_y")) / frame_height, 2));

    if (tensor.get_double("l_shoulder_x") > 0 && tensor.get_double("l_shoulder_y") > 0 &&
        pose.get_double("l_shoulder_x") > 0 && pose.get_double("l_shoulder_y") > 0)
        l_shoulder = sqrt(pow((tensor.get_double("l_shoulder_x") - pose.get_double("l_shoulder_x")) / frame_width, 2) +
                          pow((tensor.get_double("l_shoulder_y") - pose.get_double("l_shoulder_y")) / frame_height, 2));

    if (tensor.get_double("l_cubit_x") > 0 && tensor.get_double("l_cubit_y") > 0 && pose.get_double("l_cubit_x") > 0 &&
        pose.get_double("l_cubit_y") > 0)
        l_cubit = sqrt(pow((tensor.get_double("l_cubit_x") - pose.get_double("l_cubit_x")) / frame_width, 2) +
                       pow((tensor.get_double("l_cubit_y") - pose.get_double("l_cubit_y")) / frame_height, 2));

    if (tensor.get_double("l_hand_x") > 0 && tensor.get_double("l_hand_y") > 0 && pose.get_double("l_hand_x") > 0 &&
        pose.get_double("l_hand_y") > 0)
        l_hand = sqrt(pow((tensor.get_double("l_hand_x") - pose.get_double("l_hand_x")) / frame_width, 2) +
                      pow((tensor.get_double("l_hand_y") - pose.get_double("l_hand_y")) / frame_height, 2));

    if (tensor.get_double("r_hip_x") > 0 && tensor.get_double("r_hip_y") > 0 && pose.get_double("r_hip_x") > 0 &&
        pose.get_double("r_hip_y") > 0)
        r_hip = sqrt(pow((tensor.get_double("r_hip_x") - pose.get_double("r_hip_x")) / frame_width, 2) +
                     pow((tensor.get_double("r_hip_y") - pose.get_double("r_hip_y")) / frame_height, 2));

    if (tensor.get_double("r_knee_x") > 0 && tensor.get_double("r_knee_y") > 0 && pose.get_double("r_knee_x") > 0 &&
        pose.get_double("r_knee_y") > 0)
        r_knee = sqrt(pow((tensor.get_double("r_knee_x") - pose.get_double("r_knee_x")) / frame_width, 2) +
                      pow((tensor.get_double("r_knee_y") - pose.get_double("r_knee_y")) / frame_height, 2));

    if (tensor.get_double("r_foot_x") > 0 && tensor.get_double("r_foot_y") > 0 && pose.get_double("r_foot_x") > 0 &&
        pose.get_double("r_foot_y") > 0)
        r_foot = sqrt(pow((tensor.get_double("r_foot_x") - pose.get_double("r_foot_x")) / frame_width, 2) +
                      pow((tensor.get_double("r_foot_y") - pose.get_double("r_foot_y")) / frame_height, 2));

    if (tensor.get_double("l_hip_x") > 0 && tensor.get_double("l_hip_y") > 0 && pose.get_double("l_hip_x") > 0 &&
        pose.get_double("l_hip_y") > 0)
        l_hip = sqrt(pow((tensor.get_double("l_hip_x") - pose.get_double("l_hip_x")) / frame_width, 2) +
                     pow((tensor.get_double("l_hip_y") - pose.get_double("l_hip_y")) / frame_height, 2));

    if (tensor.get_double("l_knee_x") > 0 && tensor.get_double("l_knee_y") > 0 && pose.get_double("l_knee_x") > 0 &&
        pose.get_double("l_knee_y") > 0)
        l_knee = sqrt(pow((tensor.get_double("l_knee_x") - pose.get_double("l_knee_x")) / frame_width, 2) +
                      pow((tensor.get_double("l_knee_y") - pose.get_double("l_knee_y")) / frame_height, 2));

    if (tensor.get_double("l_foot_x") > 0 && tensor.get_double("l_foot_y") > 0 && pose.get_double("l_foot_x") > 0 &&
        pose.get_double("l_foot_y") > 0)
        l_foot = sqrt(pow((tensor.get_double("l_foot_x") - pose.get_double("l_foot_x")) / frame_width, 2) +
                      pow((tensor.get_double("l_foot_y") - pose.get_double("l_foot_y")) / frame_height, 2));

    if (tensor.get_double("r_eye_x") > 0 && tensor.get_double("r_eye_y") > 0 && pose.get_double("r_eye_x") > 0 &&
        pose.get_double("r_eye_y") > 0)
        r_eye = sqrt(pow((tensor.get_double("r_eye_x") - pose.get_double("r_eye_x")) / frame_width, 2) +
                     pow((tensor.get_double("r_eye_y") - pose.get_double("r_eye_y")) / frame_height, 2));

    if (tensor.get_double("l_eye_x") > 0 && tensor.get_double("l_eye_y") > 0 && pose.get_double("l_eye_x") > 0 &&
        pose.get_double("l_eye_y") > 0)
        l_eye = sqrt(pow((tensor.get_double("l_eye_x") - pose.get_double("l_eye_x")) / frame_width, 2) +
                     pow((tensor.get_double("l_eye_y") - pose.get_double("l_eye_y")) / frame_height, 2));

    if (tensor.get_double("r_ear_x") > 0 && tensor.get_double("r_ear_y") > 0 && pose.get_double("r_ear_x") > 0 &&
        pose.get_double("r_ear_y") > 0)
        r_ear = sqrt(pow((tensor.get_double("r_ear_x") - pose.get_double("r_ear_x")) / frame_width, 2) +
                     pow((tensor.get_double("r_ear_y") - pose.get_double("r_ear_y")) / frame_height, 2));

    if (tensor.get_double("l_ear_x") > 0 && tensor.get_double("l_ear_y") > 0 && pose.get_double("l_ear_x") > 0 &&
        pose.get_double("l_ear_y") > 0)
        l_ear = sqrt(pow((tensor.get_double("l_ear_x") - pose.get_double("l_ear_x")) / frame_width, 2) +
                     pow((tensor.get_double("l_ear_y") - pose.get_double("l_ear_y")) / frame_height, 2));

    float distance = (nose + neck + r_shoulder + r_cubit + r_hand + l_shoulder + l_cubit + l_hand + r_hip + r_knee +
                      r_foot + l_hip + l_knee + l_foot + r_eye + l_eye + r_ear + l_ear) /
                     18;
    return distance;
}