/*******************************************************************************
 * Copyright (C) 2018-2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#include "trackskeleton.h"
#include <math.h>

using namespace skeletontracker;
Tracker::Tracker(int _frame_width, int _frame_height, std::vector<GVA::Tensor> _poses, std::vector<int> _unique_id_vec,
                 int _object_id, float _threshold)
    : poses(_poses), unique_id_vec(_unique_id_vec), object_id(_object_id), threshold(_threshold),
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
                poses.push_back(gst_structure_copy(tensor.gst_structure()));
                unique_id_vec.push_back(object_id);
            }
        }
    } else {
        std::vector<std::pair<float, int>> matches;
        std::vector<std::vector<std::pair<float, int>>> matrix_distance;
        for (auto &tensor : frame.tensors()) {
            if (tensor.is_human_pose())
                for (auto &pose : poses) {
                    // matches.insert({CosDistance(CenterGravity(tensor), CenterGravity(pose)), pose});
                    matches.push_back({Distance(tensor, pose), pose.get_int("object_id")});
                }
            matrix_distance.push_back(matches);

            // if (matches.begin()->first < threshold) {
            //     tensor.set_int("object_id", matches.begin()->second.get_int("object_id"));
            // } else {
            //     tensor.set_int("object_id", ++object_id);
            // }

            matches.clear();
        }
        std::vector<int> vec_id;
        auto max_id = *std::min_element(
            matrix_distance[0].cbegin(), matrix_distance[0].cend(),
            [](const std::pair<float, int> &lhs, const std::pair<float, int> &rhs) { return lhs.second < rhs.second; });
        for (int i = 0; i < matrix_distance.size(); ++i) {
            auto minimum = matrix_distance[0][0];
            int min_col_index = 0;
            int min_row_index = i;
            for (int j = 0; j < matrix_distance[i].size(); ++j) {
                if (minimum > matrix_distance[i][j]) {
                    minimum = matrix_distance[i][j];
                    min_col_index = j;
                }
                matrix_distance[i][j].first *= 100000;
            }
            for (int row = i; row < matrix_distance.size(); ++row) {
                if (minimum > matrix_distance[row][min_col_index]) {
                    minimum = matrix_distance[row][min_col_index];
                    min_row_index = row;
                }
                matrix_distance[row][min_col_index].first *= 100000;
            }
            auto it = std::find(vec_id.begin(), vec_id.end(), matrix_distance[min_row_index][min_col_index].second);
            if (it != vec_id.end())
                frame.tensors()[min_row_index].set_int("object_id", ++max_id.second);
            else
                frame.tensors()[min_row_index].set_int("object_id",
                                                       matrix_distance[min_row_index][min_col_index].second);
            vec_id.push_back(matrix_distance[min_row_index][min_col_index].second);
        }
    }
    poses.clear();
    for (auto &tensor : frame.tensors()) {
        if (tensor.is_human_pose()) {
            poses.push_back(gst_structure_copy(tensor.gst_structure()));
        }
    }
}

void Tracker::MatchIdForTensors(std::vector<std::vector<std::pair<float, int>>> matrix_distance,
                                std::vector<GVA::Tensor> tensors) {
    std::vector<int> vec_id;
    auto max_id = *std::min_element(
        matrix_distance[0].cbegin(), matrix_distance[0].cend(),
        [](const std::pair<float, int> &lhs, const std::pair<float, int> &rhs) { return lhs.second < rhs.second; });
    for (int i = 0; i < matrix_distance.size(); ++i) {
        auto minimum = matrix_distance[0][0];
        int min_col_index = 0;
        int min_row_index = i;
        for (int j = 0; j < matrix_distance[i].size(); ++j) {
            if (minimum > matrix_distance[i][j]) {
                minimum = matrix_distance[i][j];
                min_col_index = j;
            }
            matrix_distance[i][j].first *= 100000;
        }
        for (int row = i; row < matrix_distance.size(); ++row) {
            if (minimum > matrix_distance[row][min_col_index]) {
                minimum = matrix_distance[row][min_col_index];
                min_row_index = row;
            }
            matrix_distance[row][min_col_index].first *= 100000;
        }
        auto it = std::find(vec_id.begin(), vec_id.end(), matrix_distance[min_row_index][min_col_index].second);
        if (it != vec_id.end())
            tensors[min_row_index].set_int("object_id", ++max_id.second);
        else
            tensors[min_row_index].set_int("object_id", matrix_distance[min_row_index][min_col_index].second);
        vec_id.push_back(matrix_distance[min_row_index][min_col_index].second);
    }
}

void Tracker::AppendObject(std::vector<GVA::Tensor> &tensors) {

    for (auto &tensor_with_unique_id : tensors) {
        int un_id = 0;
        for (auto &tensor : tensors) {
            if (tensor_with_unique_id.get_int("object_id") == tensor.get_int("object_id"))
                un_id++;
        }
        if (un_id > 1) {
            int new_id = *std::max_element(unique_id_vec.begin(), unique_id_vec.end()) + 1;
            tensor_with_unique_id.set_int("object_id", new_id);
            unique_id_vec.push_back(new_id);
        }
    }
}

void Tracker::DropObject(const std::vector<GVA::Tensor> &tensors, std::vector<int> &unique_id_vec) {
    std::vector<int> new_vec_id;
    for (auto it = unique_id_vec.begin(); it != unique_id_vec.end(); ++it) {
        int matches_count = 0;
        for (auto tensor : tensors) {
            int id = tensor.get_int("object_id");
            if (id == *it) {
                matches_count++;
            }
        }
        if (matches_count == 0) {
            *it = -1;
        }
    }
    std::vector<int>::iterator pend = std::remove(unique_id_vec.begin(), unique_id_vec.end(), -1);
    for (std::vector<int>::iterator p = unique_id_vec.begin(); p != pend; ++p)
        new_vec_id.push_back(*p);
    unique_id_vec.clear();
    unique_id_vec = new_vec_id;
}

float Tracker::CosDistance(cv::Point2f center_tensor, cv::Point2f center_pose) {
    float distance = 1 - ((center_tensor.x * center_pose.x + center_tensor.y * center_pose.y) /
                          (sqrt(pow(center_tensor.x, 2) + pow(center_tensor.y, 2)) *
                           sqrt(pow(center_pose.x, 2) + pow(center_pose.y, 2))));
    return distance;
}

float Tracker::EuclideanDistance(cv::Point2f center_tensor, cv::Point2f center_pose) {
    float distance = sqrt(pow((center_tensor.x - center_pose.x) / frame_width, 2) +
                          pow((center_tensor.y - center_pose.y) / frame_height, 2));
    return distance;
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

cv::Point2f Tracker::CenterGravity(const GVA::Tensor &tensor) {
    float p_head =
        0.07f; // подбор из наобилее заметных и по количеству точек. еще делится на количество точек на области:5
    float p_r_hand = 0.05f;   // 2
    float p_l_hand = 0.05f;   // 2
    float p_r_foot = 0.0375f; // 2
    float p_l_foot = 0.0375f; // 2
    float p_body = 0.06f;     // 5
    float xc = 0;
    float yc = 0;

    xc += tensor.get_double("nose_x") > 0 ? tensor.get_double("nose_x") * p_head : 0;
    xc += tensor.get_double("r_eye_x") > 0 ? tensor.get_double("r_eye_x") * p_head : 0;
    xc += tensor.get_double("l_eye_x") > 0 ? tensor.get_double("l_eye_x") * p_head : 0;
    xc += tensor.get_double("r_ear_x") > 0 ? tensor.get_double("r_ear_x") * p_head : 0;
    xc += tensor.get_double("l_ear_x") > 0 ? tensor.get_double("l_ear_x") * p_head : 0;
    xc += tensor.get_double("r_cubit_x") > 0 ? tensor.get_double("r_cubit_x") * p_r_hand : 0;
    xc += tensor.get_double("l_cubit_x") > 0 ? tensor.get_double("l_cubit_x") * p_l_hand : 0;
    xc += tensor.get_double("r_hand_x") > 0 ? tensor.get_double("r_hand_x") * p_r_hand : 0;
    xc += tensor.get_double("l_hand_x") > 0 ? tensor.get_double("l_hand_x") * p_l_hand : 0;
    xc += tensor.get_double("r_knee_x") > 0 ? tensor.get_double("r_knee_x") * p_r_foot : 0;
    xc += tensor.get_double("l_knee_x") > 0 ? tensor.get_double("l_knee_x") * p_l_foot : 0;
    xc += tensor.get_double("r_foot_x") > 0 ? tensor.get_double("r_foot_x") * p_r_foot : 0;
    xc += tensor.get_double("l_foot_x") > 0 ? tensor.get_double("l_foot_x") * p_l_foot : 0;
    xc += tensor.get_double("neck_x") > 0 ? tensor.get_double("neck_x") * p_body : 0;
    xc += tensor.get_double("r_shoulder_x") > 0 ? tensor.get_double("r_shoulder_x") * p_body : 0;
    xc += tensor.get_double("l_shoulder_x") > 0 ? tensor.get_double("l_shoulder_x") * p_body : 0;
    xc += tensor.get_double("r_hip_x") > 0 ? tensor.get_double("r_hip_x") * p_body : 0;
    xc += tensor.get_double("l_hip_x") > 0 ? tensor.get_double("l_hip_x") * p_body : 0;

    yc += tensor.get_double("nose_y") > 0 ? tensor.get_double("nose_y") * p_head : 0;
    yc += tensor.get_double("r_eye_y") > 0 ? tensor.get_double("r_eye_y") * p_head : 0;
    yc += tensor.get_double("l_eye_y") > 0 ? tensor.get_double("l_eye_y") * p_head : 0;
    yc += tensor.get_double("r_ear_y") > 0 ? tensor.get_double("r_ear_y") * p_head : 0;
    yc += tensor.get_double("l_ear_y") > 0 ? tensor.get_double("l_ear_y") * p_head : 0;
    yc += tensor.get_double("r_cubit_y") > 0 ? tensor.get_double("r_cubit_y") * p_r_hand : 0;
    yc += tensor.get_double("l_cubit_y") > 0 ? tensor.get_double("l_cubit_y") * p_l_hand : 0;
    yc += tensor.get_double("r_hand_y") > 0 ? tensor.get_double("r_hand_y") * p_r_hand : 0;
    yc += tensor.get_double("l_hand_y") > 0 ? tensor.get_double("l_hand_y") * p_l_hand : 0;
    yc += tensor.get_double("r_knee_y") > 0 ? tensor.get_double("r_knee_y") * p_r_foot : 0;
    yc += tensor.get_double("l_knee_y") > 0 ? tensor.get_double("l_knee_y") * p_l_foot : 0;
    yc += tensor.get_double("r_foot_y") > 0 ? tensor.get_double("r_foot_y") * p_r_foot : 0;
    yc += tensor.get_double("l_foot_y") > 0 ? tensor.get_double("l_foot_y") * p_l_foot : 0;
    yc += tensor.get_double("neck_y") > 0 ? tensor.get_double("neck_y") * p_body : 0;
    yc += tensor.get_double("r_shoulder_y") > 0 ? tensor.get_double("r_shoulder_y") * p_body : 0;
    yc += tensor.get_double("l_shoulder_y") > 0 ? tensor.get_double("l_shoulder_y") * p_body : 0;
    yc += tensor.get_double("r_hip_y") > 0 ? tensor.get_double("r_hip_y") * p_body : 0;
    yc += tensor.get_double("l_hip_y") > 0 ? tensor.get_double("l_hip_y") * p_body : 0;
    return cv::Point2f(xc, yc);
}

float Tracker::Lans_Will_Distance(const GVA::Tensor &tensor, const GVA::Tensor &pose) {

    float nose = (abs(tensor.get_double("nose_x") - pose.get_double("nose_x")) +
                  abs(tensor.get_double("nose_y") - pose.get_double("nose_y"))) /
                 (tensor.get_double("nose_x") + pose.get_double("nose_x") + tensor.get_double("nose_y") +
                  pose.get_double("nose_y"));
    float neck = (abs(tensor.get_double("neck_x") - pose.get_double("neck_x")) +
                  abs(tensor.get_double("neck_y") - pose.get_double("neck_y"))) /
                 (tensor.get_double("neck_x") + pose.get_double("neck_x") + tensor.get_double("neck_y") +
                  pose.get_double("neck_y"));
    float r_shoulder = (abs(tensor.get_double("r_shoulder_x") - pose.get_double("r_shoulder_x")) +
                        abs(tensor.get_double("r_shoulder_y") - pose.get_double("r_shoulder_y"))) /
                       (tensor.get_double("r_shoulder_x") + pose.get_double("r_shoulder_x") +
                        tensor.get_double("r_shoulder_y") + pose.get_double("r_shoulder_y"));
    float r_cubit = (abs(tensor.get_double("r_cubit_x") - pose.get_double("r_cubit_x")) +
                     abs(tensor.get_double("r_cubit_y") - pose.get_double("r_cubit_y"))) /
                    (tensor.get_double("r_cubit_x") + pose.get_double("r_cubit_x") + tensor.get_double("r_cubit_y") +
                     pose.get_double("r_cubit_y"));
    float r_hand = (abs(tensor.get_double("r_hand_x") - pose.get_double("r_hand_x")) +
                    abs(tensor.get_double("r_hand_y") - pose.get_double("r_hand_y"))) /
                   (tensor.get_double("r_hand_x") + pose.get_double("r_hand_x") + tensor.get_double("r_hand_y") +
                    pose.get_double("r_hand_y"));
    float l_shoulder = (abs(tensor.get_double("l_shoulder_x") - pose.get_double("l_shoulder_x")) +
                        abs(tensor.get_double("l_shoulder_y") - pose.get_double("l_shoulder_y"))) /
                       (tensor.get_double("l_shoulder_x") + pose.get_double("l_shoulder_x") +
                        tensor.get_double("l_shoulder_y") + pose.get_double("l_shoulder_y"));
    float l_cubit = (abs(tensor.get_double("l_cubit_x") - pose.get_double("l_cubit_x")) +
                     abs(tensor.get_double("l_cubit_y") - pose.get_double("l_cubit_y"))) /
                    (tensor.get_double("l_cubit_x") + pose.get_double("l_cubit_x") + tensor.get_double("l_cubit_y") +
                     pose.get_double("l_cubit_y"));
    float l_hand = (abs(tensor.get_double("l_hand_x") - pose.get_double("l_hand_x")) +
                    abs(tensor.get_double("l_hand_y") - pose.get_double("l_hand_y"))) /
                   (tensor.get_double("l_hand_x") + pose.get_double("l_hand_x") + tensor.get_double("l_hand_y") +
                    pose.get_double("l_hand_y"));
    float r_hip = (abs(tensor.get_double("r_hip_x") - pose.get_double("r_hip_x")) +
                   abs(tensor.get_double("r_hip_y") - pose.get_double("r_hip_y"))) /
                  (tensor.get_double("r_hip_x") + pose.get_double("r_hip_x") + tensor.get_double("r_hip_y") +
                   pose.get_double("r_hip_y"));
    float r_knee = (abs(tensor.get_double("r_knee_x") - pose.get_double("r_knee_x")) +
                    abs(tensor.get_double("r_knee_y") - pose.get_double("r_knee_y"))) /
                   (tensor.get_double("r_knee_x") + pose.get_double("r_knee_x") + tensor.get_double("r_knee_y") +
                    pose.get_double("r_knee_y"));
    float r_foot = (abs(tensor.get_double("r_foot_x") - pose.get_double("r_foot_x")) +
                    abs(tensor.get_double("r_foot_y") - pose.get_double("r_foot_y"))) /
                   (tensor.get_double("r_foot_x") + pose.get_double("r_foot_x") + tensor.get_double("r_foot_y") +
                    pose.get_double("r_foot_y"));
    float l_hip = (abs(tensor.get_double("l_hip_x") - pose.get_double("l_hip_x")) +
                   abs(tensor.get_double("l_hip_y") - pose.get_double("l_hip_y"))) /
                  (tensor.get_double("l_hip_x") + pose.get_double("l_hip_x") + tensor.get_double("l_hip_y") +
                   pose.get_double("l_hip_y"));
    float l_knee = (abs(tensor.get_double("l_knee_x") - pose.get_double("l_knee_x")) +
                    abs(tensor.get_double("l_knee_y") - pose.get_double("l_knee_y"))) /
                   (tensor.get_double("l_knee_x") + pose.get_double("l_knee_x") + tensor.get_double("l_knee_y") +
                    pose.get_double("l_knee_y"));
    float l_foot = (abs(tensor.get_double("l_foot_x") - pose.get_double("l_foot_x")) +
                    abs(tensor.get_double("l_foot_y") - pose.get_double("l_foot_y"))) /
                   (tensor.get_double("l_foot_x") + pose.get_double("l_foot_x") + tensor.get_double("l_foot_y") +
                    pose.get_double("l_foot_y"));
    float r_eye = (abs(tensor.get_double("r_eye_x") - pose.get_double("r_eye_x")) +
                   abs(tensor.get_double("r_eye_y") - pose.get_double("r_eye_y"))) /
                  (tensor.get_double("r_eye_x") + pose.get_double("r_eye_x") + tensor.get_double("r_eye_y") +
                   pose.get_double("r_eye_y"));
    float l_eye = (abs(tensor.get_double("l_eye_x") - pose.get_double("l_eye_x")) +
                   abs(tensor.get_double("l_eye_y") - pose.get_double("l_eye_y"))) /
                  (tensor.get_double("l_eye_x") + pose.get_double("l_eye_x") + tensor.get_double("l_eye_y") +
                   pose.get_double("l_eye_y"));
    float r_ear = (abs(tensor.get_double("r_ear_x") - pose.get_double("r_ear_x")) +
                   abs(tensor.get_double("r_ear_y") - pose.get_double("r_ear_y"))) /
                  (tensor.get_double("r_ear_x") + pose.get_double("r_ear_x") + tensor.get_double("r_ear_y") +
                   pose.get_double("r_ear_y"));
    float l_ear = (abs(tensor.get_double("l_ear_x") - pose.get_double("l_ear_x")) +
                   abs(tensor.get_double("l_ear_y") - pose.get_double("l_ear_y"))) /
                  (tensor.get_double("l_ear_x") + pose.get_double("l_ear_x") + tensor.get_double("l_ear_y") +
                   pose.get_double("l_ear_y"));

    float distance = (nose + neck + r_shoulder + r_cubit + r_hand + l_shoulder + l_cubit + l_hand + r_hip + r_knee +
                      r_foot + l_hip + l_knee + l_foot + r_eye + l_eye + r_ear + l_ear);
    return distance;
}