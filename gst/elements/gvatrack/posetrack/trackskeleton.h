/*******************************************************************************
 * Copyright (C) 2018-2019 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************/

#pragma once

#include "itracker.h"

#include "video_frame.h"
#include <gst/gst.h>
#include <gst/video/video.h>
#include <memory>
#include <mutex>

#include <unordered_map>
namespace skeletontracker {
class Tracker : public ITracker {
  protected:
    std::vector<GVA::Tensor> poses;
    std::vector<GVA::Tensor> unfound_tensors;
    int object_id = 0;
    float threshold;
    int frame_width;
    int frame_height;

  public:
    Tracker(int _frame_width, int _frame_height, std::vector<GVA::Tensor> _poses = std::vector<GVA::Tensor>(),
            std::vector<GVA::Tensor> _unfound_tensors = std::vector<GVA::Tensor>(), int _object_id = 0,
            float _threshold = 0.5f);
    ~Tracker() = default;
    void track(GstBuffer *buffer) override;
    static ITracker *Create(const GstVideoInfo *video_info);
    float Distance(const GVA::Tensor &tensor, const GVA::Tensor &pose);
    void RedefinitionObjectsId(std::vector<GVA::Tensor> tensors);
    void SaveLiveIfObjectsLoss(std::vector<GVA::Tensor> &_poses, const std::vector<GVA::Tensor> &tensors,
                               std::vector<GVA::Tensor> &_unfound_tensors);
    void RewriteOldObjects(std::vector<GVA::Tensor> &_poses, const std::vector<GVA::Tensor> &tensors,
                           std::vector<GVA::Tensor> &_unfound_tensors);
};
} // namespace skeletontracker