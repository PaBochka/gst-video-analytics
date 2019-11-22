#include "gvaskeleton.h"

#include "gva_buffer_map.h"
#include "inference_backend/logger.h"
#include <gst/allocators/gstfdmemory.h>

#include "human_pose_estimator.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace human_pose_estimation;

GvaSkeletonStatus hpe_initialization(HumanPoseEstimator *hpe_obj, char *model_path, char *device) {
    try {
        std::string device_str = "CPU";
        std::string model_path_str = "/home/dsmertin/video-analytics/all_models/r1_models/intel/Transportation/"
                                     "human_pose_estimation/mobilenet-v1/dldt/human-pose-estimation-0001.xml";
        hpe_obj = new HumanPoseEstimator(model_path_str, device_str);
        if (hpe_obj != nullptr) {
            return GVA_SKELETON_OK;
        }
    } catch (const std::exception &e) {
        GVA_ERROR(e.what());
    }
    return GVA_SKELETON_ERROR;
}
GvaSkeletonStatus hpe_release(HumanPoseEstimator *hpe_obj) {
    try {
        delete hpe_obj;
        hpe_obj = NULL;
        return GVA_SKELETON_OK;
    } catch (const std::exception &e) {
        GVA_ERROR(e.what());
        return GVA_SKELETON_ERROR;
    }
}

int Fourcc2OpenCVType(int fourcc) {
    switch (fourcc) {
    case InferenceBackend::FOURCC_NV12:
        return CV_8UC1; // only Y plane
    case InferenceBackend::FOURCC_BGRA:
        return CV_8UC4;
    case InferenceBackend::FOURCC_BGRX:
        return CV_8UC4;
    case InferenceBackend::FOURCC_BGRP:
        return 0;
    case InferenceBackend::FOURCC_BGR:
        return CV_8UC3;
    case InferenceBackend::FOURCC_RGBA:
        return CV_8UC4;
    case InferenceBackend::FOURCC_RGBX:
        return CV_8UC4;
    case InferenceBackend::FOURCC_RGBP:
        return 0;
    case InferenceBackend::FOURCC_I420:
        return CV_8UC1; // only Y plane
    }
    return 0;
}

GvaSkeletonStatus attach_poses_to_buffer(const std::vector<HumanPose> &poses, GstBuffer *buf) {
    try {
        // TODO: postproc

        return GVA_SKELETON_OK;
    } catch (const std::exception &e) {
        GVA_ERROR(e.what());
    }
    return GVA_SKELETON_ERROR;
}
GvaSkeletonStatus hpe_to_estimate(HumanPoseEstimator *hpe_obj, GstBuffer *buf, GstVideoInfo *info) {
    try {
        InferenceBackend::Image image;

        BufferMapContext mapContext;
        GstMemory *mem = gst_buffer_get_memory(buf, 0);
        GstMapFlags mapFlags = (mem && gst_is_fd_memory(mem)) ? GST_MAP_READWRITE : GST_MAP_READ; // TODO
        gst_memory_unref(mem);
        //  should be unmapped with gst_buffer_unmap() after usage
        gva_buffer_map(buf, image, mapContext, info, InferenceBackend::MemoryType::SYSTEM, mapFlags);
        int format = Fourcc2OpenCVType(image.format);

        const cv::Mat mat(image.height, image.width, format, image.planes[0], info->stride[0]);
        if (mat.empty())
            throw std::runtime_error("Uppsss... Preproc has not happend.");

        std::vector<HumanPose> poses = hpe_obj->estimate(mat);
        gva_buffer_unmap(buf, image, mapContext);

        if (attach_poses_to_buffer(poses, buf) == GVA_SKELETON_ERROR)
            throw std::runtime_error("Uppsss... Postproc has not happend.");

        return GVA_SKELETON_OK;
    } catch (const std::exception &e) {
        GVA_ERROR(e.what());
    }
    return GVA_SKELETON_ERROR;
}