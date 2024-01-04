// Fill out your copyright notice in the Description page of Project Settings.


#include "YuNet.h"

YuNet::YuNet()
{
}

YuNet::~YuNet()
{
}

YuNet::YuNet(const std::string& model_path,
    const cv::Size& input_size = cv::Size(320, 320),
    float conf_threshold = 0.6f,
    float nms_threshold = 0.3f,
    int top_k = 5000,
    int backend_id = 0,
    int target_id = 0)
    : model_path_(model_path), input_size_(input_size),
    conf_threshold_(conf_threshold), nms_threshold_(nms_threshold),
    top_k_(top_k), backend_id_(backend_id), target_id_(target_id)
{
    model = cv::FaceDetectorYN::create(model_path_, "", input_size_, conf_threshold_, nms_threshold_, top_k_, backend_id_, target_id_);
}

void YuNet::setBackendAndTarget(int backend_id, int target_id)
{
    backend_id_ = backend_id;
    target_id_ = target_id;
    model = cv::FaceDetectorYN::create(model_path_, "", input_size_, conf_threshold_, nms_threshold_, top_k_, backend_id_, target_id_);
}

void YuNet::setInputSize(const cv::Size& input_size)
{
    input_size_ = input_size;
    model->setInputSize(input_size_);
}

cv::Mat YuNet::infer(const cv::Mat image)
{
    cv::Mat res;
    model->detect(image, res);
    return res;
}

