// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "CoreMinimal.h"

#include "PreOpenCVHeaders.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "PostOpenCVHeaders.h"

/**
 * 
 */
class CARSIMULATOR_API YuNet
{
public:
	YuNet();
	~YuNet();
    YuNet(const std::string& model_path,
        const cv::Size& input_size,
        float conf_threshold,
        float nms_threshold,
        int top_k,
        int backend_id,
        int target_id);

    void setBackendAndTarget(int backend_id, int target_id);
    void setInputSize(const cv::Size& input_size);
    cv::Mat infer(const cv::Mat image);

private:
    cv::Ptr<cv::FaceDetectorYN> model;

    std::string model_path_;
    cv::Size input_size_;
    float conf_threshold_;
    float nms_threshold_;
    int top_k_;
    int backend_id_;
    int target_id_;
};
