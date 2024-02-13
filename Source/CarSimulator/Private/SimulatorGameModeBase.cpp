// Fill out your copyright notice in the Description page of Project Settings.


#include "SimulatorGameModeBase.h"
#include "Blueprint/UserWidget.h"
#include "MainWidget.h"
#include "Components/Image.h"
#include <chrono>



void ASimulatorGameModeBase::BeginPlay()
{
	Super::BeginPlay();
	capture = cv::VideoCapture(0);

	if (!capture.isOpened())
	{
		UE_LOG(LogTemp, Log, TEXT("Open Webcam failed"));
		return;
	}
	else
	{
		UE_LOG(LogTemp, Log, TEXT("Open Webcam Success"));
	}
	width = image.cols;
	height = image.rows;

    net_det = cv::dnn::readNetFromONNX("C:/palm_detection.onnx");
    nextIndex = 0; //tracking id

    wheelImage = cv::imread("C:/wheel.png", cv::IMREAD_UNCHANGED);
    cv::resize(wheelImage, wheelImage, cv::Size(240, 240));

    handRoiRect = cv::Rect(120, 240, 520, 240);
    objDetectRoiRect = cv::Rect(120, 40, 520, 440);
    detectInput = cv::Size(128, 128);
    detectOutput = cv::Size(objDetectRoiRect.width, objDetectRoiRect.height);

    morphKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 20));
    skinLowerBound = cv::Scalar(0, 48, 80);  // 살색의 하한값
    skinUpperBound = cv::Scalar(20, 255, 255);  // 살색의 상한값



    //휠제어 변수들
    wheelAngle = 0;
    wheelCenter = cv::Point2f(120, 120);
    wheelLeftPoint = cv::Point2f(0, 120);
    wheelRightPoint = cv::Point2f(240, 120);
    // 좌표를 2차원 행렬로 변환
    wheelPoints = cv::Mat(3, 1, CV_64FC2);
    wheelPoints.at<cv::Vec2d>(0, 0) = cv::Vec2d(wheelLeftPoint.x, wheelLeftPoint.y);
    wheelPoints.at<cv::Vec2d>(1, 0) = cv::Vec2d(wheelRightPoint.x, wheelRightPoint.y);
    wheelPoints.at<cv::Vec2d>(2, 0) = cv::Vec2d(wheelCenter.x, wheelCenter.y);
    wheelStartPos = cv::Point2f(200, 240);
    controlButtons = initButtons();

    isOutOfControlButton = true;
    control = 0;



    // 머리 자세 추론
    head_roi = cv::Rect(120, 0, 400, 400);
    head_image_size = cv::Size(int(head_roi.width / 2), int(head_roi.height / 2));
    backend_id = str2backend.at("opencv");
    target_id = str2target.at("cpu");
    cliped_xdiff = 0;

    figure_points_3D = get_figure_points_3D();
    camera_matrix = get_camera_matrix();
    distortion_coeff = get_distortion_coeff();
    vector_rotation = (cv::Mat_<double>(3, 1) << 0, 0, 0);
    vector_translation = (cv::Mat_<double>(3, 1) << 0, 0, 0);

    model = YuNet(
        "c:/yunet.onnx",
        head_image_size,
        0.6,
        0.3,
        3000,
        backend_id,
        target_id
    );

    //손 변수
    is_handle = false;
    hand_left_x = -60;
    hand_left_y = 20;
    hand_right_x = -20;
    hand_right_y = 20;
}

void ASimulatorGameModeBase::ReadFrame()
{
	UE_LOG(LogTemp, Log, TEXT("ReadFrame is called"));

	if (!capture.isOpened())
	{
		return;
	}
	capture.read(image);

	Inference();

	imageTexture = MatToTexture2D(image);
}


UTexture2D* ASimulatorGameModeBase::MatToTexture2D(const cv::Mat InMat)
{
	//create new texture, set its values
	UTexture2D* Texture = UTexture2D::CreateTransient(InMat.cols, InMat.rows, PF_B8G8R8A8);

	if (InMat.type() == CV_8UC3)//example for pre-conversion of Mat
	{
		//if the Mat is in BGR space, convert it to BGRA. There is no three channel texture in UE (at least with eight bit)
		cv::cvtColor(InMat, bgraImage, cv::COLOR_BGR2BGRA);

		//Texture->SRGB = 0;//set to 0 if Mat is not in srgb (which is likely when coming from a webcam)
		//other settings of the texture can also be changed here
		//Texture->UpdateResource();

		//actually copy the data to the new texture
		FTexture2DMipMap& Mip = Texture->GetPlatformData()->Mips[0];
		void* Data = Mip.BulkData.Lock(LOCK_READ_WRITE);//lock the texture data
		FMemory::Memcpy(Data, bgraImage.data, bgraImage.total() * bgraImage.elemSize());//copy the data
		Mip.BulkData.Unlock();
		Texture->PostEditChange();
		Texture->UpdateResource();
		return Texture;
	}
	UE_LOG(LogTemp, Log, TEXT("CV_8UC3"));
	//if the texture hasnt the right pixel format, abort.
	Texture->PostEditChange();
	Texture->UpdateResource();
	return Texture;
}




void ASimulatorGameModeBase::Inference()
{

	auto time_start = std::chrono::steady_clock::now();
    cv::flip(image, image, 1);

    /*
    손 검출 파트
    */
    std::vector<cv::Rect> skinRegions = getSkinRegions(image, handRoiRect, skinLowerBound, skinUpperBound, morphKernel);
    //skinRegions로 trackedRects 구함(등록, 생명주기 확인, 제거)
    trackedRectsMap = getTrackedRects(trackedRectsMap, skinRegions, nextIndex);
    // 객체 탐지 및 결과 반환
    cv::Mat detectFrame = image(objDetectRoiRect).clone();
    std::vector<DetectResult> detectResults = getDetectResults(detectFrame, net_det, detectInput, detectOutput);


    /*
    머리 자세 추론 파트
    */
    cv::Mat head_area = image(head_roi);
    model.setInputSize(head_image_size);
    cv::Mat resized_head;
    cv::resize(head_area, resized_head, head_image_size);
    cv::Mat faces = model.infer(resized_head);

    cv::Mat image_points_2D = get_image_points_2D();
    if (visualize(head_area, faces, image_points_2D))
    {
        estimate_chin(image_points_2D);
        cv::circle(head_area, cv::Point(int(image_points_2D.at<double>(5, 0)), int(image_points_2D.at<double>(5, 1))), 2, cv::Scalar(255, 255, 255), 2);

    }

    if (cv::solvePnPRansac(
        figure_points_3D,
        image_points_2D,
        camera_matrix,
        distortion_coeff,
        vector_rotation,
        vector_translation
    ))
    {
        std::vector<cv::Point2d> pose_points = get_pose_points(image_points_2D, vector_rotation, vector_translation, camera_matrix, distortion_coeff);
        cv::line(head_area, pose_points[0], pose_points[1], cv::Scalar(255, 255, 255), 2);
        int x_diff = pose_points[0].x - pose_points[1].x;
        xdiff_vector.push_back(x_diff);
        cv::putText(head_area, cv::format("%d", x_diff), cv::Point(pose_points[0].x - 10, pose_points[0].y - 20), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0));
        if (xdiff_vector.size() == 15)
        {
            xdiff_vector.erase(xdiff_vector.begin());
            cliped_xdiff = clip_avg(xdiff_vector);
            cv::putText(head_area, cv::format("%.1f", cliped_xdiff), cv::Point(pose_points[0].x + 40, pose_points[0].y - 20), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 255), 2);
        }
    }






    // detectResults 시각화
    for (auto& detectResult : detectResults)
        drawDetectResult(image, detectResult, objDetectRoiRect);

    // trackedRects에 있는 것들이 detectResults로 손인지 확인
    checkIsHand(trackedRectsMap, detectResults, objDetectRoiRect);

    // trackedRect가 2개 이상시, 가장 큰 trackedRect 2개 골라서 시각화
    std::map<int, TrackedRect> resRects;
    //가장 큰 박스 2개 반환
    if (trackedRectsMap.size() >= 2)
        resRects = getResRects(trackedRectsMap);
    else
        resRects = trackedRectsMap;

    // trackedRects 시각화
    for (auto& trackedRect : resRects)
        drawTrackedRect(image, trackedRect);

    cv::rectangle(image, handRoiRect, cv::Scalar(255, 0, 0), 1);  // 사각형 그리기
    cv::rectangle(image, objDetectRoiRect, cv::Scalar(0, 255, 0), 1);  // 사각형 그리기


    cv::Mat rotatedImage = wheelImage.clone();
    cv::Mat rotationMat = cv::getRotationMatrix2D(wheelCenter, wheelAngle, 1.0);

    cv::line(rotatedImage, wheelLeftPoint, wheelRightPoint, cv::Scalar(0, 0, 255), 5);
    cv::warpAffine(rotatedImage, rotatedImage, rotationMat, wheelImage.size());


    // 좌표 회전 적용
    cv::Mat rotatedPoints = drawAndGetRotatedPoints(image, wheelStartPos, wheelPoints, rotationMat);

    // 손 2개 있으면 손잡이 확인하고, 회전반영하기
    if (resRects.size() == 2)
    {
        bool wheelLeftChecked = false;
        bool wheelRightChecked = false;

        setHandPos(resRects);
        wheelCheck(image, wheelLeftChecked, wheelRightChecked, wheelStartPos, rotatedPoints, resRects);
        if (wheelLeftChecked == true && wheelRightChecked == true)
        {
            is_handle = true;
            wheelAngle = getWheelAngle(resRects);
        }
        else
            is_handle = false;
    }
    else
        is_handle = false;

    buttonEvent(image, resRects, controlButtons, isOutOfControlButton, startTimeControlButton);

    //버튼 그리기
    drawButtons(image, controlButtons);








    image = overlayTransparent(image, rotatedImage, 200, 240);


	auto time_end = std::chrono::steady_clock::now();
	auto time_diff = std::chrono::duration_cast<std::chrono::milliseconds>(time_end - time_start).count();
	std::string time_spent = "Time spent: " + std::to_string(time_diff) + "ms";
	cv::putText(image, time_spent, cv::Point(0, 50), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(125, 125, 125), 2);
}




// Function to calculate Intersection over Union (IOU) between two rectangles
float ASimulatorGameModeBase::calculateIOU(const cv::Rect& rect1, const cv::Rect& rect2) {
    int x1 = std::max(rect1.x, rect2.x);
    int y1 = std::max(rect1.y, rect2.y);
    int x2 = std::min(rect1.x + rect1.width, rect2.x + rect2.width);
    int y2 = std::min(rect1.y + rect1.height, rect2.y + rect2.height);

    int intersectionArea = std::max(0, x2 - x1) * std::max(0, y2 - y1);
    int unionArea = rect1.width * rect1.height + rect2.width * rect2.height - intersectionArea;

    return static_cast<float>(intersectionArea) / unionArea;
}



std::map<int, ASimulatorGameModeBase::TrackedRect> ASimulatorGameModeBase::getTrackedRects(std::map<int, TrackedRect> trackedRects, std::vector<cv::Rect> skinRegions, uint32_t& nextId)
{
    std::vector<int> foundIndex;

    for (const cv::Rect& skinRegion : skinRegions)
    {
        bool foundMatch = false;
        for (auto& trackedRect : trackedRects)
        {
            // Calculate IOU between detected rectangle and tracked rectangle
            float iou = calculateIOU(skinRegion, trackedRect.second.rect);

            // If IOU is above a threshold, update the tracked rectangle
            if (iou > 0.4) {
                trackedRect.second.rect = skinRegion;
                foundMatch = true;
                foundIndex.push_back(trackedRect.first);
                break;
            }
        }

        // If no match found, add new tracked rectangle
        if (!foundMatch) {
            trackedRects[nextId].rect = skinRegion;
            trackedRects[nextId].lifespan = 5;
            trackedRects[nextId].isHandDetected = false;
            nextId++;
        }
    }

    //if trackedRect not found, lifespan -1
    for (auto& trackedRect : trackedRects)
    {
        bool foundMatch = false;
        for (auto& id : foundIndex)
        {
            if (id == trackedRect.first)
            {
                foundMatch = true;
                trackedRect.second.lifespan = 5; //찾으면 5로 다시 설정
                break;
            }
        }
        if (foundMatch == false)
        {
            trackedRect.second.lifespan -= 1;
        }
    }

    // lifespan == 0인 trackedRect 삭제
    for (auto& trackedRect : trackedRects)
    {
        if (trackedRect.second.lifespan == 0)
        {
            trackedRects.erase(trackedRect.first);
        }
    }
    return trackedRects;
}


std::map<int, ASimulatorGameModeBase::TrackedRect> ASimulatorGameModeBase::getResRects(std::map<int, TrackedRect> trackedRects)
{
    std::map<int, TrackedRect> resRects;

    std::vector<std::pair<int, TrackedRect>> sortedTracedRects(trackedRects.begin(), trackedRects.end());
    std::sort(
        sortedTracedRects.begin(), sortedTracedRects.end(), [](const std::pair<int, TrackedRect>& a, const std::pair<int, TrackedRect>& b) {
            return a.second.rect.width * a.second.rect.height > b.second.rect.width * b.second.rect.height;
        });

    int key1 = sortedTracedRects[0].first;
    int key2 = sortedTracedRects[1].first;
    resRects[key1] = trackedRects[key1];
    resRects[key2] = trackedRects[key2];

    return resRects;
}


void ASimulatorGameModeBase::drawTrackedRect(cv::Mat frame, std::pair<int, TrackedRect> trackedRect)
{
    cv::putText(
        frame, cv::String(std::to_string(trackedRect.first)),
        cv::Point(int(trackedRect.second.rect.x), int(trackedRect.second.rect.y - 20)),
        cv::FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0)
    );

    cv::putText(
        frame, cv::String(std::to_string(trackedRect.second.lifespan)),
        cv::Point(int(trackedRect.second.rect.x + 100), int(trackedRect.second.rect.y - 20)),
        cv::FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255)
    );

    if (trackedRect.second.isHandDetected == true)
    {
        cv::rectangle(frame, trackedRect.second.rect, cv::Scalar(255, 0, 0), 6);  // 사각형 그리기
    }
    cv::rectangle(frame, trackedRect.second.rect, cv::Scalar(0, 255, 0), 2);  // 사각형 그리기
}



cv::Mat ASimulatorGameModeBase::overlayTransparent(const cv::Mat& background_img, const cv::Mat& img_to_overlay_t, int x, int y)
{
    cv::Mat bg_img = background_img.clone();
    cv::Mat overlay_img = img_to_overlay_t.clone();

    std::vector<cv::Mat> channels;
    cv::split(overlay_img, channels);
    cv::Mat overlay_color;
    cv::merge(std::vector<cv::Mat>{channels[0], channels[1], channels[2]}, overlay_color);

    cv::Mat mask;
    cv::medianBlur(channels[3], mask, 5);

    cv::Rect roi(x, y, overlay_color.cols, overlay_color.rows);
    cv::Mat roi_bg = bg_img(roi).clone();
    cv::Mat bitnot_mask;
    cv::bitwise_not(mask, bitnot_mask);
    cv::bitwise_and(roi_bg, roi_bg, roi_bg, bitnot_mask);
    cv::bitwise_and(overlay_color, overlay_color, overlay_color, mask);
    cv::add(roi_bg, overlay_color, roi_bg);
    roi_bg.copyTo(bg_img(roi));

    return bg_img;
}


std::vector<cv::Rect> ASimulatorGameModeBase::getSkinRegions(cv::Mat frame, cv::Rect roi, cv::Scalar lowerBound, cv::Scalar upperBound, cv::Mat kernel)
{
    cv::Mat frameHSV;
    cv::Mat croppedFrame = frame(roi);
    cv::cvtColor(croppedFrame, frameHSV, cv::COLOR_BGR2HSV);  // BGR을 HSV로 변환

    cv::Mat skinMask;
    cv::inRange(frameHSV, lowerBound, upperBound, skinMask);  // 살색 범위에 속하는 픽셀을 마스크로 생성

    cv::Mat skin;
    cv::bitwise_and(croppedFrame, croppedFrame, skin, skinMask);  // 원본 이미지와 마스크를 이용하여 살색 영역 추출

    cv::Mat skinGray;
    cv::cvtColor(skin, skinGray, cv::COLOR_BGR2GRAY);  // 추출한 살색 영역을 그레이스케일로 변환

    cv::Mat closing;
    cv::morphologyEx(skinGray, closing, cv::MORPH_CLOSE, kernel);


    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(closing, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);  // 윤곽선 검출

    std::vector<cv::Rect> skinRegions;
    for (const std::vector<cv::Point>& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);  // 윤곽선을 감싸는 사각형 생성
        auto area = rect.width * rect.height;
        if ((area >= 70 * 70) && (area <= 300 * 200))
        {
            rect.x += roi.x;
            rect.y += roi.y;
            skinRegions.push_back(rect);  // 사각형을 벡터에 추가
        }
    }

    return skinRegions;
}


float ASimulatorGameModeBase::sigmoid(float x) {
    return 1 / (1 + exp(-x));
}



ASimulatorGameModeBase::DetectResult ASimulatorGameModeBase::getDetectResult(cv::Mat& frame, cv::Mat regressor, cv::Mat classificator,
    int stride, int anchor_count, int column, int row, int anchor, int offset,
    cv::Size detectInputSize, cv::Size detectOutputSize) {

    DetectResult res{ 0.0f, 0, 0, 0, 0 };

    int index = (int(row * 128 / stride) + column) * anchor_count + anchor + offset;
    float origin_score = regressor.at<float>(0, index, 0);
    float score = sigmoid(origin_score);
    if (score < 0.5) return res;

    float x = classificator.at<float>(0, index, 0);
    float y = classificator.at<float>(0, index, 1);
    float w = classificator.at<float>(0, index, 2);
    float h = classificator.at<float>(0, index, 3);


    x += (column + 0.5) * stride - w / 2;
    y += (row + 0.5) * stride - h / 2;

    float width_ratio = static_cast<float>(detectOutputSize.width) / static_cast<float>(detectInputSize.width);
    float height_radio = static_cast<float>(detectOutputSize.height) / static_cast<float>(detectInputSize.height);
    res.score = score;
    res.x = int(x * width_ratio);
    res.y = int(y * height_radio);
    res.w = int(w * width_ratio);
    res.h = int(h * height_radio);
    return res;
}


std::vector<ASimulatorGameModeBase::DetectResult> ASimulatorGameModeBase::getDetectResults(cv::Mat frame, cv::dnn::Net net, cv::Size detectInputSize, cv::Size detectOutputSize)
{
    std::vector<DetectResult> beforeNMSResults;
    std::vector<DetectResult> afterNMSResults;
    std::vector<float> scores;
    std::vector<int> indices;
    std::vector<cv::Rect> boundingBoxes;

    cv::Mat inputImg;
    cv::resize(frame, inputImg, detectInputSize);
    cv::cvtColor(inputImg, inputImg, cv::COLOR_BGR2RGB);

    cv::Mat tensor;
    inputImg.convertTo(tensor, CV_32F, 1 / 127.5, -1.0);
    cv::Mat blob = cv::dnn::blobFromImage(tensor, 1.0, tensor.size(), 0, false, false, CV_32F);
    std::vector<cv::String> outNames(2);
    outNames[0] = "regressors";
    outNames[1] = "classificators";

    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, outNames);

    cv::Mat classificator = outputs[0];
    cv::Mat regressor = outputs[1];


    for (int y = 0; y < 16; ++y) {
        for (int x = 0; x < 16; ++x) {
            for (int a = 0; a < 2; ++a) {
                DetectResult res = getDetectResult(frame, regressor, classificator, 8, 2, x, y, a, 0, detectInputSize, detectOutputSize);
                if (res.score != 0)
                {
                    beforeNMSResults.push_back(res);
                    boundingBoxes.push_back(cv::Rect(res.x, res.y, res.w, res.h));
                    scores.push_back(res.score);
                }
            }
        }
    }

    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            for (int a = 0; a < 6; ++a) {
                DetectResult res = getDetectResult(frame, regressor, classificator, 16, 6, x, y, a, 512, detectInputSize, detectOutputSize);
                if (res.score != 0)
                {
                    beforeNMSResults.push_back(res);
                    boundingBoxes.push_back(cv::Rect(res.x, res.y, res.w, res.h));
                    scores.push_back(res.score);
                }
            }
        }
    }

    cv::dnn::NMSBoxes(boundingBoxes, scores, 0.5, 0.3, indices);

    for (int i = 0; i < indices.size(); i++) {
        int idx = indices[i];
        afterNMSResults.push_back(beforeNMSResults[idx]);
    }

    return afterNMSResults;
}

void ASimulatorGameModeBase::drawDetectResult(cv::Mat frame, ASimulatorGameModeBase::DetectResult res, cv::Rect objDetectRoi)
{
    cv::putText(
        frame, cv::String(std::to_string(res.score)),
        cv::Point(int(res.x + objDetectRoi.x), int(res.y + objDetectRoi.y - 20)),
        cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255)
    );
    cv::rectangle(frame, cv::Rect(res.x + objDetectRoi.x, res.y + objDetectRoi.y, res.w, res.h), cv::Scalar(0, 0, 255), 1);  // 사각형 그리기
}


// 가장 큰 trackedRects 2개(resRects)와 detectResults로 손인지 확인
void ASimulatorGameModeBase::checkIsHand(std::map<int, ASimulatorGameModeBase::TrackedRect>& trackedRects, std::vector<DetectResult>& detectResults, cv::Rect objDetectRoi)
{
    for (auto& trackedRect : trackedRects)
    {
        for (auto& detectResult : detectResults)
        {
            cv::Rect detectResultRect{ detectResult.x + objDetectRoi.x, detectResult.y + objDetectRoi.y, detectResult.w, detectResult.h };
            float iou = calculateIOU(detectResultRect, trackedRect.second.rect);

            if (iou > 0.4) {
                trackedRect.second.isHandDetected = true;
            }
        }
    }
}





float ASimulatorGameModeBase::getWheelAngle(std::map<int, TrackedRect> resRects)
{
    TrackedRect leftRect = resRects.begin()->second;
    TrackedRect rightRect = resRects.begin()->second;

    for (auto& trackedRect : resRects)
    {
        if (trackedRect.second.rect.x < leftRect.rect.x)
            leftRect = trackedRect.second;

        if (trackedRect.second.rect.x > rightRect.rect.x)
            rightRect = trackedRect.second;
    }

    // 두 개의 직사각형 중심점 계산
    cv::Point2f center1(leftRect.rect.x + leftRect.rect.width / 2.0f, leftRect.rect.y + leftRect.rect.height / 2.0f);
    cv::Point2f center2(rightRect.rect.x + rightRect.rect.width / 2.0f, rightRect.rect.y + rightRect.rect.height / 2.0f);
    // 두 중심점을 연결하는 선의 기울기 계산
    float dx = center2.x - center1.x;
    float dy = center2.y - center1.y;
    float angle = (-1) * atan2(dy, dx) * 180.0f / CV_PI;
    return angle;
}


void ASimulatorGameModeBase::setHandPos(std::map<int, TrackedRect> resRects)
{
    TrackedRect leftRect = resRects.begin()->second;
    TrackedRect rightRect = resRects.begin()->second;

    for (auto& trackedRect : resRects)
    {
        if (trackedRect.second.rect.x < leftRect.rect.x)
            leftRect = trackedRect.second;

        if (trackedRect.second.rect.x > rightRect.rect.x)
            rightRect = trackedRect.second;
    }

    // 두 개의 직사각형 중심점 계산
    cv::Point2f center1(leftRect.rect.x + leftRect.rect.width / 2.0f, leftRect.rect.y + leftRect.rect.height / 2.0f);
    cv::Point2f center2(rightRect.rect.x + rightRect.rect.width / 2.0f, rightRect.rect.y + rightRect.rect.height / 2.0f);
    hand_left_x = center1.x;
    hand_left_y = center1.y;
    hand_right_x = center2.x;
    hand_right_y = center2.y;
}


void ASimulatorGameModeBase::wheelCheck(cv::Mat frame, bool& wheelLeftChecked, bool& wheelRightChecked, cv::Point2f wheelStartPoint, cv::Mat rotatedPoints, std::map<int, TrackedRect> resRects)
{
    // 회전된 좌표 추출
    cv::Point2f rotatedLeftPoint(rotatedPoints.at<cv::Vec2d>(0, 0)[0], rotatedPoints.at<cv::Vec2d>(0, 0)[1]);
    cv::Point2f rotatedRightPoint(rotatedPoints.at<cv::Vec2d>(1, 0)[0], rotatedPoints.at<cv::Vec2d>(1, 0)[1]);
    cv::Point2f rotatedCenter(rotatedPoints.at<cv::Vec2d>(2, 0)[0], rotatedPoints.at<cv::Vec2d>(2, 0)[1]);

    for (auto& trackedRect : resRects)
    {
        if (trackedRect.second.isHandDetected == false)
            return;

        float circleRadius = 30;
        cv::Mat leftCircleImage = cv::Mat::zeros(640, 480, CV_8UC1);
        cv::circle(leftCircleImage, wheelStartPoint + rotatedLeftPoint, circleRadius, 255, -1);
        cv::Mat leftBoxImage = cv::Mat::zeros(640, 480, CV_8UC1);
        cv::rectangle(leftBoxImage, trackedRect.second.rect, 255, -1);

        // 원과 박스의 교차 영역 계산
        cv::Mat intersectionImageLeft = leftCircleImage & leftBoxImage;

        // 교차 영역의 픽셀 개수 계산
        int intersectionPixelsLeft = cv::countNonZero(intersectionImageLeft);


        float leftOverlapRatio = static_cast<float>(intersectionPixelsLeft) / (circleRadius * circleRadius * CV_PI);

        if (leftOverlapRatio > 0.8)
        {
            cv::circle(frame, wheelStartPoint + rotatedLeftPoint, 30, cv::Scalar(255, 0, 0), -1);
            wheelLeftChecked = true;
        }

        cv::Mat rightCircleImage = cv::Mat::zeros(640, 480, CV_8UC1);
        cv::circle(rightCircleImage, wheelStartPoint + rotatedRightPoint, circleRadius, 255, -1);
        cv::Mat rightBoxImage = cv::Mat::zeros(640, 480, CV_8UC1);
        cv::rectangle(rightBoxImage, trackedRect.second.rect, 255, -1);

        cv::Mat intersectionImageRight = rightCircleImage & rightBoxImage;
        int intersectionPixelsRight = cv::countNonZero(intersectionImageRight);
        float rightOverlapRatio = static_cast<float>(intersectionPixelsRight) / (circleRadius * circleRadius * CV_PI);

        if (rightOverlapRatio > 0.8)
        {
            cv::circle(frame, wheelStartPoint + rotatedRightPoint, 30, cv::Scalar(0, 255, 0), -1);
            wheelRightChecked = true;
        }
    }
}


cv::Mat ASimulatorGameModeBase::drawAndGetRotatedPoints(cv::Mat frame, cv::Point2f wheelStartPoint, cv::Mat points, cv::Mat rotationMat)
{
    // 좌표 회전 적용
    cv::Mat rotatedPoints = points.clone();
    cv::transform(rotatedPoints, rotatedPoints, rotationMat);

    // 회전된 좌표 추출
    cv::Point2f rotatedLeftPoint(rotatedPoints.at<cv::Vec2d>(0, 0)[0], rotatedPoints.at<cv::Vec2d>(0, 0)[1]);
    cv::Point2f rotatedRightPoint(rotatedPoints.at<cv::Vec2d>(1, 0)[0], rotatedPoints.at<cv::Vec2d>(1, 0)[1]);
    cv::Point2f rotatedCenter(rotatedPoints.at<cv::Vec2d>(2, 0)[0], rotatedPoints.at<cv::Vec2d>(2, 0)[1]);


    cv::circle(frame, wheelStartPoint + rotatedLeftPoint, 30, cv::Scalar(255, 0, 0), 5);
    cv::circle(frame, wheelStartPoint + rotatedRightPoint, 30, cv::Scalar(0, 255, 0), 5);
    cv::circle(frame, wheelStartPoint + rotatedCenter, 40, cv::Scalar(255, 255, 255), 5);
    return rotatedPoints;
}


std::vector<ASimulatorGameModeBase::Button> ASimulatorGameModeBase::initButtons()
{
    std::vector<Button> buttons;

    Button buttonForward;

    buttonForward.action = Button::forward;
    buttonForward.rect = cv::Rect(530, 270, 100, 50);
    buttonForward.isPressed = false;
    buttonForward.color = cv::Scalar(255, 0, 0);
    buttons.push_back(buttonForward);

    Button buttonStop;

    buttonStop.action = Button::stop;
    buttonStop.rect = cv::Rect(530, 340, 100, 50);
    buttonStop.isPressed = true;
    buttonStop.color = cv::Scalar(0, 0, 255);
    buttons.push_back(buttonStop);


    Button buttonBackward;

    buttonBackward.action = Button::backward;
    buttonBackward.rect = cv::Rect(530, 410, 100, 50);
    buttonBackward.isPressed = false;
    buttonBackward.color = cv::Scalar(0, 255, 0);
    buttons.push_back(buttonBackward);

    return buttons;
}

void ASimulatorGameModeBase::drawButtons(cv::Mat frame, std::vector<Button> buttons)
{
    for (auto& button : buttons)
    {
        if (button.isPressed == true)
        {
            cv::rectangle(frame, button.rect, button.color, -1);
        }
        else
        {
            cv::rectangle(frame, button.rect, button.color, 2);
        }


        std::string tmpString;
        switch (button.action) {
        case Button::Actions::forward:
            tmpString = "forward";
            break;
        case Button::Actions::stop:
            tmpString = "stop";
            break;
        case Button::Actions::backward:
            tmpString = "backward";
            break;
        }

        cv::putText(
            frame, cv::String(tmpString),
            cv::Point(int(button.rect.x), int(button.rect.y + 30)),
            cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 0, 0), 2
        );

    }
}


bool ASimulatorGameModeBase::isPointInsideRect(const cv::Point2d& point, const cv::Rect& rect) {
    return (point.x >= rect.x && point.x < rect.x + rect.width &&
        point.y >= rect.y && point.y < rect.y + rect.height);
}


bool ASimulatorGameModeBase::checkBoolVector(const std::vector<bool>& vec) {
    for (bool value : vec) {
        if (value) {
            return true;
        }
    }
    return false;
}


void ASimulatorGameModeBase::buttonEvent(cv::Mat frame, std::map<int, ASimulatorGameModeBase::TrackedRect> resRects, std::vector<ASimulatorGameModeBase::Button>& buttons, bool& isOutOfButton, std::chrono::steady_clock::time_point& startTime)
{
    //오른쪽에 있는 TrackedRect 구하기
    ASimulatorGameModeBase::TrackedRect rightRect;
    if (resRects.size() > 1)
    {
        ASimulatorGameModeBase::TrackedRect leftRect = resRects.begin()->second;
        rightRect = resRects.begin()->second;

        for (auto& trackedRect : resRects)
        {
            if (trackedRect.second.rect.x < leftRect.rect.x)
                leftRect = trackedRect.second;
            if (trackedRect.second.rect.x > rightRect.rect.x)
                rightRect = trackedRect.second;
        }
    }
    else
    {
        rightRect = resRects.begin()->second;
    }

    cv::Point2f center(rightRect.rect.x + rightRect.rect.width / 2.0f, rightRect.rect.y + rightRect.rect.height / 2.0f);
    cv::circle(frame, center, 5, cv::Scalar(0, 0, 0), -1);

    //센터점이 버튼에들어갔는지 확인
    std::vector<bool> isPointInButtonVector;
    for (auto& button : buttons)
    {
        if (isPointInsideRect(center, button.rect))
        {
            isPointInButtonVector.push_back(true);
        }
        else
        {
            isPointInButtonVector.push_back(false);
        }
    }

    bool isPointInButton = checkBoolVector(isPointInButtonVector);
    // isOutOfButton이 true이지만 isPointInButton가 true된 경우 포인트가 밖에서 버튼으로 들어옴
    // isOutOfButton = false,하고 시작시간 설정
    if (isOutOfButton == true && isPointInButton == true)
    {
        isOutOfButton = false;
        startTime = std::chrono::steady_clock::now();
    }
    // isOutOfButton == false && isPointInButton == true인 경우 포인트가 계속 안에 존재함
    // 현재 시간을 보고 경과했는지 확인
    else if (isOutOfButton == false && isPointInButton == true)
    {
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime);
        cv::putText(
            frame, cv::String(std::to_string(elapsedTime.count())),
            cv::Point(500, 40),
            cv::FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0)
        );

        // 1초 동안 겹칠 때 동작 수행
        if (elapsedTime.count() >= 1000)
        {
            ASimulatorGameModeBase::Button::Actions action;
            for (auto& button : buttons)
            {
                if (isPointInsideRect(center, button.rect))
                {
                    button.isPressed = true;
                    action = button.action;

                    switch (button.action) {
                    case Button::Actions::forward:
                        control = 1;
                        break;
                    case Button::Actions::stop:
                        control = 0;
                        break;
                    case Button::Actions::backward:
                        control = -1;
                        break;
                    }

                }
                else
                {
                    button.isPressed = false;
                }
            }
        }
    }
    // 그 외 경우 포인트는 버튼밖이므로 isOutOfButton = true, 버튼 isPressed = false
    else
    {
        isOutOfButton = true;
    }
}










cv::Mat ASimulatorGameModeBase::get_image_points_2D()
{
    cv::Mat image_points_2D = cv::Mat::zeros(6, 2, CV_64F);
    image_points_2D.at<double>(0, 0) = 0;  // right eye
    image_points_2D.at<double>(0, 1) = 0;
    image_points_2D.at<double>(1, 0) = 0;  // left eye
    image_points_2D.at<double>(1, 1) = 0;
    image_points_2D.at<double>(2, 0) = 0;  // nose tip
    image_points_2D.at<double>(2, 1) = 0;
    image_points_2D.at<double>(3, 0) = 0;  // right mouth corner
    image_points_2D.at<double>(3, 1) = 0;
    image_points_2D.at<double>(4, 0) = 0;  // left mouth corner
    image_points_2D.at<double>(4, 1) = 0;
    image_points_2D.at<double>(5, 0) = 0;  // chin
    image_points_2D.at<double>(5, 1) = 0;
    return image_points_2D;
}

cv::Mat ASimulatorGameModeBase::get_figure_points_3D()
{
    cv::Mat figure_points = cv::Mat::zeros(6, 3, CV_64F);
    figure_points.at<double>(0, 0) = 180.0;     // Right eye right corner
    figure_points.at<double>(0, 1) = 170.0;
    figure_points.at<double>(0, 2) = -135.0;
    figure_points.at<double>(1, 0) = -180.0;    // Left eye left corner
    figure_points.at<double>(1, 1) = 170.0;
    figure_points.at<double>(1, 2) = -135.0;
    figure_points.at<double>(2, 0) = 0.0;       // Nose tip
    figure_points.at<double>(2, 1) = 0.0;
    figure_points.at<double>(2, 2) = 0.0;
    figure_points.at<double>(3, 0) = 150.0;     // Right mouth corner
    figure_points.at<double>(3, 1) = -150.0;
    figure_points.at<double>(3, 2) = -125.0;
    figure_points.at<double>(4, 0) = -150.0;    // Left mouth corner
    figure_points.at<double>(4, 1) = -150.0;
    figure_points.at<double>(4, 2) = -125.0;
    figure_points.at<double>(5, 0) = 0.0;       // Chin
    figure_points.at<double>(5, 1) = -330.0;
    figure_points.at<double>(5, 2) = -65.0;
    return figure_points;
}

cv::Mat ASimulatorGameModeBase::get_camera_matrix()
{
    cv::Mat matrix_camera = cv::Mat::eye(3, 3, CV_64F);
    matrix_camera.at<double>(0, 0) = 1013.80634;
    matrix_camera.at<double>(0, 2) = 632.511658;
    matrix_camera.at<double>(1, 1) = 1020.62616;
    matrix_camera.at<double>(1, 2) = 259.604004;
    return matrix_camera;
}

cv::Mat ASimulatorGameModeBase::get_distortion_coeff()
{
    cv::Mat distortion_coeffs = cv::Mat::zeros(1, 5, CV_64F);
    distortion_coeffs.at<double>(0, 0) = 0.05955474;
    distortion_coeffs.at<double>(0, 1) = -0.6827085;
    distortion_coeffs.at<double>(0, 2) = -0.03125953;
    distortion_coeffs.at<double>(0, 3) = -0.00254411;
    distortion_coeffs.at<double>(0, 4) = 1.316122;
    return distortion_coeffs;
}

void ASimulatorGameModeBase::estimate_chin(cv::Mat& image_points_2D)
{
    cv::Point eye_midpoint((image_points_2D.at<double>(0, 0) + image_points_2D.at<double>(1, 0)) / 2, (image_points_2D.at<double>(0, 1) + image_points_2D.at<double>(1, 1)) / 2);
    cv::Point mouth_midpoint((image_points_2D.at<double>(3, 0) + image_points_2D.at<double>(4, 0)) / 2, (image_points_2D.at<double>(3, 1) + image_points_2D.at<double>(4, 1)) / 2);

    double slope;
    double intercept;

    double chin_x = 0;
    double chin_y = mouth_midpoint.y - (eye_midpoint.y - mouth_midpoint.y) / 2;

    if ((eye_midpoint.x - mouth_midpoint.x) == 0)
    {
        chin_x = mouth_midpoint.x;

    }
    else
    {
        // 두 중간 점을 지나는 직선 계산
        slope = (eye_midpoint.y - mouth_midpoint.y) / (eye_midpoint.x - mouth_midpoint.x);
        intercept = eye_midpoint.y - slope * eye_midpoint.x;

        if (slope == std::numeric_limits<double>::infinity() || intercept == std::numeric_limits<double>::infinity()) {
            chin_x = mouth_midpoint.x;
        }
        else {
            chin_x = (chin_y - intercept) / slope;
        }
    }
    image_points_2D.at<double>(5, 0) = int(chin_x);
    image_points_2D.at<double>(5, 1) = int(chin_y);
}


bool ASimulatorGameModeBase::visualize(cv::Mat& frame, const cv::Mat& faces, cv::Mat& image_points_2D)
{
    bool res = false;
    static cv::Scalar box_color{ 0, 255, 0 };
    static cv::Scalar text_color{ 0, 255, 0 };

    std::vector<cv::Scalar> landmark_color = {
        cv::Scalar(255, 0, 0),  // right eye
        cv::Scalar(0, 0, 255),  // left eye
        cv::Scalar(0, 255, 0),  // nose tip
        cv::Scalar(255, 0, 255),// right mouth corner
        cv::Scalar(0, 255, 255) // left mouth corner
    };


    for (int i = 0; i < faces.rows; ++i)
    {
        // Draw bounding boxes
        int x1 = static_cast<int>(faces.at<float>(i, 0)) * 2;
        int y1 = static_cast<int>(faces.at<float>(i, 1)) * 2;
        int w = static_cast<int>(faces.at<float>(i, 2)) * 2;
        int h = static_cast<int>(faces.at<float>(i, 3)) * 2;
        cv::rectangle(frame, cv::Rect(x1, y1, w, h), box_color, 2);

        // Confidence as text
        float conf = faces.at<float>(i, 14);
        cv::putText(frame, cv::format("%.4f", conf), cv::Point(x1, y1 + 12), cv::FONT_HERSHEY_DUPLEX, 0.5, text_color);
        // Draw landmarks
        for (int j = 0; j < landmark_color.size(); ++j)
        {
            res = true;
            int x = static_cast<int>(faces.at<float>(i, 2 * j + 4)) * 2, y = static_cast<int>(faces.at<float>(i, 2 * j + 5)) * 2;
            cv::circle(frame, cv::Point(x, y), 2, landmark_color[j], 2);
            //std::cout << x << "," << y << std::endl;
            image_points_2D.at<double>(j, 0) = static_cast<double>(x);
            image_points_2D.at<double>(j, 1) = static_cast<double>(y);
        }
    }

    return res;
}

std::vector<cv::Point2d> ASimulatorGameModeBase::get_pose_points(cv::Mat& image_points_2D, cv::Mat& vector_rot, cv::Mat& vector_tran, cv::Mat& camera_mat, cv::Mat& distort_coeff)
{
    std::vector<cv::Point2d> pose_points;
    cv::Mat nose_end_point3D = (cv::Mat_<double>(1, 3) << 0, 0, 1000.0);
    cv::Mat nose_end_point2D;
    cv::projectPoints(nose_end_point3D, vector_rot, vector_tran, camera_mat, distort_coeff, nose_end_point2D);

    cv::Point2d point1(image_points_2D.at<double>(2, 0), image_points_2D.at<double>(2, 1));
    cv::Point2d point2(nose_end_point2D.at<double>(0, 0), nose_end_point2D.at<double>(0, 1));

    pose_points.push_back(point1);
    pose_points.push_back(point2);
    return pose_points;
}

float ASimulatorGameModeBase::clip_avg(std::vector<int> xdiff_vec)
{
    float xdiff_vector_sum = std::accumulate(xdiff_vec.begin(), xdiff_vec.end(), 0.0);
    float avg = xdiff_vector_sum / xdiff_vec.size();


    if (avg >= 15) {
        avg += -15;
    }
    else if (avg > -15 && avg < 15) {
        avg = 0;
    }
    else {
        avg += 15;
    }

    return avg / 2;
}