// Fill out your copyright notice in the Description page of Project Settings.


#include "SimulatorGameModeBase.h"
#include "Blueprint/UserWidget.h"
#include "MainWidget.h"
#include "Components/Image.h"
#include <chrono>



void ASimulatorGameModeBase::BeginPlay()
{
	Super::BeginPlay();

    blaze = Blaze();

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

    nextIndex = 0; //tracking id

    wheelImage = cv::imread("C:/wheel.png", cv::IMREAD_UNCHANGED);
    cv::resize(wheelImage, wheelImage, cv::Size(240, 240));


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
		//Texture->PostEditChange();
		Texture->UpdateResource();
		return Texture;
	}
	UE_LOG(LogTemp, Log, TEXT("CV_8UC3"));
	//if the texture hasnt the right pixel format, abort.
	//Texture->PostEditChange();
	Texture->UpdateResource();
	return Texture;
}




void ASimulatorGameModeBase::Inference()
{



    auto time_start = std::chrono::steady_clock::now();
    cv::flip(image, image, 1);

    /*
    get filtered detections
    */
    blaze.ResizeAndPad(image, img256, img128, scale, pad);
    //UE_LOG(LogTemp, Log, TEXT("scale value: %f, pad value: (%f, %f)"), scale, pad[0], pad[1]);
    std::vector<Blaze::PalmDetection> normDets = blaze.PredictPalmDetections(img128);
    std::vector<Blaze::PalmDetection> denormDets = blaze.DenormalizePalmDetections(normDets, webcamWidth, webcamHeight, pad);
    std::vector<Blaze::PalmDetection> filteredDets = blaze.FilteringDets(denormDets, webcamWidth, webcamHeight);


    trackedRectsMap = getTrackedRects(trackedRectsMap, filteredDets, nextIndex);


    std::map<int, TrackedRect> resRects;

    if (trackedRectsMap.size() >= 2)
        resRects = getResRects(trackedRectsMap);
    else
        resRects = trackedRectsMap;

    for (auto& trackedRect : resRects)
        drawTrackedRect(image, trackedRect);



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
        else if (wheelLeftChecked == true && wheelRightChecked == false)
        {
            is_handle = true;
            wheelAngle = getWheelAngleWithCenter(resRects);

        }
        else if (wheelLeftChecked == false && wheelRightChecked == true)
        {
            is_handle = true;

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










std::map<int, ASimulatorGameModeBase::TrackedRect> ASimulatorGameModeBase::getTrackedRects(std::map<int, TrackedRect> trackedRects, std::vector<Blaze::PalmDetection> filteredDets, uint32_t& nextId)
{
    std::vector<int> foundIndex;

    for (const Blaze::PalmDetection& filteredDet : filteredDets)
    {
        bool foundMatch = false;
        for (auto& trackedRect : trackedRects)
        {
            // Calculate IOU between detected rectangle and tracked rectangle
            float iou = calculateIOU(filteredDet.rect, trackedRect.second.rect);

            // If IOU is above a threshold, update the tracked rectangle
            if (iou > 0.3) {
                trackedRect.second.rect = filteredDet.rect;
                foundMatch = true;
                foundIndex.push_back(trackedRect.first);
                break;
            }
        }

        // If no match found, add new tracked rectangle
        if (!foundMatch) {
            trackedRects[nextId].rect = filteredDet.rect;
            trackedRects[nextId].lifespan = 10;
            trackedRects[nextId].isHandDetected = true;
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

float ASimulatorGameModeBase::getWheelAngleWithCenter(std::map<int, TrackedRect> resRects)
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
    cv::Point2f center2(wheelStartPos.x + 120, wheelStartPos.y + 120);
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

        if (leftOverlapRatio > 0.4)
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

        if (rightOverlapRatio > 0.4)
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
    cv::circle(frame, center, 10, cv::Scalar(0, 0, 0), -1);

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