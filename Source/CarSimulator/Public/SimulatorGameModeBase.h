// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Blaze.h"

#include "PreOpenCVHeaders.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "PostOpenCVHeaders.h"
#include <numeric>

#include "CoreMinimal.h"
#include "GameFramework/GameModeBase.h"
#include "SimulatorGameModeBase.generated.h"

/**
 * 
 */
UCLASS()
class CARSIMULATOR_API ASimulatorGameModeBase : public AGameModeBase
{
	GENERATED_BODY()
protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:

	struct Button {
		enum Actions { forward, stop, backward };
		Actions action;
		cv::Rect rect;
		cv::Scalar color;
		bool isPressed;
	};

	struct WheelInfo {
		float angle;
		cv::Point2d leftPoint;
		cv::Point2d rightPoint;
		float radius;
	};

	struct TrackedRect {
		cv::Rect rect;
		int lifespan;
		bool isHandDetected;
	};

	cv::VideoCapture capture;
	cv::Mat bgraImage;

	int32 width;
	int32 height;

	uint32_t nextIndex; //tracking id
	std::map<int, TrackedRect> trackedRectsMap; // map to store tracked rectangles

	cv::Mat wheelImage;

	cv::Mat image;


	//휠제어 변수들
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float wheelAngle;

	cv::Point2f wheelCenter;
	cv::Point2f wheelLeftPoint;
	cv::Point2f wheelRightPoint;
	// 좌표를 2차원 행렬로 변환
	cv::Mat wheelPoints;
	cv::Point2f wheelStartPos;
	std::vector<Button> controlButtons;

	bool isOutOfControlButton;
	std::chrono::steady_clock::time_point startTimeControlButton;

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	int control;

	// tracking
	float calculateIOU(const cv::Rect& rect1, const cv::Rect& rect2);
	std::map<int, TrackedRect> getTrackedRects(std::map<int, TrackedRect> trackedRects, std::vector<Blaze::PalmDetection> filteredDets, uint32_t& nextId);
	std::map<int, TrackedRect> getResRects(std::map<int, TrackedRect> trackedRects);



	void drawTrackedRect(cv::Mat frame, std::pair<int, TrackedRect> trackedRect);
	cv::Mat overlayTransparent(const cv::Mat& background_img, const cv::Mat& img_to_overlay_t, int x, int y);

	// control
	cv::Mat drawAndGetRotatedPoints(cv::Mat frame, cv::Point2f wheelStartPoint, cv::Mat Points, cv::Mat rotationMat);
	void wheelCheck(cv::Mat frame, bool& wheelLeftChecked, bool& wheelRightChecked, cv::Point2f wheelStartPoint, cv::Mat rotatedPoints, std::map<int, TrackedRect> resRects);
	float getWheelAngle(std::map<int, TrackedRect> resRects);
	float getWheelAngleWithCenter(std::map<int, TrackedRect> resRects);

	void setHandPos(std::map<int, TrackedRect> resRects);
	std::vector<Button> initButtons();
	void drawButtons(cv::Mat frame, std::vector<Button> buttons);
	bool isPointInsideRect(const cv::Point2d& point, const cv::Rect& rect);
	bool checkBoolVector(const std::vector<bool>& vec);
	void buttonEvent(cv::Mat frame, std::map<int, TrackedRect> resRects, std::vector<Button>& buttons, bool& isOutOfButton, std::chrono::steady_clock::time_point& startTime);


	UFUNCTION(BlueprintCallable)
	void ReadFrame();

	void Inference();

	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	UTexture2D* imageTexture;

	UTexture2D* MatToTexture2D(const cv::Mat InMat);






	// 손 변수들
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	bool is_handle;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float hand_left_x;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float hand_left_y;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float hand_right_x;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float hand_right_y;






	int webcamWidth = 640;
	int webcamHeight = 480;


	//var and functions with blaze
	Blaze blaze;
	cv::Mat img256;
	cv::Mat img128;
	float scale;
	cv::Scalar pad;

};