// Fill out your copyright notice in the Description page of Project Settings.

#pragma once
#include "PreOpenCVHeaders.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "PostOpenCVHeaders.h"
#include "YuNet.h"
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

	struct DetectResult {
		float score;
		int x;
		int y;
		int w;
		int h;
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
	cv::dnn::Net net_det;

	uint32_t nextIndex; //tracking id
	std::map<int, TrackedRect> trackedRectsMap; // map to store tracked rectangles

	cv::Mat wheelImage;

	cv::Mat image;
	cv::Rect handRoiRect;
	cv::Rect objDetectRoiRect;
	cv::Size detectInput;
	cv::Size detectOutput;

	cv::Mat morphKernel;
	cv::Scalar skinLowerBound;  // 살색의 하한값
	cv::Scalar skinUpperBound;  // 살색의 상한값


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

	// color tracking
	float calculateIOU(const cv::Rect& rect1, const cv::Rect& rect2);
	std::map<int, TrackedRect> getTrackedRects(std::map<int, TrackedRect> trackedRects, std::vector<cv::Rect> skinRegions, uint32_t& nextId);
	std::map<int, TrackedRect> getResRects(std::map<int, TrackedRect> trackedRects);
	void drawTrackedRect(cv::Mat frame, std::pair<int, TrackedRect> trackedRect);
	cv::Mat overlayTransparent(const cv::Mat& background_img, const cv::Mat& img_to_overlay_t, int x, int y);
	std::vector<cv::Rect> getSkinRegions(cv::Mat frame, cv::Rect roi, cv::Scalar lowerBound, cv::Scalar upperBound, cv::Mat kernel);
	
	// hand detect
	float sigmoid(float x);
	DetectResult getDetectResult(cv::Mat& frame, cv::Mat regressor, cv::Mat classificator, int stride, int anchor_count, int column, int row, int anchor, int offset, cv::Size detectInputSize, cv::Size detectOutputSize);
	std::vector<DetectResult> getDetectResults(cv::Mat frame, cv::dnn::Net net, cv::Size detectInputSize, cv::Size detectOutputSize);
	void drawDetectResult(cv::Mat frame, DetectResult res, cv::Rect objDetectRoi);
	void checkIsHand(std::map<int, TrackedRect>& trackedRects, std::vector<DetectResult>& detectResults, cv::Rect objDetectRoi);

	// control
	cv::Mat drawAndGetRotatedPoints(cv::Mat frame, cv::Point2f wheelStartPoint, cv::Mat Points, cv::Mat rotationMat);
	void wheelCheck(cv::Mat frame, bool& wheelLeftChecked, bool& wheelRightChecked, cv::Point2f wheelStartPoint, cv::Mat rotatedPoints, std::map<int, TrackedRect> resRects);
	float getWheelAngle(std::map<int, TrackedRect> resRects);
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






	//머리 자세 추론
	YuNet model;
	cv::Rect head_roi;
	cv::Size head_image_size;
	std::vector<int> xdiff_vector;
	int backend_id;
	int target_id;
	UPROPERTY(EditAnywhere, BlueprintReadWrite)
	float cliped_xdiff;

	cv::Mat figure_points_3D;
	cv::Mat camera_matrix;
	cv::Mat distortion_coeff;
	cv::Mat vector_rotation;
	cv::Mat vector_translation;

	const std::map<std::string, int> str2backend{
	{"opencv", cv::dnn::DNN_BACKEND_OPENCV}, {"cuda", cv::dnn::DNN_BACKEND_CUDA}
	};
	const std::map<std::string, int> str2target{
		{"cpu", cv::dnn::DNN_TARGET_CPU}, {"cuda", cv::dnn::DNN_TARGET_CUDA}, {"cuda_fp16", cv::dnn::DNN_TARGET_CUDA_FP16}
	};

	cv::Mat get_image_points_2D();
	cv::Mat get_figure_points_3D();
	cv::Mat get_camera_matrix();
	cv::Mat get_distortion_coeff();
	void estimate_chin(cv::Mat& image_points_2D);
	bool visualize(cv::Mat& image, const cv::Mat& faces, cv::Mat& image_points_2D);
	std::vector<cv::Point2d> get_pose_points(cv::Mat& image_points_2D, cv::Mat& vector_rotation, cv::Mat& vector_translation, cv::Mat& camera_matrix, cv::Mat& distortion_coeff);
	float clip_avg(std::vector<int> xdiff_vector);


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
};