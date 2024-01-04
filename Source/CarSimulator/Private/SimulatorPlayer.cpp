// Fill out your copyright notice in the Description page of Project Settings.


#include "SimulatorPlayer.h"
#include <GameFramework/SpringArmComponent.h>
#include <GameFramework/CharacterMovementComponent.h>
#include <Camera/CameraComponent.h>
#include "Runtime/Engine/Classes/Engine/Texture2D.h"
#include "Runtime/Engine/Classes/Engine/TextureRenderTarget2D.h"



// Sets default values
ASimulatorPlayer::ASimulatorPlayer()
{
 	// Set this character to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;

	InitStaticMeshs();

	fpSpringArmComp = CreateDefaultSubobject<USpringArmComponent>(TEXT("FPSpringArmComp"));
	fpSpringArmComp->SetupAttachment(RootComponent);
	fpSpringArmComp->SetRelativeLocation(FVector(0, -60, 70));
	fpSpringArmComp->TargetArmLength = 1;
	fpSpringArmComp->bUsePawnControlRotation = true;

	fpCamComp = CreateDefaultSubobject<UCameraComponent>(TEXT("FPCamComp"));
	fpCamComp->SetupAttachment(fpSpringArmComp);
	fpCamComp->bUsePawnControlRotation = false;

	bUseControllerRotationYaw = false;
	JumpMaxCount = 3;

	GetCharacterMovement()->MaxWalkSpeed = 1000.0f;
}

// Called when the game starts or when spawned
void ASimulatorPlayer::BeginPlay()
{
	Super::BeginPlay();
}

// Called every frame
void ASimulatorPlayer::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	Move();
}

// Called to bind functionality to input
void ASimulatorPlayer::SetupPlayerInputComponent(UInputComponent* PlayerInputComponent)
{
	Super::SetupPlayerInputComponent(PlayerInputComponent);

	PlayerInputComponent->BindAxis(TEXT("Turn"), this, &ASimulatorPlayer::Turn);
	PlayerInputComponent->BindAxis(TEXT("LookUp"), this, &ASimulatorPlayer::LookUp);
	PlayerInputComponent->BindAxis(TEXT("Horizontal"), this, &ASimulatorPlayer::InputHorizontal);
	PlayerInputComponent->BindAxis(TEXT("Vertical"), this, &ASimulatorPlayer::InputVertical);
}







void ASimulatorPlayer::Turn(float value)
{
	AddControllerYawInput(value);
}

void ASimulatorPlayer::LookUp(float value)
{
	AddControllerPitchInput(value);
}

void ASimulatorPlayer::InputHorizontal(float value)
{
	direction.Y = value;
}

void ASimulatorPlayer::InputVertical(float value)
{
	direction.X = value;
}

void ASimulatorPlayer::Move()
{
	direction = FTransform(GetControlRotation()).TransformVector(direction);
	AddMovementInput(direction);
	direction = FVector::ZeroVector;
}



void ASimulatorPlayer::InitStaticMeshs()
{

	/*frame*/
	ConstructorHelpers::FObjectFinder<UStaticMesh> TempFrameMesh(
		TEXT("StaticMesh'/Game/CitySampleVehicles/vehicle13_Car/Mesh/SM_vehCar_vehicle13_No_Wheel.SM_vehCar_vehicle13_No_Wheel'")
	);
	SMFrame = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("SM_Frame"));
	SMFrame->SetStaticMesh(TempFrameMesh.Object);
	SMFrame->SetRelativeLocation(FVector(0, 0, -90));
	SMFrame->SetupAttachment(RootComponent);

	/*trans*/
	ConstructorHelpers::FObjectFinder<UStaticMesh> TempTransMesh(
		TEXT("StaticMesh'/Game/CitySampleVehicles/vehicle13_Car/Mesh/SM_All_Trans_vehCar_vehicle13.SM_All_Trans_vehCar_vehicle13'")
	);
	SMAllTrans = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("SM_AllTrans"));
	SMAllTrans->SetStaticMesh(TempTransMesh.Object);
	SMAllTrans->SetRelativeLocation(FVector(0, 0, -90));
	SMAllTrans->SetupAttachment(RootComponent);


	/*wheel*/
	ConstructorHelpers::FObjectFinder<UStaticMesh> TempWheelRearRMesh(
		TEXT("StaticMesh'/Game/CitySampleVehicles/vehicle13_Car/Mesh/SM_Wheel_Rear_R_vehCar_vehicle13.SM_Wheel_Rear_R_vehCar_vehicle13'")
	);
	SMWheelRearR = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("SM_WheelRearR"));
	SMWheelRearR->SetStaticMesh(TempWheelRearRMesh.Object);
	SMWheelRearR->SetRelativeLocation(FVector(0, 0, -90));
	SMWheelRearR->SetupAttachment(RootComponent);

	ConstructorHelpers::FObjectFinder<UStaticMesh> TempWheelRearLMesh(
		TEXT("StaticMesh'/Game/CitySampleVehicles/vehicle13_Car/Mesh/SM_Wheel_Rear_L_vehCar_vehicle13.SM_Wheel_Rear_L_vehCar_vehicle13'")
	);
	SMWheelRearL = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("SM_WheelRearL"));
	SMWheelRearL->SetStaticMesh(TempWheelRearLMesh.Object);
	SMWheelRearL->SetRelativeLocation(FVector(0, 0, -90));
	SMWheelRearL->SetupAttachment(RootComponent);

	ConstructorHelpers::FObjectFinder<UStaticMesh> TempWheelFrontRMesh(
		TEXT("StaticMesh'/Game/CitySampleVehicles/vehicle13_Car/Mesh/SM_Wheel_Front_R_vehCar_vehicle13.SM_Wheel_Front_R_vehCar_vehicle13'")
	);
	SMWheelFrontR = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("SM_WheelFrontR"));
	SMWheelFrontR->SetStaticMesh(TempWheelFrontRMesh.Object);
	SMWheelFrontR->SetRelativeLocation(FVector(0, 0, -90));
	SMWheelFrontR->SetupAttachment(RootComponent);

	ConstructorHelpers::FObjectFinder<UStaticMesh> TempWheelFrontLMesh(
		TEXT("StaticMesh'/Game/CitySampleVehicles/vehicle13_Car/Mesh/SM_Wheel_Front_L_vehCar_vehicle13.SM_Wheel_Front_L_vehCar_vehicle13'")
	);
	SMWheelFrontL = CreateDefaultSubobject<UStaticMeshComponent>(TEXT("SM_WheelFrontL"));
	SMWheelFrontL->SetStaticMesh(TempWheelFrontLMesh.Object);
	SMWheelFrontL->SetRelativeLocation(FVector(0, 0, -90));
	SMWheelFrontL->SetupAttachment(RootComponent);
}

