// Fill out your copyright notice in the Description page of Project Settings.

#pragma once



#include "CoreMinimal.h"
#include "GameFramework/Character.h"
#include "SimulatorPlayer.generated.h"

UCLASS()
class CARSIMULATOR_API ASimulatorPlayer : public ACharacter
{
	GENERATED_BODY()

public:
	// Sets default values for this character's properties
	ASimulatorPlayer();

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;

	// Called to bind functionality to input
	virtual void SetupPlayerInputComponent(class UInputComponent* PlayerInputComponent) override;

public:
	/*static meshes*/
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Mesh)
	UStaticMeshComponent* SMFrame;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Mesh)
	UStaticMeshComponent* SMAllTrans;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Mesh)
	UStaticMeshComponent* SMWheelRearR;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Mesh)
	UStaticMeshComponent* SMWheelRearL;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Mesh)
	UStaticMeshComponent* SMWheelFrontR;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = Mesh)
	UStaticMeshComponent* SMWheelFrontL;


	/*camera move*/
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category=Camera)
	class USpringArmComponent* fpSpringArmComp;
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category=Camera)
	class UCameraComponent* fpCamComp;

	void Turn(float value);
	void LookUp(float value);

	
	/*player movement*/
	UPROPERTY(EditAnywhere, Category=PlayerSetting)
	float walkSpeed = 600;
	FVector direction;

	void InputHorizontal(float value);
	void InputVertical(float value);
	void Move();

public:
	void InitStaticMeshs();

};
