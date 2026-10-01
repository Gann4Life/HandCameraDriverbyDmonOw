//============ Copyright (c) Valve Corporation, All rights reserved. ============
#pragma once

#include <array>
#include <string>

#include "openvr_driver.h"
#include <atomic>
#include <thread>

// Components of an Oculus Touch controller ({oculus}/input/touch_profile.json),
// which this device presents itself as so that games use their existing
// Touch bindings. The face buttons are A/B on the right hand and X/Y on the left;
// "primary"/"secondary" stand for whichever pair this hand has.
enum MyComponent
{
	MyComponent_primary_click,
	MyComponent_primary_touch,
	MyComponent_secondary_click,
	MyComponent_secondary_touch,

	MyComponent_trigger_value,
	MyComponent_trigger_touch,

	MyComponent_grip_value,
	MyComponent_grip_touch,

	MyComponent_joystick_x,
	MyComponent_joystick_y,
	MyComponent_joystick_click,
	MyComponent_joystick_touch,

	MyComponent_thumbrest_touch,

	MyComponent_haptic,

	MyComponent_MAX
};

//-----------------------------------------------------------------------------
// Purpose: Represents a single tracked device in the system.
// What this device actually is (controller, hmd) depends on the
// properties you set within the device (see implementation of Activate)
//-----------------------------------------------------------------------------
class MyControllerDeviceDriver : public vr::ITrackedDeviceServerDriver
{
public:
	MyControllerDeviceDriver( vr::ETrackedControllerRole role );

	vr::EVRInitError Activate( uint32_t unObjectId ) override;

	void EnterStandby() override;

	void *GetComponent( const char *pchComponentNameAndVersion ) override;

	void DebugRequest( const char *pchRequest, char *pchResponseBuffer, uint32_t unResponseBufferSize ) override;

	vr::DriverPose_t GetPose() override;

	void Deactivate() override;

	// ----- Functions we declare ourselves below -----

	const std::string &MyGetSerialNumber();

	void MyRunFrame();
	void MyProcessEvent( const vr::VREvent_t &vrevent );

	void MyPoseUpdateThread();

	// Hand tracking data update methods
	void UpdateHandPosition( float x, float y, float z );
	void UpdateHandRotation( float qw, float qx, float qy, float qz );
	void UpdateTriggerValue( float value );
	void UpdateGripValue( float value );

private:
	std::atomic< vr::TrackedDeviceIndex_t > my_controller_index_;

	vr::ETrackedControllerRole my_controller_role_;

	std::string my_controller_model_number_;
	std::string my_controller_serial_number_;

	std::array< vr::VRInputComponentHandle_t, MyComponent_MAX > input_handles_;

	std::atomic< bool > is_active_;
	std::thread my_pose_update_thread_;

	// Hand tracking data
	std::atomic< float > hand_position_x_;
	std::atomic< float > hand_position_y_;
	std::atomic< float > hand_position_z_;
	std::atomic< float > hand_rotation_qw_;
	std::atomic< float > hand_rotation_qx_;
	std::atomic< float > hand_rotation_qy_;
	std::atomic< float > hand_rotation_qz_;
	std::atomic< float > trigger_value_;
	std::atomic< float > grip_value_;
};