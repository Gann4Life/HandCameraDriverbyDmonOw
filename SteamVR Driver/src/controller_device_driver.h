//============ Copyright (c) Valve Corporation, All rights reserved. ============
#pragma once

#include <array>
#include <string>

#include "openvr_driver.h"
#include <atomic>
#include <mutex>
#include <thread>

// Which real controller the device presents itself as, so that games use the
// bindings they already ship. It is fixed once SteamVR activates the device.
enum class ControllerProfile
{
	// Oculus Touch ({oculus}/input/touch_profile.json): the widest game support
	Touch,
	// Valve Index ({indexcontroller}/input/index_controller_profile.json): adds
	// per-finger curls and a hand skeleton, so games that read them show fingers
	Index,
};

// Components of both profiles. Each profile creates only its own; the rest keep
// an invalid handle and are skipped. Ones hand tracking has no equivalent for
// are created anyway and held at rest, because bindings expect them to exist.
// "primary"/"secondary" are the face buttons: A/B on the right Touch, X/Y on the
// left, A/B on both Index controllers.
enum MyComponent
{
	MyComponent_primary_click,
	MyComponent_primary_touch,
	MyComponent_secondary_click,
	MyComponent_secondary_touch,
	MyComponent_system_click,
	MyComponent_system_touch,

	MyComponent_trigger_value,
	MyComponent_trigger_touch,
	MyComponent_trigger_click,

	MyComponent_grip_value,
	MyComponent_grip_touch,
	MyComponent_grip_force,

	MyComponent_joystick_x,
	MyComponent_joystick_y,
	MyComponent_joystick_click,
	MyComponent_joystick_touch,

	MyComponent_thumbrest_touch,

	MyComponent_trackpad_x,
	MyComponent_trackpad_y,
	MyComponent_trackpad_touch,
	MyComponent_trackpad_force,

	MyComponent_finger_index,
	MyComponent_finger_middle,
	MyComponent_finger_ring,
	MyComponent_finger_pinky,

	MyComponent_skeleton,

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

	// Must be called before the device is added to SteamVR
	void SetProfile( ControllerProfile profile );

	void MyRunFrame();
	void MyProcessEvent( const vr::VREvent_t &vrevent );

	void MyPoseUpdateThread();

	// Hand tracking data update methods
	void UpdateHandPosition( float x, float y, float z );
	void UpdateHandRotation( float qw, float qx, float qy, float qz );
	void UpdateTriggerValue( float value );
	void UpdateGripValue( float value );
	// thumb, index, middle, ring, pinky; 0 straight .. 1 fully curled
	void UpdateFingerCurls( const std::array< float, 5 > &curls );
	// The hand's offset turns with the headset (a camera that sees where you look), or stays
	// facing a camera fixed in the room, at room_yaw radians about +Y from the tracking
	// space's forward
	void SetRoomAnchor( bool room );
	void SetRoomYaw( float room_yaw );

private:
	void CreateTouchComponents( vr::PropertyContainerHandle_t container );
	void CreateIndexComponents( vr::PropertyContainerHandle_t container );
	void UpdateSkeleton();
	void SetBoolean( MyComponent component, bool value );
	void SetScalar( MyComponent component, float value );

	std::atomic< vr::TrackedDeviceIndex_t > my_controller_index_;

	vr::ETrackedControllerRole my_controller_role_;
	ControllerProfile profile_ = ControllerProfile::Touch;

	std::string my_controller_model_number_;
	std::string my_controller_serial_number_;

	std::array< vr::VRInputComponentHandle_t, MyComponent_MAX > input_handles_;

	std::atomic< bool > is_active_;
	std::thread my_pose_update_thread_;

	// Hand tracking data. The pose is written by the listener thread and read by the
	// pose thread: one lock for all of it, so a read never mixes two updates into a
	// quaternion that isn't unit length.
	struct HandPose
	{
		vr::HmdVector3_t position{ 0.f, 0.f, 0.f };
		vr::HmdQuaternion_t rotation{ 1.f, 0.f, 0.f, 0.f };
	};
	HandPose hand_pose_;
	mutable std::mutex hand_pose_mutex_;
	std::atomic< float > trigger_value_;
	std::atomic< float > grip_value_;
	std::array< std::atomic< float >, 5 > finger_curls_;
	std::atomic< bool > room_anchor_{ false };
	std::atomic< float > room_yaw_{ 0.f };

	bool trigger_clicked_ = false;
};
