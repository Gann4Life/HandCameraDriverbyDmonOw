//============ Copyright (c) Valve Corporation, All rights reserved. ============
#pragma once

#include <atomic>
#include <chrono>
#include <string>

#include "controller_device_driver.h"
#include "hand_message.h"
#include "tracker_server.h"

//-----------------------------------------------------------------------------
// Purpose: Applies the tracker's messages (docs/PROTOCOL.md) to the controllers.
// The socket itself is TrackerServer's.
//-----------------------------------------------------------------------------
class HandTrackingListener
{
public:
	HandTrackingListener( MyControllerDeviceDriver *left_controller, MyControllerDeviceDriver *right_controller );
	~HandTrackingListener();

	bool Start( int port = 65432 );
	void Stop();

	// The controller type the tracker asked for in its first message, or false
	// before any message arrived. Trackers that do not say get Touch.
	bool RequestedProfile( ControllerProfile &profile ) const;

private:
	void ProcessHandData( const std::string &data );
	// Where the headset looks now becomes the direction of a camera fixed in the room
	void CaptureRoomForward();

	MyControllerDeviceDriver *left_controller_;
	MyControllerDeviceDriver *right_controller_;

	// -1 until the first message, then a ControllerProfile
	std::atomic<int> requested_profile_{ -1 };
	// Only touched on the server's thread
	bool warned_profile_change_ = false;
	bool warned_bad_line_ = false;
	bool room_forward_set_ = false;
	std::chrono::steady_clock::time_point last_direction_log_{};

	TrackerServer server_;
};
