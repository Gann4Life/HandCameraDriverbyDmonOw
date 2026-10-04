//============ Copyright (c) Valve Corporation, All rights reserved. ============
#include "hand_tracking_listener.h"
#include "controller_device_driver.h"
#include "driverlog.h"

#include <cmath>

HandTrackingListener::HandTrackingListener( MyControllerDeviceDriver *left_controller, MyControllerDeviceDriver *right_controller )
	: left_controller_( left_controller )
	, right_controller_( right_controller )
	, server_(
		  [ this ]( const std::string &line ) { ProcessHandData( line ); },
		  []( const std::string &message ) { DriverLog( "HandTrackingListener: %s", message.c_str() ); } )
{
}

HandTrackingListener::~HandTrackingListener()
{
	Stop();
}

bool HandTrackingListener::Start( int port )
{
	return server_.Start( port );
}

void HandTrackingListener::Stop()
{
	server_.Stop();
}
void HandTrackingListener::ProcessHandData( const std::string &data )
{
	// One line of docs/PROTOCOL.md. A malformed line changes nothing.
	const std::optional<HandMessage> message = ParseHandMessage( data );
	if ( !message )
	{
		if ( !warned_bad_line_ )
		{
			warned_bad_line_ = true;
			DriverLog( "HandTrackingListener: dropped a malformed line from the tracker (logged once)" );
		}
		return;
	}

	// RECENTER:1 on a line of its own: the user is facing the camera now
	if ( message->recenter )
	{
		CaptureRoomForward();
		return;
	}

	// The devices are added to SteamVR with the type the first message asks for
	const int profile = static_cast<int>( message->index_profile ? ControllerProfile::Index : ControllerProfile::Touch );
	int expected = -1;
	if ( !requested_profile_.compare_exchange_strong( expected, profile ) && expected != profile && !warned_profile_change_ )
	{
		warned_profile_change_ = true;
		DriverLog( "HandTrackingListener: the tracker asks for another controller type; restart SteamVR to switch" );
	}

	MyControllerDeviceDriver *controller = message->hand == HandSide::Left ? left_controller_ : right_controller_;

	// A camera fixed in the room, or one that turns with the head (the default). Until the
	// user recenters, the camera is taken to be where the headset looks when the first
	// hand is seen: putting a hand in front of the camera usually means facing it.
	const bool room = message->room_anchor;
	if ( room && !room_forward_set_ )
	{
		CaptureRoomForward();
	}
	room_forward_set_ = room_forward_set_ && room;
	controller->SetRoomAnchor( room );

	if ( message->position )
	{
		const auto &p = *message->position;
		controller->UpdateHandPosition( p[ 0 ], p[ 1 ], p[ 2 ] );
	}
	if ( message->rotation )
	{
		const auto &q = *message->rotation;
		controller->UpdateHandRotation( q[ 0 ], q[ 1 ], q[ 2 ], q[ 3 ] );
	}
	if ( message->trigger )
	{
		controller->UpdateTriggerValue( *message->trigger );
	}
	if ( message->grip )
	{
		controller->UpdateGripValue( *message->grip );
	}
	if ( message->curls )
	{
		controller->UpdateFingerCurls( *message->curls );
	}
}

void HandTrackingListener::CaptureRoomForward()
{
	vr::TrackedDevicePose_t hmd_pose{};
	vr::VRServerDriverHost()->GetRawTrackedDevicePoses( 0.f, &hmd_pose, 1 );
	if ( !hmd_pose.bPoseIsValid )
	{
		return;
	}
	// Only the heading counts: the hands stay level however the head was tilted.
	// The headset looks along its -Z axis; turned by yaw about +Y, its Z axis is
	// (sin yaw, 0, cos yaw).
	const vr::HmdMatrix34_t &m = hmd_pose.mDeviceToAbsoluteTracking;
	const float yaw = std::atan2( m.m[ 0 ][ 2 ], m.m[ 2 ][ 2 ] );
	left_controller_->SetRoomYaw( yaw );
	right_controller_->SetRoomYaw( yaw );
	room_forward_set_ = true;
	// Every recenter and every ANCHOR switch lands here; a tracker can send those every frame
	const auto now = std::chrono::steady_clock::now();
	if ( now - last_direction_log_ >= std::chrono::seconds( 5 ) )
	{
		last_direction_log_ = now;
		DriverLog( "HandTrackingListener: camera direction set, %.0f degrees from the tracking space's forward",
			yaw * 57.29578f );
	}
}

bool HandTrackingListener::RequestedProfile( ControllerProfile &profile ) const
{
	const int requested = requested_profile_.load();
	if ( requested < 0 )
	{
		return false;
	}
	profile = static_cast<ControllerProfile>( requested );
	return true;
}
