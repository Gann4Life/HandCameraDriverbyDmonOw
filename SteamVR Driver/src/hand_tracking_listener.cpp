//============ Copyright (c) Valve Corporation, All rights reserved. ============
#include "hand_tracking_listener.h"
#include "controller_device_driver.h"
#include "driverlog.h"

#include <cmath>
#include <cstring>

HandTrackingListener::HandTrackingListener( MyControllerDeviceDriver *left_controller, MyControllerDeviceDriver *right_controller )
	: left_controller_( left_controller )
	, right_controller_( right_controller )
	, is_running_( false )
	, server_socket_( INVALID_SOCKET )
	, client_socket_( INVALID_SOCKET )
	, port_( 65432 )
{
}

HandTrackingListener::~HandTrackingListener()
{
	Stop();
}

bool HandTrackingListener::Start( int port )
{
	port_ = port;

#ifdef _WIN32
	// Initialize Winsock
	WSADATA wsa_data;
	if ( WSAStartup( MAKEWORD( 2, 2 ), &wsa_data ) != 0 )
	{
		DriverLog( "HandTrackingListener: WSAStartup failed" );
		return false;
	}
#endif

	// Create socket
	server_socket_ = socket( AF_INET, SOCK_STREAM, 0 );
	if ( server_socket_ == INVALID_SOCKET )
	{
		DriverLog( "HandTrackingListener: Failed to create socket" );
#ifdef _WIN32
		WSACleanup();
#endif
		return false;
	}

	// Set socket options to allow reuse
	int opt = 1;
#ifdef _WIN32
	setsockopt( server_socket_, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof( opt ) );
#else
	setsockopt( server_socket_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof( opt ) );
#endif

	// Bind socket
	struct sockaddr_in server_addr;
	memset( &server_addr, 0, sizeof( server_addr ) );
	server_addr.sin_family = AF_INET;
	server_addr.sin_addr.s_addr = inet_addr( "127.0.0.1" );
	server_addr.sin_port = htons( port_ );

	if ( bind( server_socket_, (struct sockaddr *)&server_addr, sizeof( server_addr ) ) == SOCKET_ERROR )
	{
		DriverLog( "HandTrackingListener: Failed to bind socket to port %d", port_ );
		closesocket( server_socket_ );
#ifdef _WIN32
		WSACleanup();
#endif
		return false;
	}

	// Listen
	if ( listen( server_socket_, 3 ) == SOCKET_ERROR )
	{
		DriverLog( "HandTrackingListener: Failed to listen on socket" );
		closesocket( server_socket_ );
#ifdef _WIN32
		WSACleanup();
#endif
		return false;
	}

	DriverLog( "HandTrackingListener: Listening on port %d", port_ );

	// Start listening thread
	is_running_ = true;
	listen_thread_ = std::thread( &HandTrackingListener::ListenThread, this );

	return true;
}

void HandTrackingListener::Stop()
{
	if ( is_running_.exchange( false ) )
	{
		// Close sockets
		if ( client_socket_ != INVALID_SOCKET )
		{
			closesocket( client_socket_ );
			client_socket_ = INVALID_SOCKET;
		}
		if ( server_socket_ != INVALID_SOCKET )
		{
			closesocket( server_socket_ );
			server_socket_ = INVALID_SOCKET;
		}

		// Wait for thread to finish
		if ( listen_thread_.joinable() )
		{
			listen_thread_.join();
		}

#ifdef _WIN32
		WSACleanup();
#endif

		DriverLog( "HandTrackingListener: Stopped" );
	}
}

void HandTrackingListener::ListenThread()
{
	DriverLog( "HandTrackingListener: Thread started" );

	while ( is_running_ )
	{
		// Accept connection
		DriverLog( "HandTrackingListener: Waiting for client connection..." );
		struct sockaddr_in client_addr;
		socklen_t client_addr_len = sizeof( client_addr );
		client_socket_ = accept( server_socket_, (struct sockaddr *)&client_addr, &client_addr_len );

		if ( client_socket_ == INVALID_SOCKET )
		{
			if ( is_running_ )
			{
				DriverLog( "HandTrackingListener: Failed to accept connection" );
			}
			break;
		}

		DriverLog( "HandTrackingListener: Client connected" );

		// Receive data. TCP is a stream: one recv can end in the middle of a line, so
		// a line is only processed once its newline arrives.
		char buffer[ 2048 ];
		LineSplitter lines;
		while ( is_running_ )
		{
			int recv_size = recv( client_socket_, buffer, sizeof( buffer ), 0 );

			if ( recv_size > 0 )
			{
				lines.Feed( buffer, static_cast<size_t>( recv_size ), [ this ]( const std::string &line ) { ProcessHandData( line ); } );
			}
			else if ( recv_size == 0 )
			{
				DriverLog( "HandTrackingListener: Client disconnected" );
				break;
			}
			else
			{
				if ( is_running_ )
				{
					DriverLog( "HandTrackingListener: Receive error" );
				}
				break;
			}
		}

		closesocket( client_socket_ );
		client_socket_ = INVALID_SOCKET;
	}

	DriverLog( "HandTrackingListener: Thread stopped" );
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
	DriverLog( "HandTrackingListener: camera direction set, %.0f degrees from the tracking space's forward",
		yaw * 57.29578f );
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
