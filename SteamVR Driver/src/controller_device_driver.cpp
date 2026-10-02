//============ Copyright (c) Valve Corporation, All rights reserved. ============
#include "controller_device_driver.h"

#include "driverlog.h"
#include "hand_simulation.h"
#include "vrmath.h"

// Let's create some variables for strings used in getting settings.
// This is the section where all of the settings we want are stored. A section name can be anything,
// but if you want to store driver specific settings, it's best to namespace the section with the driver identifier
// ie "<my_driver>_<section>" to avoid collisions
static const char *my_controller_main_settings_section = "driver_hand_camera_tracking";

// Individual right/left hand settings sections
static const char *my_controller_left_settings_section = "driver_hand_camera_tracking_left_hand";
static const char *my_controller_right_settings_section = "driver_hand_camera_tracking_right_hand";

// These are the keys we want to retrieve the values for in the settings
static const char *my_controller_settings_key_model_number = "model_number";
static const char *my_controller_settings_key_serial_number = "serial_number";


MyControllerDeviceDriver::MyControllerDeviceDriver( vr::ETrackedControllerRole role )
{
	// Set a member to keep track of whether we've activated yet or not
	is_active_ = false;

	// The constructor takes a role argument, that gives us information about if our controller is a left or right hand.
	// Let's store it for later use. We'll need it.
	my_controller_role_ = role;

	// No handles exist until Activate creates them; MyRunFrame checks for this
	input_handles_.fill( vr::k_ulInvalidInputComponentHandle );

	// We have our model number and serial number stored in SteamVR settings. We need to get them and do so here.
	// Other IVRSettings methods (to get int32, floats, bools) return the data, instead of modifying, but strings are
	// different.
	// Default to the model number a real Quest 2 controller reports, since some games pick
	// per-controller hand offsets from it.
	char model_number[ 1024 ] = { 0 };
	vr::VRSettings()->GetString( my_controller_main_settings_section, my_controller_settings_key_model_number, model_number, sizeof( model_number ) );
	my_controller_model_number_ = model_number[ 0 ] ? model_number
		: ( my_controller_role_ == vr::TrackedControllerRole_LeftHand ? "Oculus Quest2 (Left Controller)" : "Oculus Quest2 (Right Controller)" );

	// Get our serial number depending on our "handedness"
	// SteamVR rejects devices that register with an empty serial, so fall back to a
	// synthetic unique value rather than letting TrackedDeviceAdded fail.
	const char *serial_section = my_controller_role_ == vr::TrackedControllerRole_LeftHand ? my_controller_left_settings_section : my_controller_right_settings_section;
	char serial_number[ 1024 ] = { 0 };
	vr::VRSettings()->GetString( serial_section, my_controller_settings_key_serial_number, serial_number, sizeof( serial_number ) );
	if ( serial_number[ 0 ] )
	{
		my_controller_serial_number_ = serial_number;
	}
	else
	{
		my_controller_serial_number_ = my_controller_role_ == vr::TrackedControllerRole_LeftHand ? "WebcamLeftHandABC123" : "WebcamRightHandXYZ789";
	}

	// Initialize hand tracking data with neutral values
	hand_position_x_ = 0.0f;
	hand_position_y_ = 0.0f;
	hand_position_z_ = 0.0f;
	hand_rotation_qw_ = 1.0f;  // Identity quaternion
	hand_rotation_qx_ = 0.0f;
	hand_rotation_qy_ = 0.0f;
	hand_rotation_qz_ = 0.0f;
	trigger_value_ = 0.0f;
	grip_value_ = 0.0f;
	for ( auto &curl : finger_curls_ )
	{
		curl = 0.0f;
	}

	// Here's an example of how to use our logging wrapper around IVRDriverLog
	// In SteamVR logs (SteamVR Hamburger Menu > Developer Settings > Web console) drivers have a prefix of
	// "<driver_name>:". You can search this in the top search bar to find the info that you've logged.
	DriverLog( "My Controller Model Number: %s", my_controller_model_number_.c_str() );
	DriverLog( "My Controller Serial Number: %s", my_controller_serial_number_.c_str() );
}

//-----------------------------------------------------------------------------
// Purpose: This is called by vrserver after our
//  IServerTrackedDeviceProvider calls IVRServerDriverHost::TrackedDeviceAdded.
//-----------------------------------------------------------------------------
vr::EVRInitError MyControllerDeviceDriver::Activate( uint32_t unObjectId )
{
	// Set an member to keep track of whether we've activated yet or not
	is_active_ = true;

	// Let's keep track of our device index. It'll be useful later.
	my_controller_index_ = unObjectId;

	// Properties are stored in containers, usually one container per device index. We need to get this container to set
	// The properties we want, so we call this to retrieve a handle to it.
	vr::PropertyContainerHandle_t container = vr::VRProperties()->TrackedDeviceToPropertyContainer( my_controller_index_ );

	// Let's begin setting up the properties now we've got our container.
	// A list of properties available is contained in vr::ETrackedDeviceProperty.

	// Let's tell SteamVR our role which we received from the constructor earlier.
	vr::VRProperties()->SetInt32Property( container, vr::Prop_ControllerRoleHint_Int32, my_controller_role_ );

	// Other drivers (e.g. a streaming headset's own controllers) may claim the
	// same hand roles. Higher numbers win the hand assignment.
	vr::VRProperties()->SetInt32Property( container, vr::Prop_ControllerHandSelectionPriority_Int32, 1000 );

	// Present as a real controller. Games ship bindings for Touch and Index but
	// have never heard of a custom controller type, and under SteamVR Input even
	// the hand pose is an action that only reaches the game through a binding;
	// without one the hands simply do not appear. The profiles, render models and
	// legacy bindings all come from the drivers bundled with SteamVR.
	if ( profile_ == ControllerProfile::Index )
	{
		CreateIndexComponents( container );
	}
	else
	{
		CreateTouchComponents( container );
	}

	// Let's create our haptic component.
	// These are global across the device, and you can only have one per device.
	vr::VRDriverInput()->CreateHapticComponent( container, "/output/haptic", &input_handles_[ MyComponent_haptic ] );

	// SteamVR wants skeletal data as soon as the skeleton exists
	UpdateSkeleton();

	my_pose_update_thread_ = std::thread( &MyControllerDeviceDriver::MyPoseUpdateThread, this );

	// We've activated everything successfully!
	// Let's tell SteamVR that by saying we don't have any errors.
	return vr::VRInitError_None;
}

// CreateScalarComponent requires:
// EVRScalarType - whether the device can give an absolute position, or just one relative to where it was last. We
// can do it absolute.
// EVRScalarUnits - whether the devices has two "sides", like a joystick. This makes the range of valid inputs -1
// to 1. Otherwise, it's 0 to 1. Triggers are one-sided, the joystick two-sided.
static void CreateOneSided( vr::PropertyContainerHandle_t container, const char *path, vr::VRInputComponentHandle_t *handle )
{
	vr::VRDriverInput()->CreateScalarComponent( container, path, handle, vr::VRScalarType_Absolute, vr::VRScalarUnits_NormalizedOneSided );
}

static void CreateTwoSided( vr::PropertyContainerHandle_t container, const char *path, vr::VRInputComponentHandle_t *handle )
{
	vr::VRDriverInput()->CreateScalarComponent( container, path, handle, vr::VRScalarType_Absolute, vr::VRScalarUnits_NormalizedTwoSided );
}

void MyControllerDeviceDriver::CreateTouchComponents( vr::PropertyContainerHandle_t container )
{
	const bool is_left = my_controller_role_ == vr::TrackedControllerRole_LeftHand;
	// Default to the model number a real Quest 2 controller reports (see the constructor),
	// since some games pick per-controller hand offsets from it.
	vr::VRProperties()->SetStringProperty( container, vr::Prop_ModelNumber_String, my_controller_model_number_.c_str() );
	vr::VRProperties()->SetStringProperty( container, vr::Prop_ControllerType_String, "oculus_touch" );
	vr::VRProperties()->SetStringProperty( container, vr::Prop_InputProfilePath_String, "{oculus}/input/touch_profile.json" );
	vr::VRProperties()->SetStringProperty( container, vr::Prop_RenderModelName_String, is_left ? "oculus_quest2_controller_left" : "oculus_quest2_controller_right" );
	vr::VRProperties()->SetStringProperty( container, vr::Prop_ManufacturerName_String, "Oculus" );

	const std::string primary = is_left ? "/input/x" : "/input/a";
	const std::string secondary = is_left ? "/input/y" : "/input/b";
	vr::VRDriverInput()->CreateBooleanComponent( container, ( primary + "/click" ).c_str(), &input_handles_[ MyComponent_primary_click ] );
	vr::VRDriverInput()->CreateBooleanComponent( container, ( primary + "/touch" ).c_str(), &input_handles_[ MyComponent_primary_touch ] );
	vr::VRDriverInput()->CreateBooleanComponent( container, ( secondary + "/click" ).c_str(), &input_handles_[ MyComponent_secondary_click ] );
	vr::VRDriverInput()->CreateBooleanComponent( container, ( secondary + "/touch" ).c_str(), &input_handles_[ MyComponent_secondary_touch ] );

	CreateOneSided( container, "/input/trigger/value", &input_handles_[ MyComponent_trigger_value ] );
	vr::VRDriverInput()->CreateBooleanComponent( container, "/input/trigger/touch", &input_handles_[ MyComponent_trigger_touch ] );

	CreateOneSided( container, "/input/grip/value", &input_handles_[ MyComponent_grip_value ] );
	vr::VRDriverInput()->CreateBooleanComponent( container, "/input/grip/touch", &input_handles_[ MyComponent_grip_touch ] );

	CreateTwoSided( container, "/input/joystick/x", &input_handles_[ MyComponent_joystick_x ] );
	CreateTwoSided( container, "/input/joystick/y", &input_handles_[ MyComponent_joystick_y ] );
	vr::VRDriverInput()->CreateBooleanComponent( container, "/input/joystick/click", &input_handles_[ MyComponent_joystick_click ] );
	vr::VRDriverInput()->CreateBooleanComponent( container, "/input/joystick/touch", &input_handles_[ MyComponent_joystick_touch ] );

	vr::VRDriverInput()->CreateBooleanComponent( container, "/input/thumbrest/touch", &input_handles_[ MyComponent_thumbrest_touch ] );
}

void MyControllerDeviceDriver::CreateIndexComponents( vr::PropertyContainerHandle_t container )
{
	const bool is_left = my_controller_role_ == vr::TrackedControllerRole_LeftHand;
	// What a real Index controller reports, for games that pick hand offsets from it
	vr::VRProperties()->SetStringProperty( container, vr::Prop_ModelNumber_String, is_left ? "Knuckles Left" : "Knuckles Right" );
	vr::VRProperties()->SetStringProperty( container, vr::Prop_ControllerType_String, "knuckles" );
	vr::VRProperties()->SetStringProperty( container, vr::Prop_InputProfilePath_String, "{indexcontroller}/input/index_controller_profile.json" );
	vr::VRProperties()->SetStringProperty( container, vr::Prop_RenderModelName_String,
		is_left ? "{indexcontroller}valve_controller_knu_1_0_left" : "{indexcontroller}valve_controller_knu_1_0_right" );
	vr::VRProperties()->SetStringProperty( container, vr::Prop_ManufacturerName_String, "Valve" );

	vr::VRDriverInput()->CreateBooleanComponent( container, "/input/a/click", &input_handles_[ MyComponent_primary_click ] );
	vr::VRDriverInput()->CreateBooleanComponent( container, "/input/a/touch", &input_handles_[ MyComponent_primary_touch ] );
	vr::VRDriverInput()->CreateBooleanComponent( container, "/input/b/click", &input_handles_[ MyComponent_secondary_click ] );
	vr::VRDriverInput()->CreateBooleanComponent( container, "/input/b/touch", &input_handles_[ MyComponent_secondary_touch ] );
	vr::VRDriverInput()->CreateBooleanComponent( container, "/input/system/click", &input_handles_[ MyComponent_system_click ] );
	vr::VRDriverInput()->CreateBooleanComponent( container, "/input/system/touch", &input_handles_[ MyComponent_system_touch ] );

	CreateOneSided( container, "/input/trigger/value", &input_handles_[ MyComponent_trigger_value ] );
	vr::VRDriverInput()->CreateBooleanComponent( container, "/input/trigger/touch", &input_handles_[ MyComponent_trigger_touch ] );
	vr::VRDriverInput()->CreateBooleanComponent( container, "/input/trigger/click", &input_handles_[ MyComponent_trigger_click ] );

	// Index grip: value is how far the fingers close around it, force how hard they squeeze
	CreateOneSided( container, "/input/grip/value", &input_handles_[ MyComponent_grip_value ] );
	vr::VRDriverInput()->CreateBooleanComponent( container, "/input/grip/touch", &input_handles_[ MyComponent_grip_touch ] );
	CreateOneSided( container, "/input/grip/force", &input_handles_[ MyComponent_grip_force ] );

	CreateTwoSided( container, "/input/thumbstick/x", &input_handles_[ MyComponent_joystick_x ] );
	CreateTwoSided( container, "/input/thumbstick/y", &input_handles_[ MyComponent_joystick_y ] );
	vr::VRDriverInput()->CreateBooleanComponent( container, "/input/thumbstick/click", &input_handles_[ MyComponent_joystick_click ] );
	vr::VRDriverInput()->CreateBooleanComponent( container, "/input/thumbstick/touch", &input_handles_[ MyComponent_joystick_touch ] );

	CreateTwoSided( container, "/input/trackpad/x", &input_handles_[ MyComponent_trackpad_x ] );
	CreateTwoSided( container, "/input/trackpad/y", &input_handles_[ MyComponent_trackpad_y ] );
	vr::VRDriverInput()->CreateBooleanComponent( container, "/input/trackpad/touch", &input_handles_[ MyComponent_trackpad_touch ] );
	CreateOneSided( container, "/input/trackpad/force", &input_handles_[ MyComponent_trackpad_force ] );

	CreateOneSided( container, "/input/finger/index", &input_handles_[ MyComponent_finger_index ] );
	CreateOneSided( container, "/input/finger/middle", &input_handles_[ MyComponent_finger_middle ] );
	CreateOneSided( container, "/input/finger/ring", &input_handles_[ MyComponent_finger_ring ] );
	CreateOneSided( container, "/input/finger/pinky", &input_handles_[ MyComponent_finger_pinky ] );

	vr::VRDriverInput()->CreateSkeletonComponent( container,
		is_left ? "/input/skeleton/left" : "/input/skeleton/right",
		is_left ? "/skeleton/hand/left" : "/skeleton/hand/right",
		"/pose/raw",					// the skeleton's origin, from the render model
		vr::VRSkeletalTracking_Partial, // fingers come from a camera, not measured per joint
		nullptr, 0,						// default grip limits
		&input_handles_[ MyComponent_skeleton ] );
}

//-----------------------------------------------------------------------------
// Purpose: If you're an HMD, this is where you would return an implementation
// of vr::IVRDisplayComponent, vr::IVRVirtualDisplay or vr::IVRDirectModeComponent.
//
// But this a simple example to demo for a controller, so we'll just return nullptr here.
//-----------------------------------------------------------------------------
void *MyControllerDeviceDriver::GetComponent( const char *pchComponentNameAndVersion )
{
	return nullptr;
}

//-----------------------------------------------------------------------------
// Purpose: This is called by vrserver when a debug request has been made from an application to the driver.
// What is in the response and request is up to the application and driver to figure out themselves.
//-----------------------------------------------------------------------------
void MyControllerDeviceDriver::DebugRequest( const char *pchRequest, char *pchResponseBuffer, uint32_t unResponseBufferSize )
{
	if ( unResponseBufferSize >= 1 )
		pchResponseBuffer[ 0 ] = 0;
}

//-----------------------------------------------------------------------------
// Purpose: This is never called by vrserver in recent OpenVR versions,
// but is useful for giving data to vr::VRServerDriverHost::TrackedDevicePoseUpdated.
//-----------------------------------------------------------------------------
vr::DriverPose_t MyControllerDeviceDriver::GetPose()
{
	// Let's retrieve the Hmd pose to base our controller pose off.

	// First, initialize the struct that we'll be submitting to the runtime to tell it we've updated our pose.
	vr::DriverPose_t pose = { 0 };

	// These need to be set to be valid quaternions. The device won't appear otherwise.
	pose.qWorldFromDriverRotation.w = 1.f;
	pose.qDriverFromHeadRotation.w = 1.f;

	vr::TrackedDevicePose_t hmd_pose{};

	// GetRawTrackedDevicePoses expects an array.
	// We only want the hmd pose, which is at index 0 of the array so we can just pass the struct in directly, instead of in an array
	vr::VRServerDriverHost()->GetRawTrackedDevicePoses( 0.f, &hmd_pose, 1 );

	// Get the position of the hmd from the 3x4 matrix GetRawTrackedDevicePoses returns
	const vr::HmdVector3_t hmd_position = HmdVector3_From34Matrix( hmd_pose.mDeviceToAbsoluteTracking );
	// Get the orientation of the hmd from the 3x4 matrix GetRawTrackedDevicePoses returns
	const vr::HmdQuaternion_t hmd_orientation = HmdQuaternion_FromMatrix( hmd_pose.mDeviceToAbsoluteTracking );

	// Use hand tracking rotation if available, otherwise use default orientation
	vr::HmdQuaternion_t hand_rotation;
	hand_rotation.w = hand_rotation_qw_.load();
	hand_rotation.x = hand_rotation_qx_.load();
	hand_rotation.y = hand_rotation_qy_.load();
	hand_rotation.z = hand_rotation_qz_.load();

	// Apply hand rotation to the HMD orientation
	pose.qRotation = hmd_orientation * hand_rotation;

	// Use hand tracking position if available
	const vr::HmdVector3_t offset_position = {
		hand_position_x_.load(),
		hand_position_y_.load(),
		hand_position_z_.load()
	};

	// Rotate our offset by the hmd quaternion (so the controllers are always facing towards us), and add then add the position of the hmd to put it into position.
	vr::HmdVector3_t position = hmd_position + ( offset_position * hmd_orientation );

	// The tracker places a Touch controller in the hand. An Index controller's
	// origin sits elsewhere in the hand, so move it to where an Index would be
	// held. Both render models give the pose of the hand on the controller
	// ("openxr_handmodel"): Touch (-0.01125, -0.00183, 0.10195) m at (-39.4, 0, 0)
	// deg, Index (-0.015, -0.015, 0.13) m at (-40, -5, 0) deg, left hand; the right
	// is the mirror image. Index = Touch * handmodel_touch * inverse(handmodel_index).
	if ( profile_ == ControllerProfile::Index )
	{
		const bool is_left = my_controller_role_ == vr::TrackedControllerRole_LeftHand;
		const float side = is_left ? 1.f : -1.f;
		const vr::HmdQuaternion_t touch_to_index = { 0.99903, 0.00523, side * 0.04362, side * 0.00023 };
		const vr::HmdVector3_t touch_to_index_offset = { side * -0.0076f, 0.0145f, -0.0287f };
		position = position + ( touch_to_index_offset * pose.qRotation );
		pose.qRotation = pose.qRotation * touch_to_index;
	}

	// copy our position to our pose
	pose.vecPosition[ 0 ] = position.v[ 0 ];
	pose.vecPosition[ 1 ] = position.v[ 1 ];
	pose.vecPosition[ 2 ] = position.v[ 2 ];

	// The pose we provided is valid.
	// This should be set is
	pose.poseIsValid = true;

	// Our device is always connected.
	// In reality with physical devices, when they get disconnected,
	// set this to false and icons in SteamVR will be updated to show the device is disconnected
	pose.deviceIsConnected = true;

	// The state of our tracking. For our virtual device, it's always going to be ok,
	// but this can get set differently to inform the runtime about the state of the device's tracking
	// and update the icons to inform the user accordingly.
	pose.result = vr::TrackingResult_Running_OK;

	return pose;
}

void MyControllerDeviceDriver::MyPoseUpdateThread()
{
	while ( is_active_ )
	{
		// Inform the vrserver that our tracked device's pose has updated, giving it the pose returned by our GetPose().
		vr::VRServerDriverHost()->TrackedDevicePoseUpdated( my_controller_index_, GetPose(), sizeof( vr::DriverPose_t ) );

		// Update our pose every five milliseconds.
		// In reality, you should update the pose whenever you have new data from your device.
		std::this_thread::sleep_for( std::chrono::milliseconds( 5 ) );
	}
}

//-----------------------------------------------------------------------------
// Purpose: This is called by vrserver when the device should enter standby mode.
// The device should be put into whatever low power mode it has.
// We don't really have anything to do here, so let's just log something.
//-----------------------------------------------------------------------------
void MyControllerDeviceDriver::EnterStandby()
{
	DriverLog( "%s hand has been put on standby", my_controller_role_ == vr::TrackedControllerRole_LeftHand ? "Left" : "Right" );
}

//-----------------------------------------------------------------------------
// Purpose: This is called by vrserver when the device should deactivate.
// This is typically at the end of a session
// The device should free any resources it has allocated here.
//-----------------------------------------------------------------------------
void MyControllerDeviceDriver::Deactivate()
{
	// Let's join our pose thread that's running
	// by first checking then setting is_active_ to false to break out
	// of the while loop, if it's running, then call .join() on the thread
	if ( is_active_.exchange( false ) )
	{
		my_pose_update_thread_.join();
	}

	// unassign our controller index (we don't want to be calling vrserver anymore after Deactivate() has been called
	my_controller_index_ = vr::k_unTrackedDeviceIndexInvalid;
}


//-----------------------------------------------------------------------------
// Purpose: This is called by our IServerTrackedDeviceProvider when its RunFrame() method gets called.
// It's not part of the ITrackedDeviceServerDriver interface, we created it ourselves.
//-----------------------------------------------------------------------------
void MyControllerDeviceDriver::MyRunFrame()
{
	// RunFrame can be called before Activate has populated the input handles, in which case
	// every handle is still 0 and SteamVR logs "Invalid handle" for each call.
	if ( !is_active_.load() || input_handles_[ MyComponent_trigger_value ] == vr::k_ulInvalidInputComponentHandle )
	{
		return;
	}

	// Update our inputs here with data from hand tracking
	float trigger_val = trigger_value_.load();
	float grip_val = grip_value_.load();

	// Games read the analog values, and a "touch" when the finger rests on them
	SetScalar( MyComponent_trigger_value, trigger_val );
	SetBoolean( MyComponent_trigger_touch, trigger_val > 0.1f );
	SetScalar( MyComponent_grip_value, grip_val );
	SetBoolean( MyComponent_grip_touch, grip_val > 0.1f );

	// Index only (the others have no handle on Touch). The trigger clicks near
	// the end of its travel, with some slack so it does not chatter; squeezing
	// starts once the hand is mostly closed.
	trigger_clicked_ = trigger_val > ( trigger_clicked_ ? 0.85f : 0.95f );
	SetBoolean( MyComponent_trigger_click, trigger_clicked_ );
	SetScalar( MyComponent_grip_force, grip_val > 0.7f ? ( grip_val - 0.7f ) / 0.3f : 0.f );
	SetScalar( MyComponent_finger_index, finger_curls_[ 1 ].load() );
	SetScalar( MyComponent_finger_middle, finger_curls_[ 2 ].load() );
	SetScalar( MyComponent_finger_ring, finger_curls_[ 3 ].load() );
	SetScalar( MyComponent_finger_pinky, finger_curls_[ 4 ].load() );
	UpdateSkeleton();

	// No hand-tracking equivalent yet for face buttons, stick, trackpad or thumbrest:
	// hold them at rest (gestures could be mapped here later)
	for ( MyComponent button : { MyComponent_primary_click, MyComponent_primary_touch, MyComponent_secondary_click,
			  MyComponent_secondary_touch, MyComponent_system_click, MyComponent_system_touch, MyComponent_joystick_click,
			  MyComponent_joystick_touch, MyComponent_thumbrest_touch, MyComponent_trackpad_touch } )
	{
		SetBoolean( button, false );
	}
	for ( MyComponent axis : { MyComponent_joystick_x, MyComponent_joystick_y, MyComponent_trackpad_x, MyComponent_trackpad_y,
			  MyComponent_trackpad_force } )
	{
		SetScalar( axis, 0.f );
	}
}

void MyControllerDeviceDriver::SetBoolean( MyComponent component, bool value )
{
	if ( input_handles_[ component ] != vr::k_ulInvalidInputComponentHandle )
	{
		vr::VRDriverInput()->UpdateBooleanComponent( input_handles_[ component ], value, 0 );
	}
}

void MyControllerDeviceDriver::SetScalar( MyComponent component, float value )
{
	if ( input_handles_[ component ] != vr::k_ulInvalidInputComponentHandle )
	{
		vr::VRDriverInput()->UpdateScalarComponent( input_handles_[ component ], value, 0 );
	}
}

//-----------------------------------------------------------------------------
// Purpose: Pose the hand skeleton from the finger curls (Index only). Valve's
// hand skeleton simulation sample turns curls into bone transforms.
//-----------------------------------------------------------------------------
void MyControllerDeviceDriver::UpdateSkeleton()
{
	if ( input_handles_[ MyComponent_skeleton ] == vr::k_ulInvalidInputComponentHandle )
	{
		return;
	}
	const MyFingerCurls curls = { finger_curls_[ 0 ].load(), finger_curls_[ 1 ].load(), finger_curls_[ 2 ].load(),
		finger_curls_[ 3 ].load(), finger_curls_[ 4 ].load() };
	const MyFingerSplays splays = { 0.f, 0.f, 0.f, 0.f, 0.f };
	vr::VRBoneTransform_t transforms[ eBone_Count ];
	MyHandSimulation().ComputeSkeletonTransforms( my_controller_role_, curls, splays, transforms );
	// The same hand with and without a controller: there is no controller to hold
	vr::VRDriverInput()->UpdateSkeletonComponent( input_handles_[ MyComponent_skeleton ], vr::VRSkeletalMotionRange_WithController, transforms, eBone_Count );
	vr::VRDriverInput()->UpdateSkeletonComponent( input_handles_[ MyComponent_skeleton ], vr::VRSkeletalMotionRange_WithoutController, transforms, eBone_Count );
}


//-----------------------------------------------------------------------------
// Purpose: This is called by our IServerTrackedDeviceProvider when it pops an event off the event queue.
// It's not part of the ITrackedDeviceServerDriver interface, we created it ourselves.
//-----------------------------------------------------------------------------
void MyControllerDeviceDriver::MyProcessEvent( const vr::VREvent_t &vrevent )
{
	switch ( vrevent.eventType )
	{
		// Listen for haptic events
		case vr::VREvent_Input_HapticVibration:
		{
			// We now need to make sure that the event was intended for this device.
			// So let's compare handles of the event and our haptic component

			if ( vrevent.data.hapticVibration.componentHandle == input_handles_[ MyComponent_haptic ] )
			{
				// The event was intended for us!
				// To convert the data to a pulse, see the docs.
				// For this driver, we'll just print the values.

				float duration = vrevent.data.hapticVibration.fDurationSeconds;
				float frequency = vrevent.data.hapticVibration.fFrequency;
				float amplitude = vrevent.data.hapticVibration.fAmplitude;

				DriverLog( "Haptic event triggered for %s hand. Duration: %.2f, Frequency: %.2f, Amplitude: %.2f", my_controller_role_ == vr::TrackedControllerRole_LeftHand ? "left" : "right",
					duration, frequency, amplitude );
			}
			break;
		}
		default:
			break;
	}
}

//-----------------------------------------------------------------------------
// Purpose: Our IServerTrackedDeviceProvider needs our serial number to add us to vrserver.
// It's not part of the ITrackedDeviceServerDriver interface, we created it ourselves.
//-----------------------------------------------------------------------------
const std::string &MyControllerDeviceDriver::MyGetSerialNumber()
{
	return my_controller_serial_number_;
}

//-----------------------------------------------------------------------------
// Purpose: Update hand position from hand tracking data
//-----------------------------------------------------------------------------
void MyControllerDeviceDriver::UpdateHandPosition( float x, float y, float z )
{
	// Throttled diagnostic so incoming tracking values can be verified from vrserver.txt.
	// Roughly one line per second, per hand.
	static std::atomic<int> log_counter{ 0 };
	if ( ( ++log_counter % 90 ) == 1 )
	{
		const bool is_left = my_controller_role_ == vr::TrackedControllerRole_LeftHand;
		DriverLog( "HandTracking %s pos: x=%.3f y=%.3f z=%.3f trigger=%.2f grip=%.2f active=%d",
			is_left ? "LEFT" : "RIGHT",
			x, y, z,
			trigger_value_.load(), grip_value_.load(),
			is_active_.load() ? 1 : 0 );
	}

	hand_position_x_.store( x );
	hand_position_y_.store( y );
	hand_position_z_.store( z );
}

//-----------------------------------------------------------------------------
// Purpose: Update hand rotation from hand tracking data
//-----------------------------------------------------------------------------
void MyControllerDeviceDriver::UpdateHandRotation( float qw, float qx, float qy, float qz )
{
	hand_rotation_qw_.store( qw );
	hand_rotation_qx_.store( qx );
	hand_rotation_qy_.store( qy );
	hand_rotation_qz_.store( qz );
}

//-----------------------------------------------------------------------------
// Purpose: Update trigger value from gesture detection
//-----------------------------------------------------------------------------
void MyControllerDeviceDriver::UpdateTriggerValue( float value )
{
	trigger_value_.store( value );
}

//-----------------------------------------------------------------------------
// Purpose: Update grip value from gesture detection
//-----------------------------------------------------------------------------
void MyControllerDeviceDriver::UpdateGripValue( float value )
{
	grip_value_.store( value );
}

void MyControllerDeviceDriver::UpdateFingerCurls( const std::array< float, 5 > &curls )
{
	for ( size_t i = 0; i < curls.size(); i++ )
	{
		finger_curls_[ i ].store( curls[ i ] );
	}
}

void MyControllerDeviceDriver::SetProfile( ControllerProfile profile )
{
	profile_ = profile;
}