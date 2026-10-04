#pragma once

// The tracker's line protocol (docs/PROTOCOL.md), parsed and validated without
// OpenVR so it can be unit-tested. Everything from the socket is untrusted: a
// line with any malformed field is dropped whole, never half-applied.

#include <array>
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

enum class HandSide { Left, Right };

struct HandMessage
{
	// RECENTER:1. When set, the other fields are not filled in.
	bool recenter = false;

	HandSide hand = HandSide::Left;
	bool index_profile = false;  // TYPE:INDEX, otherwise Touch
	bool room_anchor = false;    // ANCHOR:ROOM

	// Fields the line leaves out keep the controller's previous value
	std::optional< std::array< float, 3 > > position;  // metres, each within +-kMaxPositionM
	std::optional< std::array< float, 4 > > rotation;  // qw, qx, qy, qz, unit length
	std::optional< float > trigger;                     // 0..1
	std::optional< float > grip;                        // 0..1
	std::optional< std::array< float, 5 > > curls;     // thumb..pinky, 0..1
};

// Hands further than this from the headset are a broken message, not a pose
constexpr float kMaxPositionM = 10.f;

// Longest line accepted; a real hand line is under 200 bytes
constexpr size_t kMaxLineBytes = 1024;

// The message in one line (without its newline), or nothing when the line is
// malformed: a bad number, NaN or infinity, a position out of range, a zero
// quaternion, a CURL without exactly 5 values, or a HAND other than LEFT/RIGHT.
std::optional< HandMessage > ParseHandMessage( std::string_view line );

// Splits a TCP byte stream into lines. A line longer than kMaxLineBytes is
// dropped whole, including the part that arrives after the limit.
class LineSplitter
{
public:
	void Feed( const char *data, size_t size, const std::function< void( const std::string & ) > &on_line );

private:
	std::string pending_;
	bool discarding_ = false;
};
