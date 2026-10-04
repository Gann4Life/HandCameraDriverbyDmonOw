// Unit tests for the line parser (hand_message.h). No framework: each CHECK
// prints the failing expression, and the exit code is the number of failures.
// Build and run: cmake --build <build> --config Release --target hand_message_tests,
// then ctest --test-dir <build> -C Release.

#include "hand_message.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace
{
	int failures = 0;
	const char *current_test = "";

#define CHECK( expr ) \
	do { if ( !( expr ) ) { ++failures; std::printf( "FAIL %s: %s (line %d)\n", current_test, #expr, __LINE__ ); } } while ( 0 )

	bool Near( float a, float b )
	{
		return std::fabs( a - b ) < 1e-4f;
	}

	const std::string kGood =
		"HAND:LEFT,X:0.1200,Y:-0.3000,Z:-0.4000,QW:1.0000,QX:0.0000,QY:0.0000,QZ:0.0000,TRIGGER:0.80,GRIP:0.00,"
		"GESTURE:POINT,TYPE:INDEX,CURL:0.10;0.20;0.30;0.40;0.50,ANCHOR:ROOM";

	// kGood with one field's value replaced
	std::string With( const std::string &key, const std::string &value )
	{
		std::string line = kGood;
		const size_t start = line.find( key + ":" ) + key.size() + 1;
		const size_t end = line.find( ',', start );
		return line.replace( start, end == std::string::npos ? std::string::npos : end - start, value );
	}

	void ParsesTheProtocolExample()
	{
		current_test = "ParsesTheProtocolExample";
		const auto m = ParseHandMessage( kGood );
		CHECK( m.has_value() );
		if ( !m ) return;
		CHECK( !m->recenter );
		CHECK( m->hand == HandSide::Left );
		CHECK( m->index_profile );
		CHECK( m->room_anchor );
		CHECK( m->position && Near( ( *m->position )[ 0 ], 0.12f ) && Near( ( *m->position )[ 1 ], -0.3f ) && Near( ( *m->position )[ 2 ], -0.4f ) );
		CHECK( m->rotation && Near( ( *m->rotation )[ 0 ], 1.f ) );
		CHECK( m->trigger && Near( *m->trigger, 0.8f ) );
		CHECK( m->grip && Near( *m->grip, 0.f ) );
		CHECK( m->curls && Near( ( *m->curls )[ 0 ], 0.1f ) && Near( ( *m->curls )[ 4 ], 0.5f ) );
	}

	void OlderTrackerLineStillWorks()
	{
		current_test = "OlderTrackerLineStillWorks";
		const auto m = ParseHandMessage( "HAND:RIGHT,X:0,Y:0,Z:-0.3,QW:1,QX:0,QY:0,QZ:0,TRIGGER:0,GRIP:1,GESTURE:FIST" );
		CHECK( m && m->hand == HandSide::Right && !m->index_profile && !m->room_anchor && !m->curls );
	}

	void MissingFieldsAreLeftOut()
	{
		current_test = "MissingFieldsAreLeftOut";
		const auto m = ParseHandMessage( "HAND:LEFT,TRIGGER:0.5" );
		CHECK( m && !m->position && !m->rotation && !m->grip && !m->curls && m->trigger );
	}

	void Recenter()
	{
		current_test = "Recenter";
		const auto m = ParseHandMessage( "RECENTER:1" );
		CHECK( m && m->recenter );
	}

	void MalformedNumbersDropTheLine()
	{
		current_test = "MalformedNumbersDropTheLine";
		for ( const char *bad : { "", "abc", "nan", "NaN", "inf", "-inf", "1e40", "0.5x", " 0.5", "0..5", "--1" } )
		{
			for ( const char *key : { "X", "QW", "TRIGGER", "GRIP" } )
			{
				const bool dropped = !ParseHandMessage( With( key, bad ) );
				if ( !dropped ) std::printf( "  %s:'%s' was accepted\n", key, bad );
				CHECK( dropped );
			}
		}
	}

	void BadHandDropsTheLine()
	{
		current_test = "BadHandDropsTheLine";
		CHECK( !ParseHandMessage( With( "HAND", "MIDDLE" ) ) );
		CHECK( !ParseHandMessage( With( "HAND", "" ) ) );
		CHECK( !ParseHandMessage( "X:0,Y:0,Z:0" ) );
		CHECK( !ParseHandMessage( "" ) );
		CHECK( !ParseHandMessage( "garbage without any colon" ) );
	}

	void PartialPositionOrRotationDropsTheLine()
	{
		current_test = "PartialPositionOrRotationDropsTheLine";
		CHECK( !ParseHandMessage( "HAND:LEFT,X:0.1,Y:0.2" ) );
		CHECK( !ParseHandMessage( "HAND:LEFT,QW:1,QX:0,QY:0" ) );
	}

	void PositionsOutOfRangeDropTheLine()
	{
		current_test = "PositionsOutOfRangeDropTheLine";
		CHECK( !ParseHandMessage( With( "Z", "-10.5" ) ) );
		CHECK( !ParseHandMessage( With( "Y", "1e30" ) ) );
		CHECK( ParseHandMessage( With( "Z", "-9.9" ) ).has_value() );
	}

	void QuaternionIsNormalised()
	{
		current_test = "QuaternionIsNormalised";
		const auto m = ParseHandMessage( "HAND:LEFT,QW:2,QX:0,QY:0,QZ:0" );
		CHECK( m && m->rotation && Near( ( *m->rotation )[ 0 ], 1.f ) );
		CHECK( !ParseHandMessage( "HAND:LEFT,QW:0,QX:0,QY:0,QZ:0" ) );
		// Each component is finite but the sum of squares overflows a float
		CHECK( !ParseHandMessage( "HAND:LEFT,QW:3e38,QX:3e38,QY:0,QZ:0" ) );
	}

	void ControlsAreClamped()
	{
		current_test = "ControlsAreClamped";
		const auto high = ParseHandMessage( With( "TRIGGER", "1.7" ) );
		CHECK( high && Near( *high->trigger, 1.f ) );
		const auto low = ParseHandMessage( With( "GRIP", "-0.4" ) );
		CHECK( low && Near( *low->grip, 0.f ) );
		const auto curls = ParseHandMessage( With( "CURL", "-1;0.5;2;0;1" ) );
		CHECK( curls && Near( ( *curls->curls )[ 0 ], 0.f ) && Near( ( *curls->curls )[ 2 ], 1.f ) );
	}

	void CurlNeedsExactlyFiveValues()
	{
		current_test = "CurlNeedsExactlyFiveValues";
		for ( const char *bad : { "0.1;0.2;0.3;0.4", "0.1;0.2;0.3;0.4;0.5;0.6", "0.1;0.2;0.3;0.4;", "0.1 0.2 0.3 0.4 0.5",
			"0.1;0.2;;0.4;0.5", "0.1;0.2;nan;0.4;0.5", "" } )
		{
			const bool dropped = !ParseHandMessage( With( "CURL", bad ) );
			if ( !dropped ) std::printf( "  CURL:'%s' was accepted\n", bad );
			CHECK( dropped );
		}
	}

	void UnknownKeysAreIgnored()
	{
		current_test = "UnknownKeysAreIgnored";
		CHECK( ParseHandMessage( kGood + ",FUTURE:42" ).has_value() );
		CHECK( ParseHandMessage( With( "GESTURE", "SOMETHING_NEW" ) ).has_value() );
	}

	void OverlongLineIsDropped()
	{
		current_test = "OverlongLineIsDropped";
		CHECK( !ParseHandMessage( kGood + ",PAD:" + std::string( kMaxLineBytes, 'x' ) ) );
	}

	std::vector< std::string > Split( LineSplitter &splitter, const std::string &bytes )
	{
		std::vector< std::string > lines;
		splitter.Feed( bytes.data(), bytes.size(), [ & ]( const std::string &line ) { lines.push_back( line ); } );
		return lines;
	}

	void SplitterJoinsLinesAcrossReads()
	{
		current_test = "SplitterJoinsLinesAcrossReads";
		LineSplitter splitter;
		CHECK( Split( splitter, "HAND:LEFT,X:0." ).empty() );
		const auto lines = Split( splitter, "5\nHAND:RIGHT\r\n\nRECENTER:1\n" );
		CHECK( lines.size() == 3 );
		if ( lines.size() == 3 )
		{
			CHECK( lines[ 0 ] == "HAND:LEFT,X:0.5" );
			CHECK( lines[ 1 ] == "HAND:RIGHT" );
			CHECK( lines[ 2 ] == "RECENTER:1" );
		}
	}

	void SplitterDropsAnOverlongLineWhole()
	{
		current_test = "SplitterDropsAnOverlongLineWhole";
		LineSplitter splitter;
		// 1 MB without a newline, in reads like the socket's, then the line ends
		const std::string chunk( 2048, 'x' );
		size_t delivered = 0;
		for ( int i = 0; i < 512; i++ )
		{
			delivered += Split( splitter, chunk ).size();
		}
		// The tail of the long line must not come out as a line of its own
		const auto lines = Split( splitter, "HAND:LEFT,X:9\nRECENTER:1\n" );
		CHECK( delivered == 0 );
		CHECK( lines.size() == 1 && lines[ 0 ] == "RECENTER:1" );
	}
}

int main()
{
	ParsesTheProtocolExample();
	OlderTrackerLineStillWorks();
	MissingFieldsAreLeftOut();
	Recenter();
	MalformedNumbersDropTheLine();
	BadHandDropsTheLine();
	PartialPositionOrRotationDropsTheLine();
	PositionsOutOfRangeDropTheLine();
	QuaternionIsNormalised();
	ControlsAreClamped();
	CurlNeedsExactlyFiveValues();
	UnknownKeysAreIgnored();
	OverlongLineIsDropped();
	SplitterJoinsLinesAcrossReads();
	SplitterDropsAnOverlongLineWhole();
	std::printf( failures ? "%d check(s) failed\n" : "all checks passed\n", failures );
	return failures;
}
