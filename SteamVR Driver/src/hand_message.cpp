#include "hand_message.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <map>

namespace
{
	// A finite float filling the whole value, or nothing. from_chars never
	// throws and ignores the locale, unlike std::stof.
	std::optional< float > ParseFloat( std::string_view text )
	{
		float value = 0.f;
		const char *end = text.data() + text.size();
		const auto result = std::from_chars( text.data(), end, value );
		if ( result.ec != std::errc() || result.ptr != end || !std::isfinite( value ) )
		{
			return std::nullopt;
		}
		return value;
	}

	float Clamp01( float value )
	{
		return std::clamp( value, 0.f, 1.f );
	}

	std::map< std::string_view, std::string_view > SplitFields( std::string_view line )
	{
		std::map< std::string_view, std::string_view > fields;
		while ( !line.empty() )
		{
			const size_t comma = line.find( ',' );
			const std::string_view token = line.substr( 0, comma );
			const size_t colon = token.find( ':' );
			if ( colon != std::string_view::npos )
			{
				fields[ token.substr( 0, colon ) ] = token.substr( colon + 1 );
			}
			if ( comma == std::string_view::npos )
			{
				break;
			}
			line.remove_prefix( comma + 1 );
		}
		return fields;
	}

	// Every listed key as a float; false when any is missing or malformed
	template < size_t N >
	bool ReadFloats( const std::map< std::string_view, std::string_view > &fields, const char *const ( &keys )[ N ],
		std::array< float, N > &out )
	{
		for ( size_t i = 0; i < N; i++ )
		{
			const auto it = fields.find( keys[ i ] );
			const std::optional< float > value = it == fields.end() ? std::nullopt : ParseFloat( it->second );
			if ( !value )
			{
				return false;
			}
			out[ i ] = *value;
		}
		return true;
	}

	// Whether any of the keys is present
	template < size_t N >
	bool AnyOf( const std::map< std::string_view, std::string_view > &fields, const char *const ( &keys )[ N ] )
	{
		return std::any_of( std::begin( keys ), std::end( keys ), [ & ]( const char *key ) { return fields.count( key ) > 0; } );
	}
}

std::optional< HandMessage > ParseHandMessage( std::string_view line )
{
	if ( line.size() > kMaxLineBytes )
	{
		return std::nullopt;
	}
	const auto fields = SplitFields( line );
	HandMessage message;

	if ( fields.count( "RECENTER" ) )
	{
		message.recenter = true;
		return message;
	}

	const auto hand = fields.find( "HAND" );
	if ( hand == fields.end() || ( hand->second != "LEFT" && hand->second != "RIGHT" ) )
	{
		return std::nullopt;
	}
	message.hand = hand->second == "LEFT" ? HandSide::Left : HandSide::Right;

	const auto type = fields.find( "TYPE" );
	message.index_profile = type != fields.end() && type->second == "INDEX";
	const auto anchor = fields.find( "ANCHOR" );
	message.room_anchor = anchor != fields.end() && anchor->second == "ROOM";

	// A group that is partly present is malformed: X without Y and Z can't place a hand
	static const char *const kPosition[] = { "X", "Y", "Z" };
	if ( AnyOf( fields, kPosition ) )
	{
		std::array< float, 3 > position{};
		if ( !ReadFloats( fields, kPosition, position ) )
		{
			return std::nullopt;
		}
		for ( float v : position )
		{
			if ( std::fabs( v ) > kMaxPositionM )
			{
				return std::nullopt;
			}
		}
		message.position = position;
	}

	static const char *const kRotation[] = { "QW", "QX", "QY", "QZ" };
	if ( AnyOf( fields, kRotation ) )
	{
		std::array< float, 4 > q{};
		if ( !ReadFloats( fields, kRotation, q ) )
		{
			return std::nullopt;
		}
		const float norm = std::sqrt( q[ 0 ] * q[ 0 ] + q[ 1 ] * q[ 1 ] + q[ 2 ] * q[ 2 ] + q[ 3 ] * q[ 3 ] );
		if ( !std::isfinite( norm ) || norm < 1e-6f )
		{
			return std::nullopt;
		}
		for ( float &c : q )
		{
			c /= norm;
		}
		message.rotation = q;
	}

	for ( auto [ key, target ] : { std::pair{ "TRIGGER", &message.trigger }, std::pair{ "GRIP", &message.grip } } )
	{
		const auto it = fields.find( key );
		if ( it == fields.end() )
		{
			continue;
		}
		const std::optional< float > value = ParseFloat( it->second );
		if ( !value )
		{
			return std::nullopt;
		}
		*target = Clamp01( *value );
	}

	const auto curl = fields.find( "CURL" );
	if ( curl != fields.end() )
	{
		std::array< float, 5 > curls{};
		std::string_view rest = curl->second;
		for ( size_t i = 0; i < curls.size(); i++ )
		{
			const size_t semicolon = rest.find( ';' );
			const bool last = i + 1 == curls.size();
			// Exactly five values: a separator after each but the last
			if ( ( semicolon == std::string_view::npos ) != last )
			{
				return std::nullopt;
			}
			const std::optional< float > value = ParseFloat( rest.substr( 0, semicolon ) );
			if ( !value )
			{
				return std::nullopt;
			}
			curls[ i ] = Clamp01( *value );
			if ( !last )
			{
				rest.remove_prefix( semicolon + 1 );
			}
		}
		message.curls = curls;
	}

	return message;
}

void LineSplitter::Feed( const char *data, size_t size, const std::function< void( const std::string & ) > &on_line )
{
	for ( size_t i = 0; i < size; i++ )
	{
		const char c = data[ i ];
		if ( c == '\n' )
		{
			if ( !discarding_ )
			{
				if ( !pending_.empty() && pending_.back() == '\r' )
				{
					pending_.pop_back();
				}
				if ( !pending_.empty() )
				{
					on_line( pending_ );
				}
			}
			pending_.clear();
			discarding_ = false;
		}
		else if ( !discarding_ )
		{
			if ( pending_.size() >= kMaxLineBytes )
			{
				// Too long to be a hand line: drop it up to its newline
				pending_.clear();
				discarding_ = true;
			}
			else
			{
				pending_.push_back( c );
			}
		}
	}
}
