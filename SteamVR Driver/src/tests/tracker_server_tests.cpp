// Tests for the tracker's socket (tracker_server.h) with real connections on a
// free local port. Same style as hand_message_tests: each CHECK prints the
// failing expression, and the exit code is 1 when any failed.

#include "tracker_server.h"
#include "hand_message.h"

#include <chrono>
#include <cstdio>
#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono_literals;

namespace
{
	int failures = 0;
	const char *current_test = "";

#define CHECK( expr ) \
	do { if ( !( expr ) ) { ++failures; std::printf( "FAIL %s: %s (line %d)\n", current_test, #expr, __LINE__ ); } } while ( 0 )

	constexpr auto kGreetingTimeout = 300ms;
	constexpr auto kIdleTimeout = 600ms;
	const std::string kGreeting = "HELLO:HANDCAM,VERSION:" + std::to_string( kProtocolVersion ) + ",TYPE:INDEX\n";
	const std::string kHandLine = "HAND:LEFT,X:0.1000,Y:0.0000,Z:-0.3000";

	// A server that records the lines it hands over
	struct Fixture
	{
		std::mutex mutex;
		std::vector< std::string > lines;
		std::function< void( const std::string & ) > on_line = [ this ]( const std::string &line ) {
			std::lock_guard< std::mutex > lock( mutex );
			lines.push_back( line );
		};
		TrackerServer server{ [ this ]( const std::string &line ) { on_line( line ); },
			[]( const std::string & ) {}, kGreetingTimeout, kIdleTimeout };

		Fixture() { CHECK( server.Start( 0 ) ); }

		std::vector< std::string > Lines()
		{
			std::lock_guard< std::mutex > lock( mutex );
			return lines;
		}

		// Waits up to timeout for at least count lines
		bool WaitForLines( size_t count, std::chrono::milliseconds timeout = 2s )
		{
			const auto deadline = std::chrono::steady_clock::now() + timeout;
			while ( std::chrono::steady_clock::now() < deadline )
			{
				if ( Lines().size() >= count )
				{
					return true;
				}
				std::this_thread::sleep_for( 10ms );
			}
			return false;
		}
	};

	class Client
	{
	public:
		explicit Client( int port )
		{
			socket_ = socket( AF_INET, SOCK_STREAM, 0 );
			sockaddr_in address{};
			address.sin_family = AF_INET;
			address.sin_addr.s_addr = htonl( INADDR_LOOPBACK );
			address.sin_port = htons( static_cast< unsigned short >( port ) );
			connected_ = connect( socket_, (sockaddr *)&address, sizeof( address ) ) == 0;
		}
		~Client() { closesocket( socket_ ); }

		bool Connected() const { return connected_; }
		void Send( const std::string &data ) { send( socket_, data.data(), static_cast< int >( data.size() ), 0 ); }

		// Whether the server closes the connection within timeout
		bool ClosedByServer( std::chrono::milliseconds timeout )
		{
			fd_set readable;
			FD_ZERO( &readable );
			FD_SET( socket_, &readable );
			timeval tv{ static_cast< long >( timeout.count() / 1000 ), static_cast< long >( ( timeout.count() % 1000 ) * 1000 ) };
			if ( select( static_cast< int >( socket_ + 1 ), &readable, nullptr, nullptr, &tv ) <= 0 )
			{
				return false;
			}
			char byte;
			return recv( socket_, &byte, 1, 0 ) <= 0;
		}

	private:
		SOCKET socket_ = INVALID_SOCKET;
		bool connected_ = false;
	};

	void GreetedTrackerLinesArrive()
	{
		current_test = "GreetedTrackerLinesArrive";
		Fixture f;
		Client tracker( f.server.Port() );
		CHECK( tracker.Connected() );
		// Split mid-line, as TCP may deliver it
		tracker.Send( kGreeting + kHandLine.substr( 0, 10 ) );
		std::this_thread::sleep_for( 50ms );
		tracker.Send( kHandLine.substr( 10 ) + "\nRECENTER:1\n" );
		CHECK( f.WaitForLines( 2 ) );
		const auto lines = f.Lines();
		// The greeting itself is not a message
		CHECK( lines.size() == 2 && lines[ 0 ] == kHandLine && lines[ 1 ] == "RECENTER:1" );
	}

	void SilentClientIsDroppedAndTheTrackerGetsIn()
	{
		current_test = "SilentClientIsDroppedAndTheTrackerGetsIn";
		Fixture f;
		Client silent( f.server.Port() );
		CHECK( silent.Connected() );
		std::this_thread::sleep_for( 50ms );
		// Queued behind the silent one until it is dropped
		Client tracker( f.server.Port() );
		tracker.Send( kGreeting + kHandLine + "\n" );
		CHECK( silent.ClosedByServer( kGreetingTimeout + 1s ) );
		CHECK( f.WaitForLines( 1 ) );
	}

	void GreetedThenSilentIsDroppedAndTheTrackerGetsIn()
	{
		current_test = "GreetedThenSilentIsDroppedAndTheTrackerGetsIn";
		Fixture f;
		Client squatter( f.server.Port() );
		squatter.Send( kGreeting );
		std::this_thread::sleep_for( 50ms );
		Client tracker( f.server.Port() );
		tracker.Send( kGreeting + kHandLine + "\n" );
		CHECK( squatter.ClosedByServer( kIdleTimeout + 1s ) );
		CHECK( f.WaitForLines( 1 ) );
	}

	void KeepaliveHoldsTheConnectionWithoutLines()
	{
		current_test = "KeepaliveHoldsTheConnectionWithoutLines";
		Fixture f;
		Client tracker( f.server.Port() );
		tracker.Send( kGreeting );
		// Twice the idle timeout, greeting again as the tracker does while it sees no hands
		for ( int i = 0; i < 6; i++ )
		{
			CHECK( !tracker.ClosedByServer( kIdleTimeout / 3 ) );
			tracker.Send( kGreeting );
		}
		tracker.Send( kHandLine + "\n" );
		CHECK( f.WaitForLines( 1 ) );
		// The repeated greetings are not messages
		CHECK( f.Lines().size() == 1 );
	}

	void SlowDripGreetingIsDropped()
	{
		current_test = "SlowDripGreetingIsDropped";
		Fixture f;
		Client client( f.server.Port() );
		// Bytes keep coming, but the greeting never ends: the deadline still holds
		const auto start = std::chrono::steady_clock::now();
		bool closed = false;
		while ( !closed && std::chrono::steady_clock::now() - start < kGreetingTimeout + 1s )
		{
			client.Send( "H" );
			closed = client.ClosedByServer( 50ms );
		}
		CHECK( closed );
	}

	void OtherFirstLinesAreClosed()
	{
		current_test = "OtherFirstLinesAreClosed";
		Fixture f;
		for ( const std::string &first : { std::string( "GET / HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n" ), kHandLine + "\n",
				  std::string( "HELLO:HANDCAM,VERSION:" ) + std::to_string( kProtocolVersion + 1 ) + "\n" } )
		{
			Client client( f.server.Port() );
			client.Send( first + kHandLine + "\n" );
			CHECK( client.ClosedByServer( 1s ) );
		}
		CHECK( f.Lines().empty() );
	}

	void PortIsNotShared()
	{
		current_test = "PortIsNotShared";
		Fixture f;
		// Another process trying to share it with SO_REUSEADDR must fail
		SOCKET other = socket( AF_INET, SOCK_STREAM, 0 );
		int opt = 1;
		setsockopt( other, SOL_SOCKET, SO_REUSEADDR, (const char *)&opt, sizeof( opt ) );
		sockaddr_in address{};
		address.sin_family = AF_INET;
		address.sin_addr.s_addr = htonl( INADDR_LOOPBACK );
		address.sin_port = htons( static_cast< unsigned short >( f.server.Port() ) );
		CHECK( bind( other, (sockaddr *)&address, sizeof( address ) ) == SOCKET_ERROR );
		closesocket( other );

		TrackerServer second( []( const std::string & ) {}, []( const std::string & ) {} );
		CHECK( !second.Start( f.server.Port() ) );
	}

	void StopIsQuickWithATrackerConnected()
	{
		current_test = "StopIsQuickWithATrackerConnected";
		Fixture f;
		Client tracker( f.server.Port() );
		tracker.Send( kGreeting + kHandLine + "\n" );
		CHECK( f.WaitForLines( 1 ) );
		const auto start = std::chrono::steady_clock::now();
		f.server.Stop();
		CHECK( std::chrono::steady_clock::now() - start < 1s );
		CHECK( tracker.ClosedByServer( 1s ) );
		CHECK( f.server.Port() == 0 );
	}

	void AnErrorInAHandlerKeepsTheServerRunning()
	{
		current_test = "AnErrorInAHandlerKeepsTheServerRunning";
		Fixture f;
		bool thrown = false;
		f.on_line = [ &f, &thrown ]( const std::string &line ) {
			if ( !thrown )
			{
				thrown = true;
				throw std::runtime_error( "test" );
			}
			std::lock_guard< std::mutex > lock( f.mutex );
			f.lines.push_back( line );
		};
		{
			Client first( f.server.Port() );
			first.Send( kGreeting + kHandLine + "\n" );
			CHECK( first.ClosedByServer( 1s ) );
		}
		Client second( f.server.Port() );
		second.Send( kGreeting + kHandLine + "\n" );
		CHECK( f.WaitForLines( 1 ) );
	}
}

int main()
{
#ifdef _WIN32
	// The clients need Winsock before any server starts it
	WSADATA wsa_data;
	WSAStartup( MAKEWORD( 2, 2 ), &wsa_data );
#endif
	GreetedTrackerLinesArrive();
	SilentClientIsDroppedAndTheTrackerGetsIn();
	GreetedThenSilentIsDroppedAndTheTrackerGetsIn();
	KeepaliveHoldsTheConnectionWithoutLines();
	SlowDripGreetingIsDropped();
	OtherFirstLinesAreClosed();
	PortIsNotShared();
	StopIsQuickWithATrackerConnected();
	AnErrorInAHandlerKeepsTheServerRunning();
#ifdef _WIN32
	WSACleanup();
#endif
	std::printf( failures ? "%d check(s) failed\n" : "all checks passed\n", failures );
	return failures ? 1 : 0;
}
