#include "tracker_server.h"
#include "hand_message.h"

#include <cstring>
#include <exception>

namespace
{
	// 1 readable (or closed), 0 timed out, -1 error. poll rather than select: select's
	// fd_set overflows on POSIX for descriptors past FD_SETSIZE.
	int WaitReadable( SOCKET socket, std::chrono::milliseconds timeout )
	{
#ifdef _WIN32
		WSAPOLLFD fd{ socket, POLLRDNORM, 0 };
		const int ready = WSAPoll( &fd, 1, static_cast< INT >( timeout.count() ) );
#else
		pollfd fd{ socket, POLLIN, 0 };
		const int ready = poll( &fd, 1, static_cast< int >( timeout.count() ) );
#endif
		return ready > 0 ? 1 : ready;
	}
}

TrackerServer::TrackerServer( LineHandler on_line, Logger log, std::chrono::milliseconds greeting_timeout,
	std::chrono::milliseconds idle_timeout )
	: on_line_( std::move( on_line ) )
	, log_( std::move( log ) )
	, greeting_timeout_( greeting_timeout )
	, idle_timeout_( idle_timeout )
{
}

TrackerServer::~TrackerServer()
{
	Stop();
}

bool TrackerServer::Start( int port )
{
	if ( running_ )
	{
		return true;
	}
#ifdef _WIN32
	WSADATA wsa_data;
	if ( WSAStartup( MAKEWORD( 2, 2 ), &wsa_data ) != 0 )
	{
		log_( "WSAStartup failed" );
		return false;
	}
	winsock_started_ = true;
#endif

	server_ = socket( AF_INET, SOCK_STREAM, 0 );
	if ( server_ == INVALID_SOCKET )
	{
		log_( "failed to create the socket" );
		Stop();
		return false;
	}

	int opt = 1;
#ifdef _WIN32
	// Windows lets another process bind the same port with SO_REUSEADDR and take
	// the tracker's connections; this forbids it
	setsockopt( server_, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char *)&opt, sizeof( opt ) );
#else
	setsockopt( server_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof( opt ) );
#endif

	sockaddr_in address;
	std::memset( &address, 0, sizeof( address ) );
	address.sin_family = AF_INET;
	address.sin_addr.s_addr = htonl( INADDR_LOOPBACK );
	address.sin_port = htons( static_cast< unsigned short >( port ) );
	if ( bind( server_, (sockaddr *)&address, sizeof( address ) ) == SOCKET_ERROR
		|| listen( server_, 3 ) == SOCKET_ERROR )
	{
		log_( "port " + std::to_string( port ) + " is taken or can't be opened" );
		Stop();
		return false;
	}

	log_( "listening on port " + std::to_string( Port() ) );
	running_ = true;
	thread_ = std::thread( &TrackerServer::Run, this );
	return true;
}

void TrackerServer::Stop()
{
	running_ = false;
	if ( thread_.joinable() )
	{
		thread_.join();
	}
	if ( server_ != INVALID_SOCKET )
	{
		closesocket( server_ );
		server_ = INVALID_SOCKET;
	}
#ifdef _WIN32
	if ( winsock_started_ )
	{
		WSACleanup();
		winsock_started_ = false;
	}
#endif
}

int TrackerServer::Port() const
{
	if ( server_ == INVALID_SOCKET )
	{
		return 0;
	}
	sockaddr_in address{};
	socklen_t length = sizeof( address );
	if ( getsockname( server_, (sockaddr *)&address, &length ) == SOCKET_ERROR )
	{
		return 0;
	}
	return ntohs( address.sin_port );
}

void TrackerServer::Run()
{
	while ( running_ )
	{
		SOCKET client = INVALID_SOCKET;
		// Nothing may escape the thread: an exception leaving a std::thread ends vrserver
		try
		{
			// Polling instead of blocking in accept/recv lets Stop end the thread
			// without closing a socket under it
			const int ready = WaitReadable( server_, kPoll );
			if ( ready == 0 )
			{
				continue;
			}
			client = ready > 0 ? accept( server_, nullptr, nullptr ) : INVALID_SOCKET;
			if ( client == INVALID_SOCKET )
			{
				// Usually transient (a client that gave up while queued): keep listening
				LogOnce( logged_accept_error_, "failed to accept a connection (logged once)" );
				std::this_thread::sleep_for( kPoll );
				continue;
			}
			Serve( client );
		}
		catch ( ... )
		{
			try
			{
				log_( "connection closed after an error" );
			}
			catch ( ... )
			{
			}
		}
		if ( client != INVALID_SOCKET )
		{
			closesocket( client );
		}
	}
}

void TrackerServer::Serve( SOCKET client )
{
	using Clock = std::chrono::steady_clock;
	const auto greeting_deadline = Clock::now() + greeting_timeout_;
	auto last_line = Clock::now();
	bool greeted = false;
	bool refused = false;
	LineSplitter lines;
	char buffer[ 2048 ];

	while ( running_ && !refused )
	{
		// Checked on every pass, so a client dripping bytes can't hold the socket
		// either: only complete lines count
		const auto now = Clock::now();
		if ( !greeted && now >= greeting_deadline )
		{
			LogOnce( logged_no_greeting_, "closed a connection that sent no greeting in time (logged once)" );
			return;
		}
		if ( greeted && now - last_line >= idle_timeout_ )
		{
			LogConnection( "closed the tracker's connection: it went silent" );
			return;
		}
		const int ready = WaitReadable( client, kPoll );
		if ( ready < 0 )
		{
			LogConnection( "receive error" );
			return;
		}
		if ( ready == 0 )
		{
			continue;
		}

		const int received = recv( client, buffer, sizeof( buffer ), 0 );
		if ( received <= 0 )
		{
			LogConnection( received == 0 ? "tracker disconnected" : "receive error" );
			return;
		}
		// TCP is a stream: one recv can end in the middle of a line, so a line is
		// only handled once its newline arrives
		lines.Feed( buffer, static_cast< size_t >( received ), [ & ]( const std::string &line ) {
			if ( refused )
			{
				return;
			}
			last_line = Clock::now();
			if ( !greeted )
			{
				greeted = AcceptGreeting( line );
				refused = !greeted;
			}
			// A repeated greeting is the tracker's keepalive while it sees no hands
			else if ( !ParseGreeting( line ) )
			{
				on_line_( line );
			}
		} );
	}
}

bool TrackerServer::AcceptGreeting( const std::string &line )
{
	const std::optional< int > version = ParseGreeting( line );
	if ( !version )
	{
		LogOnce( logged_bad_greeting_,
			"closed a connection that didn't start with the tracker's greeting: another program, or a tracker "
			"app older than this driver (logged once)" );
		return false;
	}
	if ( *version != kProtocolVersion )
	{
		LogOnce( logged_version_, "the tracker speaks protocol version " + std::to_string( *version ) + " and this driver "
			+ std::to_string( kProtocolVersion ) + ": update " + ( *version > kProtocolVersion ? "the driver from the app's Add-ons" : "the app" ) );
		return false;
	}
	LogConnection( "tracker connected (protocol version " + std::to_string( *version ) + ")" );
	return true;
}

void TrackerServer::LogOnce( bool &logged, const std::string &message )
{
	if ( !logged )
	{
		logged = true;
		log_( message );
	}
}

void TrackerServer::LogConnection( const std::string &message )
{
	// A local loop of connect and close would otherwise fill the SteamVR log
	const auto now = std::chrono::steady_clock::now();
	if ( now - last_connection_log_ >= std::chrono::seconds( 1 ) )
	{
		last_connection_log_ = now;
		log_( message );
	}
}

