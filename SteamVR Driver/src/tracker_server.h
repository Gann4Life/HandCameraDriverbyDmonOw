#pragma once

// The socket the tracker connects to (docs/PROTOCOL.md), without OpenVR so it can
// be tested with real sockets. One client at a time, on 127.0.0.1 only. Anything
// can connect, so a connection is only served after the tracker's greeting: a
// client that stays silent, or opens with something else (a browser, another
// program), is closed and the next one is accepted.

#include <atomic>
#include <chrono>
#include <functional>
#include <string>
#include <thread>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <poll.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#define SOCKET int
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
#define closesocket close
#endif

class TrackerServer
{
public:
	using LineHandler = std::function< void( const std::string & ) >;
	using Logger = std::function< void( const std::string & ) >;

	// on_line gets each line after the greeting, without its newline, on the
	// server's thread. A client gets greeting_timeout to send its greeting, and is
	// closed after idle_timeout without a line, so it can't hold the only slot.
	TrackerServer( LineHandler on_line, Logger log,
		std::chrono::milliseconds greeting_timeout = std::chrono::seconds( 1 ),
		std::chrono::milliseconds idle_timeout = std::chrono::seconds( 5 ) );
	~TrackerServer();

	TrackerServer( const TrackerServer & ) = delete;
	TrackerServer &operator=( const TrackerServer & ) = delete;

	// Listens on 127.0.0.1:port (0 picks a free one). False when the port is
	// taken: no other process may share it.
	bool Start( int port );
	// Closes the connection and waits for the thread, within about kPoll
	void Stop();
	// The port it listens on, or 0 when not started
	int Port() const;

	static constexpr std::chrono::milliseconds kPoll{ 200 };

private:
	void Run();
	void Serve( SOCKET client );
	// The tracker's first line: true to serve the connection
	bool AcceptGreeting( const std::string &line );
	void LogOnce( bool &logged, const std::string &message );
	// At most one connection message a second
	void LogConnection( const std::string &message );

	LineHandler on_line_;
	Logger log_;
	std::chrono::milliseconds greeting_timeout_;
	std::chrono::milliseconds idle_timeout_;

	std::atomic< bool > running_{ false };
	std::thread thread_;
	// Opened in Start, closed in Stop after the thread is gone; only the thread
	// touches a client socket
	SOCKET server_ = INVALID_SOCKET;
	bool winsock_started_ = false;

	// Refusals repeat every few seconds while an old tracker retries
	bool logged_accept_error_ = false;
	bool logged_no_greeting_ = false;
	bool logged_bad_greeting_ = false;
	bool logged_version_ = false;
	std::chrono::steady_clock::time_point last_connection_log_{};
};
