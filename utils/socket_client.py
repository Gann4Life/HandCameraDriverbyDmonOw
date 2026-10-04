"""
Socket client for communication with SteamVR driver.
"""
import socket
import threading
import time
from typing import Callable, Optional


class SocketClient:
    """Handles socket communication with the SteamVR driver."""

    def __init__(self, host: str = "127.0.0.1", port: int = 65432,
                 auto_reconnect: bool = True, reconnect_interval: float = 5.0,
                 greeting: Optional[Callable[[], str]] = None):
        """
        Initialize socket client.

        Args:
            host: Server host address
            port: Server port
            auto_reconnect: Whether to automatically reconnect on connection loss
            reconnect_interval: Seconds between reconnection attempts
            greeting: Returns the line sent first on every connection, before any data
        """
        self.host = host
        self.port = port
        self.auto_reconnect = auto_reconnect
        self.reconnect_interval = reconnect_interval
        self.greeting = greeting
        self.last_sent = 0.0
        self.socket: Optional[socket.socket] = None
        self.connected = False
        self.last_reconnect_attempt = 0.0
        self._connect_thread: Optional[threading.Thread] = None

    def connect(self) -> bool:
        """
        Connect to the server.
        
        Returns:
            True if connected successfully, False otherwise
        """
        try:
            if self.socket:
                self.close()
            
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(5.0)
            self.socket.connect((self.host, self.port))
            if self.greeting:
                self.socket.sendall((self.greeting() + "\n").encode("utf-8"))
            self.last_sent = time.monotonic()
            self.connected = True
            print(f"Connected to SteamVR driver at {self.host}:{self.port}")
            return True
            
        except Exception as e:
            print(f"Failed to connect to {self.host}:{self.port}: {e}")
            self.connected = False
            return False
    
    def send(self, data: str) -> bool:
        """
        Send data to the server.
        
        Args:
            data: String data to send
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self.connected:
            if self.auto_reconnect:
                current_time = time.time()
                if current_time - self.last_reconnect_attempt >= self.reconnect_interval:
                    self.last_reconnect_attempt = current_time
                    # On Windows a refused connect to localhost blocks ~2 s, so retry
                    # off the tracking loop instead of freezing it
                    if not (self._connect_thread and self._connect_thread.is_alive()):
                        print("Attempting to reconnect...")
                        self._connect_thread = threading.Thread(target=self.connect, daemon=True)
                        self._connect_thread.start()
            
            if not self.connected:
                return False
        
        try:
            # Ensure data ends with newline for easier parsing
            if not data.endswith('\n'):
                data += '\n'
            
            self.socket.sendall(data.encode('utf-8'))
            self.last_sent = time.monotonic()
            return True
            
        except Exception as e:
            print(f"Error sending data: {e}")
            self.connected = False
            return False
    
    def keepalive(self, interval: float = 1.0) -> None:
        """Repeat the greeting when nothing was sent for interval seconds: the driver drops silent connections."""
        if self.connected and self.greeting and time.monotonic() - self.last_sent >= interval:
            self.send(self.greeting())

    def close(self):
        """Close the socket connection."""
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
            self.socket = None
        self.connected = False
        print("Socket closed")
    
    def is_connected(self) -> bool:
        """Check if socket is connected."""
        return self.connected
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()