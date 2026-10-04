"""SocketClient against a local server socket standing in for the driver."""
import socket

import pytest

from utils.socket_client import SocketClient


@pytest.fixture
def driver():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    server.settimeout(5.0)
    yield server
    server.close()


def received(server: socket.socket, client: SocketClient) -> bytes:
    """Everything the client sent, once it closes."""
    connection, _ = server.accept()
    with connection:
        connection.settimeout(5.0)
        client.close()
        data = b""
        while chunk := connection.recv(4096):
            data += chunk
    return data


def test_the_greeting_comes_first_on_every_connection(driver):
    client = SocketClient(port=driver.getsockname()[1], auto_reconnect=False, greeting=lambda: "HELLO:TEST")
    for _ in range(2):
        assert client.connect()
        client.send("HAND:LEFT")
        assert received(driver, client) == b"HELLO:TEST\nHAND:LEFT\n"


def test_keepalive_repeats_the_greeting_only_when_idle(driver):
    client = SocketClient(port=driver.getsockname()[1], auto_reconnect=False, greeting=lambda: "HELLO:TEST")
    assert client.connect()
    client.keepalive(interval=60.0)  # just connected: nothing to add
    client.send("HAND:LEFT")
    client.keepalive(interval=0.0)
    assert received(driver, client) == b"HELLO:TEST\nHAND:LEFT\nHELLO:TEST\n"


def test_no_greeting_by_default(driver):
    client = SocketClient(port=driver.getsockname()[1], auto_reconnect=False)
    assert client.connect()
    client.send("HAND:LEFT")
    assert received(driver, client) == b"HAND:LEFT\n"
