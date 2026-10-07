import http.client
import json
import threading

import pytest
from http.server import ThreadingHTTPServer

from tools.dev_portal import ACTIONS, PortalHandler, validate_action


@pytest.fixture
def portal_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), PortalHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=2)


def test_portal_exposes_only_fixed_project_actions():
    assert set(ACTIONS) == {"prepare", "start", "stop", "test", "smoke", "validate"}


def test_portal_rejects_arbitrary_actions():
    with pytest.raises(ValueError, match="Unsupported action"):
        validate_action("run arbitrary shell command")


def test_portal_serves_ui_only_through_loopback(portal_server):
    connection = http.client.HTTPConnection("127.0.0.1", portal_server.server_port)
    connection.request("GET", "/", headers={"Host": f"127.0.0.1:{portal_server.server_port}"})

    response = connection.getresponse()

    assert response.status == 200
    assert b"RAG Service Portal" in response.read()
    connection.close()


def test_portal_blocks_action_without_same_origin(portal_server):
    connection = http.client.HTTPConnection("127.0.0.1", portal_server.server_port)
    connection.request(
        "POST",
        "/api/actions",
        body=json.dumps({"action": "test"}),
        headers={
            "Host": f"127.0.0.1:{portal_server.server_port}",
            "Content-Type": "application/json",
        },
    )

    response = connection.getresponse()

    assert response.status == 403
    connection.close()


def test_portal_rejects_unknown_action_over_http(portal_server):
    connection = http.client.HTTPConnection("127.0.0.1", portal_server.server_port)
    connection.request(
        "POST",
        "/api/actions",
        body=json.dumps({"action": "arbitrary"}),
        headers={
            "Host": f"127.0.0.1:{portal_server.server_port}",
            "Origin": f"http://127.0.0.1:{portal_server.server_port}",
            "Content-Type": "application/json",
        },
    )

    response = connection.getresponse()

    assert response.status == 400
    connection.close()