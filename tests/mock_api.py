"""In-process mock of the Pet Clinic API used by the end-to-end tests."""

from __future__ import annotations

import socket
import threading
import time

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

API_KEY = "test-key"


def make_app() -> Starlette:
    pets = {1: {"id": 1, "name": "Bruno", "species": "dog", "age": 4}, 2: {"id": 2, "name": "Mia", "species": "cat", "age": 2}}
    seen_keys: list[str] = []

    def authorized(request: Request) -> bool:
        seen_keys.append(request.headers.get("x-api-key", ""))
        return request.headers.get("x-api-key") == API_KEY

    async def pets_collection(request: Request):
        if not authorized(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        if request.method == "GET":
            species = request.query_params.get("species")
            return JSONResponse([p for p in pets.values() if not species or p["species"] == species])
        data = await request.json()
        new_id = max(pets) + 1
        pets[new_id] = {"id": new_id, **data}
        return JSONResponse(pets[new_id], status_code=201)

    async def pet_item(request: Request):
        if not authorized(request):
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        pet_id = int(request.path_params["pet_id"])
        if pet_id not in pets:
            return JSONResponse({"error": "not found"}, status_code=404)
        if request.method == "DELETE":
            del pets[pet_id]
            return Response(status_code=204)
        return JSONResponse(pets[pet_id])

    async def search(request: Request):
        q = (await request.json()).get("query", "").lower()
        return JSONResponse([p for p in pets.values() if q in p["name"].lower() or q in p["species"]])

    app = Starlette(routes=[
        Route("/api/v1/pets", pets_collection, methods=["GET", "POST"]),
        Route("/api/v1/pets/search", search, methods=["POST"]),
        Route("/api/v1/pets/{pet_id:int}", pet_item, methods=["GET", "DELETE", "PUT"]),
    ])
    app.state.pets = pets
    app.state.seen_keys = seen_keys
    return app


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MockServer:
    def __init__(self):
        self.app = make_app()
        self.port = free_port()
        self.server = uvicorn.Server(uvicorn.Config(self.app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/api/v1"

    def __enter__(self):
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                break
            time.sleep(0.05)
        return self

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(timeout=5)
