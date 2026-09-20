"""Plain HTTP calls to the registration-service (registration-service/server.js).
No token minting happens in this process — the server already does that via
livekit-server-sdk, same as it does for the Unity apps.
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import requests


@dataclasses.dataclass
class ViewerToken:
    token: str
    livekit_url: str


@dataclasses.dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: list[float]
    image_width: int
    image_height: int
    reprojection_error: float

    @staticmethod
    def from_json(d: dict) -> "Intrinsics":
        return Intrinsics(
            fx=d["fx"], fy=d["fy"], cx=d["cx"], cy=d["cy"],
            dist_coeffs=list(d.get("distCoeffs", [0, 0, 0, 0, 0])),
            image_width=d["imageWidth"], image_height=d["imageHeight"],
            reprojection_error=d.get("reprojectionError", -1.0),
        )


@dataclasses.dataclass
class Extrinsics:
    rvec: list[float]  # board->camera rotation (Rodrigues), see CameraCalibrationData.cs
    tvec: list[float]  # board->camera translation, metres

    @staticmethod
    def from_json(d: dict) -> "Extrinsics":
        return Extrinsics(rvec=list(d["rvec"]), tvec=list(d["tvec"]))


@dataclasses.dataclass
class CameraCalibration:
    camera_name: str
    intrinsics: Intrinsics
    extrinsics: Extrinsics

    @staticmethod
    def from_json(d: dict) -> "CameraCalibration":
        return CameraCalibration(
            camera_name=d.get("cameraName", ""),
            intrinsics=Intrinsics.from_json(d["intrinsics"]),
            extrinsics=Extrinsics.from_json(d["extrinsics"]),
        )


class RegistrationClient:
    def __init__(self, server_url: str, timeout_s: float = 5.0):
        self._base = server_url.rstrip("/")
        self._timeout = timeout_s

    def fetch_viewer_token(self, room_code: Optional[str] = None) -> ViewerToken:
        body = {}
        if room_code:
            body["roomCode"] = room_code
        resp = requests.post(f"{self._base}/viewer-token", json=body, timeout=self._timeout)
        resp.raise_for_status()
        data = resp.json()
        return ViewerToken(token=data["token"], livekit_url=data["livekit_url"])

    def fetch_calibration_pair(self, cam_a: str, cam_b: str) -> Optional[tuple[CameraCalibration, CameraCalibration]]:
        """Returns (calib_for_cam_a, calib_for_cam_b), or None if either is
        missing (404 — not calibrated / calibration expired)."""
        resp = requests.get(
            f"{self._base}/calibration-data/pair",
            params={"cam1": cam_a, "cam2": cam_b},
            timeout=self._timeout,
        )
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        data = resp.json()
        return CameraCalibration.from_json(data["cam1"]), CameraCalibration.from_json(data["cam2"])
