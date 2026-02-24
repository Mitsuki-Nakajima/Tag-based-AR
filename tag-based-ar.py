import argparse
import sys
from dataclasses import dataclass

import cv2
import numpy as np

ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
}


@dataclass
class CameraModel:
    K: np.ndarray      # 3x3 camera matrix
    dist: np.ndarray   # distortion coefficients


def default_camera_model(frame_w: int, frame_h: int) -> CameraModel:
    # For accurate pose, calibrate your camera.
    # fx, fy are focal lengths in pixels; cx, cy is the principal point.
    fx = 0.9 * frame_w
    fy = 0.9 * frame_w
    cx = frame_w / 2.0
    cy = frame_h / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    return CameraModel(K=K, dist=dist)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ArUco tag detection + pose demo")
    p.add_argument("--camera", type=int, default=0, help="Webcam index (0, 1, ...)")
    p.add_argument("--dict", type=str, default="DICT_4X4_50", choices=ARUCO_DICTS.keys(),
                   help="ArUco dictionary")
    p.add_argument("--marker-length", type=float, default=0.04,
                   help="Marker side length in meters (e.g., 0.04 = 4cm)")
    p.add_argument("--no-pose", action="store_true",
                   help="Only detect markers (skip pose estimation)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: cannot open camera index {args.camera}", file=sys.stderr)
        return 1

    # Read one frame to get size
    ok, frame = cap.read()
    if not ok:
        print("Error: cannot read from camera", file=sys.stderr)
        return 1
    h, w = frame.shape[:2]

    cam = default_camera_model(w, h)

    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICTS[args.dict])
    params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    print("Press 'q' to quit.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)

        if ids is not None and len(ids) > 0:
            # Draw boxes + ids
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            if not args.no_pose:
                # Estimate pose (rvec, tvec for each marker)
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, args.marker_length, cam.K, cam.dist
                )

                for i in range(len(ids)):
                    rvec = rvecs[i]
                    tvec = tvecs[i]

                    # Draw axes on the marker
                    cv2.drawFrameAxes(frame, cam.K, cam.dist, rvec, tvec, args.marker_length * 0.5)

                    # Optional: show distance (meters)
                    dist_m = float(np.linalg.norm(tvec))
                    tag_id = int(ids[i][0])
                    # Put text near first corner
                    pt = corners[i][0][0].astype(int)
                    cv2.putText(frame, f"id {tag_id}  {dist_m:.2f}m",
                                (pt[0], pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, f"id {tag_id}  {dist_m:.2f}m",
                                (pt[0], pt[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("ArUco Demo", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())