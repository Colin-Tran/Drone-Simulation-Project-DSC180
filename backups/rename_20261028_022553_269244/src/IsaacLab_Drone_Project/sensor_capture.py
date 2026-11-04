
"""Record onboard drone camera and IMU data."""

import csv
from pathlib import Path

import numpy as np
from PIL import Image

OUTPUT = Path.cwd() / "outputs"
MAX_FRAMES = 100


def capture_sensors(env, env_ids):
    """Called periodically by the Isaac Lab event manager."""
    frame = getattr(capture_sensors, "frame", 0)

    if frame >= MAX_FRAMES:
        return

    camera = env.scene["front_camera"]
    imu = env.scene["imu"]

    if not camera.data.output:
        return

    rgb = camera.data.output["rgb"].torch[0].cpu().numpy()
    depth = (
        camera.data.output["distance_to_image_plane"]
        .torch[0].cpu().numpy().squeeze()
    )

    acceleration = imu.data.lin_acc_b.torch[0].cpu().numpy()
    angular_velocity = imu.data.ang_vel_b.torch[0].cpu().numpy()

    for folder in ("rgb", "depth"):
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)

    name = f"frame_{frame:04d}"

    # Save color image
    Image.fromarray(rgb[..., :3].astype(np.uint8)).save(
        OUTPUT / "rgb" / f"{name}.png"
    )

    # Save the original numerical depth measurements
    np.save(OUTPUT / "depth" / f"{name}.npy", depth)

    # Save IMU measurements
    csv_path = OUTPUT / "imu.csv"
    write_header = not csv_path.exists()

    with csv_path.open("a", newline="") as file:
        writer = csv.writer(file)

        if write_header:
            writer.writerow([
                "time_s",
                "acc_x", "acc_y", "acc_z",
                "gyro_x", "gyro_y", "gyro_z",
            ])

        writer.writerow([
            env.common_step_counter * env.step_dt,
            *acceleration.tolist(),
            *angular_velocity.tolist(),
        ])

    capture_sensors.frame = frame + 1

    if frame % 10 == 0:
        print(f"[SENSOR] Captured frame {frame}")
