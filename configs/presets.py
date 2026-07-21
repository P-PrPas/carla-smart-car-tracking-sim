"""Dataset generation presets for the Honda Smart Car Tracking POC.

Everything that used to be hard-coded inside
``scripts/generate_carla_dataset_aaa.py`` for the one dataset we shipped now
lives here as a single named preset. A teammate can:

  * read/edit one file to change map, weather, camera angles, FOVs, the CCTV
    sensor look, the vehicle model, car colours, resolution, fps, etc.
  * reproduce the exact dataset we already shipped with ``--preset aaa``
    (the default), or
  * copy the ``aaa`` block, rename it, tweak values, and register it in
    ``PRESETS`` to produce a new dataset variant.

Only *tuning* values live here. Structural POC logic (route topology,
GOOD/DEFECT status rule, parking-slot assignment) stays in the generator
script because changing it is a code change, not a config change.

Values are plain Python literals so nested dicts/tuples (camera_plan, colours)
stay readable and type-checked with zero extra dependencies. Run this file
directly (``python configs/presets.py``) to self-check the ``aaa`` preset.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConfig:
    """One named, reproducible dataset-generation setup."""

    name: str

    # --- output ---------------------------------------------------------
    output_dir: str
    docs_dir: str

    # --- scene ----------------------------------------------------------
    carla_map: str
    # Passed to carla.WeatherParameters via setattr, guarded by hasattr so a
    # slightly different CARLA build silently ignores unknown keys.
    weather: dict[str, float]

    # --- render ---------------------------------------------------------
    fps: int
    duration_sec: float
    width: int
    height: int
    random_seed: int
    num_cars: int

    # --- CCTV post-process (Python-side, in the generator) --------------
    cctv_grain_strength: float
    temporal_blend_strength: float

    # --- cameras --------------------------------------------------------
    # Per-camera placement relative to its route target: yaw_offset (deg),
    # distance (m) back from target, height (m) above target.
    camera_plan: dict[str, dict[str, float]]
    # Per-camera horizontal FOV (deg). Must list every camera id.
    camera_fovs: dict[str, float]
    # CARLA sensor.camera.rgb attributes that shape the cinematic CCTV look.
    rgb_sensor_attributes: dict[str, str]

    # --- vehicles -------------------------------------------------------
    # Blueprint id substrings tried in order; first match in the CARLA
    # library wins, else the first available vehicle is used.
    vehicle_blueprints: tuple[str, ...]
    # BGR colours cycled across cars (all near-white: identical-looking cars
    # is the whole point of the re-id task).
    car_colors_bgr: tuple[tuple[int, int, int], ...]
    # speed_factor for car i = base + (i % cycle) * step
    speed_factor_base: float
    speed_factor_step: float
    speed_factor_cycle: int
    # Per-car start delay is drawn uniformly from this (min, max) seconds.
    start_offset_range_sec: tuple[float, float]


AAA = DatasetConfig(
    name="carla_honda_poc_aaa",
    output_dir="datasets/carla_honda_poc_aaa",
    docs_dir="docs/aaa",
    carla_map="Town05_Opt",
    weather={
        "cloudiness": 32.0,
        "precipitation": 0.0,
        "sun_altitude_angle": 28.0,
        "sun_azimuth_angle": 128.0,
        "fog_density": 1.0,
        "wetness": 10.0,
        "fog_distance": 115.0,
        "fog_falloff": 0.12,
        "precipitation_deposits": 0.0,
        "wind_intensity": 4.0,
        "scattering_intensity": 0.85,
        "mie_scattering_scale": 0.015,
        "rayleigh_scattering_scale": 0.0331,
        "dust_storm": 0.0,
    },
    fps=20,
    duration_sec=60.0,
    width=1920,
    height=1080,
    random_seed=7,
    num_cars=6,
    cctv_grain_strength=1.35,
    temporal_blend_strength=0.10,
    camera_plan={
        "CAM_01_START": {"yaw_offset": -125.0, "distance": 18.0, "height": 9.0},
        "CAM_02_TRANSIT": {"yaw_offset": -160.0, "distance": 22.0, "height": 13.0},
        "CAM_03_JUNCTION_STATUS": {"yaw_offset": -48.0, "distance": 30.0, "height": 17.0},
        "CAM_04_GOOD_ROUTE": {"yaw_offset": -82.0, "distance": 20.0, "height": 13.0},
        "CAM_05_DEFECT_ROUTE": {"yaw_offset": 82.0, "distance": 20.0, "height": 13.0},
        "CAM_06_GOOD_PARKING": {"yaw_offset": 145.0, "distance": 34.0, "height": 15.0},
        "CAM_07_DEFECT_PARKING": {"yaw_offset": 112.0, "distance": 34.0, "height": 15.0},
    },
    camera_fovs={
        "CAM_01_START": 85.0,
        "CAM_02_TRANSIT": 78.0,
        "CAM_03_JUNCTION_STATUS": 66.0,
        "CAM_04_GOOD_ROUTE": 85.0,
        "CAM_05_DEFECT_ROUTE": 85.0,
        "CAM_06_GOOD_PARKING": 72.0,
        "CAM_07_DEFECT_PARKING": 85.0,
    },
    rgb_sensor_attributes={
        "enable_postprocess_effects": "True",
        "exposure_mode": "manual",
        "exposure_compensation": "-0.25",
        "gamma": "2.0",
        "motion_blur_intensity": "0.12",
        "lens_flare_intensity": "0.04",
        "bloom_intensity": "0.12",
        "chromatic_aberration_intensity": "0.06",
        "chromatic_aberration_offset": "0.0",
        "iso": "100.0",
        "shutter_speed": "180.0",
        "fstop": "6.3",
    },
    vehicle_blueprints=("vehicle.lincoln.mkz_2020", "vehicle.tesla.model3"),
    car_colors_bgr=(
        (210, 210, 205),
        (190, 195, 200),
        (225, 225, 220),
        (175, 185, 195),
        (205, 205, 215),
        (185, 190, 185),
        (220, 215, 205),
        (200, 205, 210),
        (170, 180, 190),
        (230, 230, 225),
        (195, 198, 205),
        (212, 216, 218),
    ),
    speed_factor_base=1.18,
    speed_factor_step=0.06,
    speed_factor_cycle=5,
    start_offset_range_sec=(3.0, 5.0),
)


# Register presets here. Add new dataset variants by copying AAA, editing, and
# giving it a new key.
PRESETS: dict[str, DatasetConfig] = {
    "aaa": AAA,
}


def get_preset(name: str) -> DatasetConfig:
    try:
        return PRESETS[name]
    except KeyError:
        known = ", ".join(sorted(PRESETS))
        raise ValueError(f"Unknown preset {name!r}. Known presets: {known}") from None


def _self_check() -> None:
    # Locks the shipped-dataset values so an accidental edit is caught.
    cfg = get_preset("aaa")
    assert (cfg.fps, cfg.width, cfg.height) == (20, 1920, 1080)
    assert cfg.carla_map == "Town05_Opt"
    assert cfg.weather["cloudiness"] == 32.0 and cfg.weather["sun_altitude_angle"] == 28.0
    assert cfg.camera_plan["CAM_03_JUNCTION_STATUS"]["distance"] == 30.0
    assert cfg.camera_fovs["CAM_03_JUNCTION_STATUS"] == 66.0
    assert set(cfg.camera_plan) == set(cfg.camera_fovs)
    assert cfg.vehicle_blueprints[0] == "vehicle.lincoln.mkz_2020"
    assert len(cfg.car_colors_bgr) >= cfg.num_cars
    assert get_preset("aaa") is AAA
    try:
        get_preset("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("get_preset should reject unknown names")
    print("presets self-check OK")


if __name__ == "__main__":
    _self_check()
