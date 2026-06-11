# CARLA Honda POC Dataset

This dataset supports the Honda Smart Car Tracking System POC described in
`Proposal_v1_0_Honda_Smart_Car_Tracking_System.md`.

The final dataset must be generated with `--renderer carla`. The storyboard
renderer is only a lightweight dry-run for validating metadata and camera graph
logic on machines that do not have CARLA installed.

## Scenario

- CARLA target map: `Town05_Opt` with fallback to `Town05`
- Dataset scale: Focused POC, 5-6 vehicles
- Cameras: 7 fixed virtual CCTV viewpoints rendered by CARLA RGB sensors
- Status rule: at `CAM_03_JUNCTION_STATUS`, left turn is `GOOD` and right turn is `DEFECT`
- Parking slots: `G01-G06` for GOOD vehicles and `D01-D04` for DEFECT vehicles
- Vehicle spawn cooldown: deterministic random 3-5 seconds
- Parking behavior: approach, overshoot, reverse entry, correction, final stop
- Parking lot markings: projected slot lines on parking camera videos
- Bounding boxes: CARLA 3D vehicle bounding boxes projected into each camera plane
- OCR is disabled for this POC iteration while vehicle tracking and parking are validated.

## Generated Files

- `datasets/carla_honda_poc/videos/*.mp4`
- `datasets/carla_honda_poc/metadata/cars.csv`
- `datasets/carla_honda_poc/metadata/camera_graph.json`
- `datasets/carla_honda_poc/metadata/route_plan.json` (CARLA renderer)
- `datasets/carla_honda_poc/metadata/events.jsonl`
- `datasets/carla_honda_poc/annotations/bboxes.jsonl`
- `docs/carla_honda_poc_map.png`

## Generate Locally

Run CARLA 0.9.15 first, then generate the real 3D CCTV dataset:

```bash
python scripts/generate_carla_dataset.py --renderer carla --clean
```

For metadata-only development without CARLA, use the storyboard dry-run:

```bash
python scripts/generate_carla_dataset.py --renderer storyboard --clean
```
