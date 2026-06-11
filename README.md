# Honda Smart Car Tracking System - CARLA Dataset POC

Synthetic video dataset generator สำหรับทำ POC ระบบ **Honda Smart Car Tracking System**
ตาม proposal ใน [docs/Proposal_v1_0_Honda_Smart_Car_Tracking_System.md](docs/Proposal_v1_0_Honda_Smart_Car_Tracking_System.md)

โปรเจคนี้ใช้ **CARLA Simulator 0.9.15** เพื่อสร้างวิดีโอ 3D จากมุมกล้อง CCTV เสมือนจริง สำหรับทดสอบงาน vehicle identification, cross-camera tracking, status detection และ parking slot detection ในลานจอดรถใหม่

> Dataset final ต้องสร้างด้วย `--renderer carla` เท่านั้น  
> `--renderer storyboard` มีไว้สำหรับ dry-run/debug metadata บนเครื่องที่ยังไม่มี CARLA

---

## Overview

โจทย์ของระบบคือการติดตามรถแต่ละคันตั้งแต่ออกจากไลน์ผลิตจนถึงจุดจอด โดยรถไม่มีป้ายทะเบียนและมีลักษณะใกล้เคียงกันมาก จึงต้องสร้าง unique tracking flow จากข้อมูลภาพและ metadata จำลอง

Pipeline ของ dataset:

```text
Start OCR Camera
  -> Transit Camera
  -> Junction Status Camera
      -> Good Route Camera
          -> Good Parking Camera
      -> Defect Route Camera
          -> Defect Parking Camera
```

สิ่งที่ dataset ต้องรองรับ:

- อ่าน `oil_tank_id` จากป้ายกระดาษเลข 6 หลักบนกระจกหน้ารถ
- track รถข้ามกล้องหลายตัว
- จำแนกสถานะรถจากทางแยก
  - เลี้ยวซ้าย = `GOOD`
  - เลี้ยวขวา = `DEFECT`
- ระบุ parking slot สุดท้าย เช่น `G01`, `G02`, `D01`
- มี metadata สำหรับ validate detector/tracker/OCR/parking logic

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── scripts/
│   └── generate_carla_dataset.py
├── docs/
│   ├── Proposal_v1_0_Honda_Smart_Car_Tracking_System.md
│   ├── carla_honda_poc_dataset.md
│   └── carla_honda_poc_map.png
└── datasets/
    └── carla_honda_poc/
        ├── videos/
        ├── metadata/
        └── annotations/
```

หมายเหตุ: `datasets/` ถูก ignore ด้วย `.gitignore` เพราะวิดีโอและ annotations เป็น generated artifacts

---

## Dataset Design

### Cameras

Dataset ใช้กล้อง fixed CCTV ทั้งหมด 7 ตัว

| Camera ID | Purpose |
|---|---|
| `CAM_01_START_OCR` | จุดเริ่มต้น อ่านเลข `oil_tank_id` 6 หลักบนกระจกหน้ารถ |
| `CAM_02_TRANSIT` | เส้นทางหลังออกจาก start ใช้ทดสอบ cross-camera tracking |
| `CAM_03_JUNCTION_STATUS` | ทางแยกสำหรับจำแนกสถานะรถ |
| `CAM_04_GOOD_ROUTE` | เส้นทางหลังแยกของรถสถานะ `GOOD` |
| `CAM_05_DEFECT_ROUTE` | เส้นทางหลังแยกของรถสถานะ `DEFECT` |
| `CAM_06_GOOD_PARKING` | ลานจอดรถดี มี slot `G01-G06` |
| `CAM_07_DEFECT_PARKING` | ลานจอดรถเสีย มี slot `D01-D04` |

### Vehicle Identity

- `tracking_id`: internal tracking ID เช่น `TRK_0001`
- `oil_tank_id`: เลขล้วน 6 หลัก เช่น `100001`
- `status`: `GOOD` หรือ `DEFECT`
- `parking_slot_id`: slot ปลายทาง เช่น `G01` หรือ `D01`

### CARLA Rendering

เมื่อใช้ `--renderer carla` script จะ:

- เชื่อมต่อ CARLA server ที่ `127.0.0.1:2000`
- โหลด map `Town05`
- เปิด synchronous mode
- ตั้ง `fixed_delta_seconds = 1 / fps`
- spawn รถเป็น CARLA vehicle actors
- spawn กล้องเป็น fixed `sensor.camera.rgb` actors
- render วิดีโอจาก sensor จริงของ CARLA
- project 3D vehicle bounding box ลงบนภาพ 2D
- overlay ป้าย `oil_tank_id` 6 หลักบริเวณกระจกหน้ารถสำหรับกล้อง OCR

---

## Output Files

หลัง generate dataset จะได้ไฟล์เหล่านี้

```text
datasets/carla_honda_poc/
├── videos/
│   ├── CAM_01_START_OCR.mp4
│   ├── CAM_02_TRANSIT.mp4
│   ├── CAM_03_JUNCTION_STATUS.mp4
│   ├── CAM_04_GOOD_ROUTE.mp4
│   ├── CAM_05_DEFECT_ROUTE.mp4
│   ├── CAM_06_GOOD_PARKING.mp4
│   └── CAM_07_DEFECT_PARKING.mp4
├── metadata/
│   ├── cars.csv
│   ├── camera_graph.json
│   └── events.jsonl
└── annotations/
    └── bboxes.jsonl
```

และเอกสาร map:

```text
docs/carla_honda_poc_map.png
docs/carla_honda_poc_dataset.md
```

### Metadata Fields

`annotations/bboxes.jsonl` มีข้อมูลต่อ frame เช่น:

- `tracking_id`
- `oil_tank_id`
- `status`
- `camera_id`
- `frame_id`
- `timestamp_sec`
- `bbox`
- `bbox_source`
- `ocr_bbox`
- `parking_slot_id`
- `vehicle_actor_id`
- `camera_transform`
- `world_transform`

---

## Prerequisites

### Hardware

แนะนำ VM หรือ workstation ที่มี:

- NVIDIA GPU
- VRAM อย่างน้อย 8 GB
- Docker พร้อม NVIDIA Container Toolkit
- พื้นที่ว่างสำหรับ dataset output

โปรเจคนี้เคยทดสอบแนวทางบน VM ที่มี Tesla V100 16 GB

### Software

- Linux
- Docker
- NVIDIA driver ที่รองรับ graphics/Vulkan
- NVIDIA Container Toolkit
- Conda หรือ Python virtual environment
- Python 3.10 สำหรับ CARLA 0.9.15 client
- CARLA Docker image `carlasim/carla:0.9.15`

> ไม่แนะนำใช้ Python 3.13 กับ CARLA 0.9.15 client เพราะ wheel/API compatibility มักมีปัญหา

---

## Installation

### 1. Clone Project

```bash
git clone <repo-url>
cd poc-honda
```

หรือถ้าอยู่ใน workspace นี้แล้ว:

```bash
cd ~/volume/project/poc-honda
```

### 2. Prepare Python Environment

สร้าง conda environment สำหรับ CARLA client:

```bash
conda create -n carla-0915 python=3.10 -y
conda activate carla-0915
```

ติดตั้ง Python dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

ถ้า `pip install carla==0.9.15` มีปัญหา ให้ติดตั้ง CARLA Python API จาก package ที่มากับ CARLA release แทน โดยต้องใช้ egg/wheel ที่ตรงกับ Python version

### 3. Pull CARLA Docker Image

```bash
docker pull carlasim/carla:0.9.15
```

### 4. Verify NVIDIA Docker

```bash
docker run --rm --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

ถ้าคำสั่งนี้ไม่ผ่าน ให้แก้ NVIDIA Container Toolkit หรือ driver ก่อน

---

## Running CARLA Server

เปิด terminal แรก แล้วรัน CARLA server:

```bash
docker run --rm -it \
  --privileged \
  --gpus '"device=0"' \
  --net=host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display \
  carlasim/carla:0.9.15 \
  /bin/bash ./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-rpc-port=2000
```

ใช้ `-RenderOffScreen` สำหรับ VM/headless server

ห้ามใช้ `-no-rendering` สำหรับงานนี้ เพราะ camera/GPU sensors จะไม่มีภาพ ทำให้สร้าง video dataset ไม่ได้

---

## Verify CARLA Client Connection

เปิด terminal ที่สอง:

```bash
conda activate carla-0915
cd ~/volume/project/poc-honda
```

ทดสอบเชื่อมต่อ CARLA:

```bash
python - <<'PY'
import carla

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(10.0)

print("client:", client.get_client_version())
print("server:", client.get_server_version())

world = client.load_world("Town05")
print("map:", world.get_map().name)
PY
```

ต้องเห็นว่าโหลด `Town05` ได้สำเร็จ

---

## Generate Dataset

### Generate Final 3D CCTV Dataset

ใช้คำสั่งนี้เมื่อ CARLA server เปิดอยู่:

```bash
python scripts/generate_carla_dataset.py --renderer carla --clean
```

ค่า default:

- `num-cars`: 10
- `fps`: 10
- `duration-sec`: 45
- `width`: 1280
- `height`: 720
- `oil-start`: 100001
- `carla-host`: 127.0.0.1
- `carla-port`: 2000

### Generate With Custom Settings

```bash
python scripts/generate_carla_dataset.py \
  --renderer carla \
  --num-cars 12 \
  --fps 10 \
  --duration-sec 60 \
  --width 1280 \
  --height 720 \
  --oil-start 100001 \
  --clean
```

### Dry-Run Without CARLA

ใช้สำหรับเช็ก metadata/camera graph เท่านั้น ไม่ใช่ final dataset:

```bash
python scripts/generate_carla_dataset.py --renderer storyboard --clean
```

---

## CLI Reference

```bash
python scripts/generate_carla_dataset.py --help
```

Options:

| Option | Default | Description |
|---|---:|---|
| `--output-dir` | `datasets/carla_honda_poc` | output directory |
| `--docs-dir` | `docs` | directory สำหรับ map/doc generated files |
| `--renderer` | `storyboard` | `carla` สำหรับ final dataset, `storyboard` สำหรับ dry-run |
| `--num-cars` | `10` | จำนวนรถ ต้องอยู่ระหว่าง 8-12 |
| `--fps` | `10` | frame rate ของ video output |
| `--duration-sec` | `45` | ความยาววิดีโอต่อกล้อง |
| `--width` | `1280` | video width |
| `--height` | `720` | video height |
| `--oil-start` | `100001` | เลขเริ่มต้นของ oil tank ID |
| `--carla-host` | `127.0.0.1` | CARLA server host |
| `--carla-port` | `2000` | CARLA RPC port |
| `--clean` | off | ลบ output directory เดิมก่อนสร้างใหม่ |

---

## Validation Checklist

หลัง generate dataset ควรตรวจดังนี้:

```bash
find datasets/carla_honda_poc -maxdepth 3 -type f | sort
```

เช็กวิดีโอ:

```bash
python - <<'PY'
import cv2
from pathlib import Path

for path in sorted(Path("datasets/carla_honda_poc/videos").glob("*.mp4")):
    cap = cv2.VideoCapture(str(path))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(path.name, frames, fps, f"{width}x{height}")
PY
```

เช็ก oil tank ID:

```bash
python - <<'PY'
import csv
import re

with open("datasets/carla_honda_poc/metadata/cars.csv") as fh:
    cars = list(csv.DictReader(fh))

assert all(re.fullmatch(r"\d{6}", car["oil_tank_id"]) for car in cars)
assert len({car["oil_tank_id"] for car in cars}) == len(cars)
print("oil_tank_id validation ok:", [car["oil_tank_id"] for car in cars])
PY
```

เช็ก annotations:

```bash
python - <<'PY'
import json

count = 0
with open("datasets/carla_honda_poc/annotations/bboxes.jsonl") as fh:
    for line in fh:
        row = json.loads(line)
        x1, y1, x2, y2 = row["bbox"]
        assert x1 < x2 and y1 < y2
        count += 1

print("bbox records:", count)
PY
```

---

## Troubleshooting

### `ERROR: CARLA Python API is not installed`

เกิดจาก Python environment ไม่มี module `carla`

แก้โดยใช้ Python 3.10 environment และติดตั้ง:

```bash
conda activate carla-0915
pip install carla==0.9.15
```

ถ้ายังไม่ได้ ให้ใช้ Python API egg/wheel ที่มากับ CARLA 0.9.15 release

### Docker error: `/dev/nvidia-modeset: no such file or directory`

แปลว่า host มี NVIDIA compute device แต่ graphics stack ยังไม่ครบสำหรับ CARLA rendering

เช็ก:

```bash
nvidia-smi
ls -l /dev/nvidia*
lsmod | grep nvidia
vulkaninfo --summary
```

ควรเห็น:

```text
/dev/nvidia-modeset
nvidia_modeset
nvidia_drm
```

บน Debian ต้องเปิด repo `non-free non-free-firmware` ก่อนติดตั้ง driver:

```text
deb http://deb.debian.org/debian trixie main contrib non-free non-free-firmware
deb http://deb.debian.org/debian trixie-updates main contrib non-free non-free-firmware
deb http://deb.debian.org/debian-security/ trixie-security main contrib non-free non-free-firmware
```

จากนั้น:

```bash
sudo apt update
sudo apt install -y linux-headers-amd64 build-essential dkms
sudo apt install -y nvidia-driver firmware-misc-nonfree libvulkan1 vulkan-tools
sudo reboot
```

### `apt update` fail เพราะ NVIDIA mirror 404

ถ้าเจอ repo ลักษณะนี้ fail:

```text
dist.bsthun.com/mirror/apt/nvidia
```

ให้ disable ก่อน:

```bash
sudo mv /etc/apt/sources.list.d/bsthun-mirror.list \
  /etc/apt/sources.list.d/bsthun-mirror.list.disabled

sudo apt update
```

### CARLA server เปิดได้ แต่ video เป็นภาพว่าง

ตรวจว่าไม่ได้ใช้ `-no-rendering`

สำหรับงานนี้ต้องใช้:

```bash
-RenderOffScreen
```

ไม่ใช่:

```bash
-no-rendering
```

### CARLA connection timeout

เช็กว่า server ยังรันอยู่และ port ถูกต้อง:

```bash
ss -lntp | grep 2000
```

ถ้าใช้ Docker แบบไม่ใช่ `--net=host` ต้อง map port เอง แต่แนะนำใช้ `--net=host` ตามคำสั่งใน README นี้

---

## Development Notes

### Renderer Modes

| Renderer | Use Case | Output Type |
|---|---|---|
| `carla` | final dataset | 3D CCTV render จาก CARLA RGB sensors |
| `storyboard` | dry-run/debug | ภาพ 2D schematic สำหรับเช็ก metadata เท่านั้น |

### Why Projected Overlay For Oil Tank ID?

รอบ POC ใช้วิธี overlay เลข 6 หลักลงบนบริเวณกระจกหน้ารถ เพื่อควบคุม OCR target ให้ชัดและ reproducible โดยยังคงใช้ฉาก รถ และกล้องจาก CARLA 3D จริง

ถ้าต้องการความสมจริงสูงขึ้นในอนาคต สามารถต่อยอดเป็น custom Unreal/CARLA asset สำหรับป้ายกระดาษจริงได้

---

## References

- CARLA Simulator GitHub: https://github.com/carla-simulator/carla
- CARLA 0.9.15 documentation: https://carla.readthedocs.io/en/0.9.15/
- CARLA sensors reference: https://carla.readthedocs.io/en/0.9.15/ref_sensors/
- CARLA rendering options: https://carla.readthedocs.io/en/0.9.15/adv_rendering_options/
- Debian NVIDIA driver guide: https://wiki.debian.org/NvidiaGraphicsDrivers

