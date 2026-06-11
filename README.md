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
Start Camera
  -> Transit Camera
  -> Junction Status Camera
      -> Good Route Camera
          -> Good Parking Camera
      -> Defect Route Camera
          -> Defect Parking Camera
```

สิ่งที่ dataset ต้องรองรับ:

- track รถข้ามกล้องหลายตัว
- จำแนกสถานะรถจากทางแยก
  - เลี้ยวซ้าย = `GOOD`
  - เลี้ยวขวา = `DEFECT`
- ระบุ parking slot สุดท้าย เช่น `G01`, `G02`, `D01`
- มี metadata สำหรับ validate detector/tracker/status/parking logic

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
| `CAM_01_START` | จุดเริ่มต้น ตรวจจับรถที่ออกจากไลน์ผลิต |
| `CAM_02_TRANSIT` | เส้นทางหลังออกจาก start ใช้ทดสอบ cross-camera tracking |
| `CAM_03_JUNCTION_STATUS` | ทางแยกสำหรับจำแนกสถานะรถ |
| `CAM_04_GOOD_ROUTE` | เส้นทางหลังแยกของรถสถานะ `GOOD` |
| `CAM_05_DEFECT_ROUTE` | เส้นทางหลังแยกของรถสถานะ `DEFECT` |
| `CAM_06_GOOD_PARKING` | ลานจอดรถดี มี slot `G01-G06` |
| `CAM_07_DEFECT_PARKING` | ลานจอดรถเสีย มี slot `D01-D04` |

### Vehicle Identity

- `tracking_id`: internal tracking ID เช่น `TRK_0001`
- `status`: `GOOD` หรือ `DEFECT`
- `parking_slot_id`: slot ปลายทาง เช่น `G01` หรือ `D01`

### CARLA Rendering

เมื่อใช้ `--renderer carla` script จะ:

- เชื่อมต่อ CARLA server ที่ `127.0.0.1:2000`
- โหลด map `Town05_Opt` และ fallback เป็น `Town05` ถ้า `_Opt` ไม่มี
- เปิด synchronous mode
- ตั้ง `fixed_delta_seconds = 1 / fps`
- spawn รถเป็น CARLA vehicle actors
- spawn กล้องเป็น fixed `sensor.camera.rgb` actors
- render วิดีโอจาก sensor จริงของ CARLA
- project 3D vehicle bounding box ลงบนภาพ 2D
- OCR ถูกปิดไว้ในรอบ POC นี้ เพื่อโฟกัส vehicle tracking และ parking ก่อน

---

## Output Files

หลัง generate dataset จะได้ไฟล์เหล่านี้

```text
datasets/carla_honda_poc/
├── videos/
│   ├── CAM_01_START.mp4
│   ├── CAM_02_TRANSIT.mp4
│   ├── CAM_03_JUNCTION_STATUS.mp4
│   ├── CAM_04_GOOD_ROUTE.mp4
│   ├── CAM_05_DEFECT_ROUTE.mp4
│   ├── CAM_06_GOOD_PARKING.mp4
│   └── CAM_07_DEFECT_PARKING.mp4
├── metadata/
│   ├── cars.csv
│   ├── camera_graph.json
│   ├── route_plan.json  # CARLA renderer
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
- `status`
- `camera_id`
- `frame_id`
- `timestamp_sec`
- `bbox`
- `bbox_source`
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
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

ถ้าคำสั่งนี้ไม่ผ่าน ให้แก้ NVIDIA Container Toolkit หรือ driver ก่อน

หมายเหตุ: บน VM/headless server บางเครื่องจะไม่มี `/dev/nvidia-modeset`
ให้หลีกเลี่ยง capability `display` เพราะ NVIDIA runtime จะพยายาม mount device นี้
และทำให้ Docker fail ตั้งแต่ container init

---

## Running CARLA Server

เปิด terminal แรก แล้วรัน CARLA server:

```bash
docker run --rm -it \
  --privileged \
  --gpus '"device=0"' \
  --net=host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  carlasim/carla:0.9.15 \
  /bin/bash ./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-rpc-port=2000 -nosound -NoSound
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

world = client.load_world("Town05_Opt")
print("map:", world.get_map().name)
PY
```

ต้องเห็นว่าโหลด `Town05_Opt` ได้สำเร็จ ถ้า image ไม่มี `_Opt` ให้ใช้ `Town05`

---

## Generate Dataset

### Generate Final 3D CCTV Dataset

ใช้คำสั่งนี้เมื่อ CARLA server เปิดอยู่:

```bash
python scripts/generate_carla_dataset.py --renderer carla --clean
```

ค่า default:

- `fps`: 10
- `num-cars`: 6
- `duration-sec`: 60
- `width`: 1280
- `height`: 720
- `carla-map`: Town05_Opt
- `carla-host`: 127.0.0.1
- `carla-port`: 2000

### Generate With Custom Settings

```bash
python scripts/generate_carla_dataset.py \
  --renderer carla \
  --num-cars 6 \
  --fps 10 \
  --duration-sec 60 \
  --width 1280 \
  --height 720 \
  --write-contact-sheets \
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
| `--num-cars` | `6` | จำนวนรถ ต้องอยู่ระหว่าง 5-6 |
| `--fps` | `10` | frame rate ของ video output |
| `--duration-sec` | `60` | ความยาววิดีโอต่อกล้อง |
| `--width` | `1280` | video width |
| `--height` | `720` | video height |
| `--carla-host` | `127.0.0.1` | CARLA server host |
| `--carla-port` | `2000` | CARLA RPC port |
| `--carla-map` | `Town05_Opt` | CARLA map ที่ใช้สร้าง POC; fallback เป็น `Town05` ถ้า `_Opt` ไม่มี |
| `--carla-timeout-sec` | `120` | timeout สำหรับ CARLA RPC เช่น load/get world บน server ที่โหลดช้า |
| `--random-seed` | `7` | seed สำหรับ start delay แบบ deterministic |
| `--camera-ids` | all | comma-separated camera IDs สำหรับ render เฉพาะบางกล้อง เช่น resume หลัง CARLA crash |
| `--append-annotations` | off | append `annotations/bboxes.jsonl` แทนการเขียนใหม่ ใช้คู่กับ `--camera-ids` ตอน resume |
| `--write-contact-sheets` | off | สร้างภาพ sample frames ต่อกล้องไว้ที่ `docs/video_contact_sheets/` สำหรับตรวจมุมกล้อง |
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
หรือ Docker command ขอ `NVIDIA_DRIVER_CAPABILITIES=display` บนเครื่อง headless ที่ไม่มี display device

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

ถ้าใช้ VM/headless server และคำสั่งนี้ผ่าน:

```bash
docker run --rm --gpus all \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

ให้รัน CARLA โดยไม่ใส่ `display`:

```bash
-e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
```

ถ้ายังต้องใช้ `display` จริง ๆ ต้องแก้ driver/module ฝั่ง host ให้มี `/dev/nvidia-modeset`
ก่อน ไม่ใช่แก้ใน container

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

### CARLA error: `xdg-user-dir: not found`

CARLA Docker image อาจไม่มี `xdg-user-dir` ทำให้ Unreal Engine หยุดตั้งแต่เริ่ม
ให้ทดสอบด้วย wrapper ที่สร้าง fallback command ให้ container:

```bash
docker run --rm -it \
  --privileged \
  --gpus '"device=0"' \
  --net=host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --entrypoint /bin/bash \
  carlasim/carla:0.9.15 \
  -lc 'mkdir -p /tmp/carla-bin;
       printf '"'"'#!/bin/sh\nprintf "%s\\n" "${HOME:-/home/carla}"\n'"'"' > /tmp/carla-bin/xdg-user-dir;
       chmod +x /tmp/carla-bin/xdg-user-dir;
       export PATH=/tmp/carla-bin:$PATH;
       ./CarlaUE4.sh -RenderOffScreen -quality-level=Low -carla-rpc-port=2000 -nosound -NoSound'
```

บน headless server ให้ใส่ `-nosound -NoSound` เสมอ เพราะ Unreal Engine อาจ crash
หลัง ALSA หา default audio device ไม่เจอ

ถ้ายัง exit ทันที ให้เช็ก Vulkan/NVIDIA userspace libs ใน container:

```bash
docker run --rm --gpus '"device=0"' \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  --entrypoint /bin/bash \
  carlasim/carla:0.9.15 \
  -lc 'ls -l /etc/vulkan/icd.d;
       ls -l /usr/lib/x86_64-linux-gnu/libGLX_nvidia.so* 2>/dev/null || true'
```

ถ้าไม่มี `libGLX_nvidia.so.0` หรือ `vulkaninfo` รายงาน
`VK_ERROR_INCOMPATIBLE_DRIVER` แปลว่า host มี NVIDIA compute driver
แต่ยังไม่มี NVIDIA OpenGL/Vulkan userspace libraries ที่ CARLA ต้องใช้
ต้องติดตั้ง NVIDIA driver/GL/Vulkan package ฝั่ง host ให้ครบและ version ตรงกับ kernel driver
ก่อน ไม่ใช่แก้เฉพาะใน container

ถ้า `nvidia-smi` ขึ้น `driver/library version mismatch` หลังติดตั้ง driver:

```text
Failed to initialize NVML: Driver/library version mismatch
NVML library version: 550.163
```

ให้เช็ก kernel module ที่กำลังรัน:

```bash
cat /proc/driver/nvidia/version
dpkg -l | grep -E 'nvidia-driver|libnvidia-ml1|nvidia-vulkan-icd'
```

ถ้า kernel module เป็นคนละ version กับ package เช่น kernel ยังเป็น `580.x`
แต่ package เป็น `550.x` ต้อง reboot ให้ kernel โหลด module version ใหม่
หลังจากแก้ `modprobe/depmod` ให้เป็นของจริงก่อน

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

### Custom Map Phase

รอบนี้ใช้ `Town05_Opt`/`Town05` เป็น Hybrid POC ก่อน เพราะมี road network และ carpark ที่พร้อมใช้งานใน CARLA package

ถ้าต้องการ map ที่ตรงกับ `docs/carla_honda_poc_map.png` แบบ 1:1 ต้องสร้าง custom map ด้วย RoadRunner หรือเครื่องมือ OpenDRIVE/FBX อื่น แล้ว import เข้า CARLA:

- CARLA custom map ต้องมี geometry `.fbx` และ OpenDRIVE `.xodr`
- RoadRunner สามารถ export เป็น CARLA format ได้
- ถ้าใช้ CARLA binary/Docker ต้อง ingest map package แยกก่อนนำมาใช้จริง

---

## References

- CARLA Simulator GitHub: https://github.com/carla-simulator/carla
- CARLA 0.9.15 documentation: https://carla.readthedocs.io/en/0.9.15/
- CARLA sensors reference: https://carla.readthedocs.io/en/0.9.15/ref_sensors/
- CARLA rendering options: https://carla.readthedocs.io/en/0.9.15/adv_rendering_options/
- Debian NVIDIA driver guide: https://wiki.debian.org/NvidiaGraphicsDrivers
