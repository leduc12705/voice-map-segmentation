# Voice Map Segmentation

Dieu khien robot bang giong noi tren ban do da phan vung (ROS 2 + Vosk).

## Cau truc
- `voice_nav/`: package ROS 2 (nhan dang giong noi, phan tich lenh, dieu huong)
- `tools/segment_map.py`: phan vung ban do
- `maps/`: ban do goc (maze.pgm, maze.yaml)
- `zones/`: ban do phan vung (zones_labeled.png) va thong tin cac khu vuc (zones.yaml)
- `example/`: ban do mau va diem seed
- `models/`: model Vosk (tieng Anh va tieng Viet), da kem san

## Cai dat va chay
```bash
git clone https://github.com/leduc12705/voice-map-segmentation.git
cd voice-map-segmentation
pip install vosk
mkdir -p ~/ros2_ws/src && cp -r voice_nav ~/ros2_ws/src/
cd ~/ros2_ws && colcon build --packages-select voice_nav
source install/setup.bash
ros2 launch voice_nav voice_nav.launch.py
```
