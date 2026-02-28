# Datasets

Download these datasets in **YOLO26 format** from Roboflow and place each in this directory before running `uv run data.py`.

| Dataset | Classes | Link |
|---------|---------|------|
| Drone Top View | vehicles | [Roboflow](https://universe.roboflow.com/yolo-training-uygwi/drone-top-view-emuop) |
| UAE With Building 2 | Building, Door, Grass, Roof, Stairs, Tree, Under_Construction, Wall, Windows | [Roboflow](https://universe.roboflow.com/labelling-a5t57/uae_with_building-2) |
| UAE With Building 2_2 | Building, Door, Grass, Roof, Stairs, Tree, Under_Construction, Walls, Windows | [Roboflow](https://universe.roboflow.com/labelling-a5t57/uae_with_building-2_2) |
| UAE Without Building | Advertisement, Dustbin, Road_Sign, Speed_limit, Street_light, Traffic_Signal, Zebra_Crossing | [Roboflow](https://universe.roboflow.com/labelling-a5t57/uae_without_building) |
| Road Assets | — | [Roboflow](https://universe.roboflow.com/labelling-a5t57/road_assets-kq7gx) |
| New Person Posture | — | [Roboflow](https://universe.roboflow.com/labelling-a5t57/new_person_posture) |

## Directory Structure

After downloading, the directory should look like:

```
data/
├── Drone top view.v1i.yolo26/
├── UAE_With_Building 2.v2i.yolo26/
├── UAE_With_Building 2_2.v2i.yolo26/
├── UAE_Without_Building.v10i.yolo26/
└── ...
```

Then run `uv run data.py` to merge everything into `data/unified/`.
