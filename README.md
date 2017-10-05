# deeptracking

Torch + python implementation of [Deep 6-DOF Tracking](https://arxiv.org/abs/1703.09771)


## Generating data
### Generating synthetic data
Run:
```bash
python generate_synthetic_data.py config_file.json
```

#### dependencies
- cv2
- tqdm
- pyOpenGL
- glfw
- plyfile
- numpngw

#### configuration
see this [example file](https://github.com/lvsn/deeptracking/blob/develop/configs/generate_synthetic_example.json)

### Generating real data
Run:
```bash
python generate_real_data.py config_file.json
```

#### dependencies
- cv2
- tqdm
- pyOpenGL
- glfw
- plyfile
- numpngw

#### configuration
see this [example file](https://github.com/lvsn/deeptracking/blob/develop/configs/generate_real_example.json)

## Train
Run:
```bash
python train.py config_file.json
```

#### dependencies
- Hugh Perkins's [pytorch](https://github.com/hughperkins/pytorch)
- scipy, skimage, numpy
- tqdm
- plyfile
- numpngw
- slackclient (could be removed)

#### configuration
see this [example file](https://github.com/lvsn/deeptracking/blob/develop/configs/train_example.json)

## Test
#### Sensor
Will run the tracker with a sensor (kinect 2)
```bash
python test_sensor.py config_file.json
```

#### Sequence
Will run the tracker on a sequence (folder) and save the error in a file
```bash
python test_sequence.py config_file.json
```

#### dependencies
- cv2
- Hugh Perkins's [pytorch](https://github.com/hughperkins/pytorch)
- pyOpenGL
- glfw
- plyfile
- numpngw
- [pyfreenect2](https://github.com/MathGaron/py3freenect2) (for Kinect2 sensor)

#### configuration
see this [example file](https://github.com/lvsn/deeptracking/blob/develop/configs/test_example.json)
